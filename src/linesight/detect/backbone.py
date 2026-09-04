"""Frozen ImageNet feature extractor - the only neural network in the product.

No gradients ever flow through this. It is a fixed function from a fabric tile
to a grid of patch embeddings: ``timm``'s ``resnet18`` (chosen for CPU speed)
with ``layer2`` and ``layer3`` hooked out and concatenated.

Why these two layers: ``layer1`` is too texture-local to distinguish a slub from
grain noise, ``layer4`` is too semantic - it has thrown away the spatial detail
we need to localise a 1 mm defect.

**Grid alignment.** The hooked layers have different strides, so one must be
resampled onto the other's grid before they can be concatenated, and the choice
sizes the whole system. At 320 px input, ``layer2`` has stride 8 (a 40x40 grid)
and ``layer3`` stride 16 (20x20). Aligning to the coarsest gives 400 patches per
tile, which is where the ~12,000-embedding bank, the ~1,200-point coreset, the
~600 KB artifact and ADR-011's 61 MFLOP per tile all come from. Aligning to the
finest localises more sharply and multiplies every one of those by four. So
``grid_align="coarsest"`` is the default and the shipped configuration, and
``"finest"`` is a config key rather than a constant so the trade can be
measured rather than argued.
"""

from __future__ import annotations

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

__all__ = ["FeatureExtractor", "resolve_device"]

_GRID_ALIGN = ("coarsest", "finest")


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn ``"auto" | "cpu" | "cuda"`` into a concrete ``torch.device``."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but no CUDA device is available")
    return torch.device(spec)


class FeatureExtractor:
    """A frozen backbone with forward hooks on the chosen layers.

    ``embed`` is the entry point: tiles in, an (N, G, G, D) grid of patch
    embeddings out. Both dimensions are measured at construction rather than
    assumed, so swapping the backbone needs no arithmetic elsewhere.

    Attributes set after construction:
        embed_dim: summed channel count of the hooked layers (resnet18
            layer2 + layer3 = 128 + 256 = 384).
        grid_size: spatial side of the returned grid at ``input_size``
            (320 with ``grid_align="coarsest"`` -> layer3 stride 16 -> 20).
            Recorded so the bank can assert on it.
    """

    IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        input_size: int = 320,
        device: str | torch.device = "auto",
        grid_align: str = "coarsest",
    ) -> None:
        """Build the model, register hooks, freeze, and put it in eval mode.

        Raises:
            ValueError: if a requested layer name is not a module of the
                backbone, if no layers are requested, or on an unknown
                ``grid_align``.
        """
        if not layers:
            raise ValueError("at least one layer must be hooked")
        if grid_align not in _GRID_ALIGN:
            raise ValueError(f"grid_align must be one of {_GRID_ALIGN}, got {grid_align!r}")

        self.backbone_name = backbone
        self.layers = tuple(layers)
        self.input_size = int(input_size)
        self.grid_align = grid_align
        self.device = (
            device if isinstance(device, torch.device) else resolve_device(str(device))
        )

        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)
        named = dict(self.model.named_modules())
        missing = [name for name in self.layers if name not in named]
        if missing:
            raise ValueError(
                f"{backbone} has no module(s) {missing}. "
                f"Available top-level: {[n for n in named if n and '.' not in n]}"
            )

        self._features: dict[str, torch.Tensor] = {}
        for name in self.layers:
            named[name].register_forward_hook(self._make_hook(name))

        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

        self.embed_dim, self.grid_size = self._probe_shapes()

    def _make_hook(self, name: str):
        def hook(_module: object, _inputs: object, output: torch.Tensor) -> None:
            self._features[name] = output

        return hook

    @torch.no_grad()
    def _probe_shapes(self) -> tuple[int, int]:
        """One forward pass on a blank input, to measure embed_dim and grid_size.

        Derived by measurement rather than by a stride table, so a different
        backbone works without anyone editing an arithmetic comment.
        """
        dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
        self._forward(dummy)
        maps = [self._features[name] for name in self.layers]
        embed_dim = sum(int(m.shape[1]) for m in maps)
        sizes = [int(m.shape[-1]) for m in maps]
        grid = min(sizes) if self.grid_align == "coarsest" else max(sizes)
        return embed_dim, grid

    def _forward(self, batch: torch.Tensor) -> None:
        """Run the feature trunk only; the hooks capture the layers of interest."""
        self._features.clear()
        self.model.forward_features(batch)

    # -- preprocessing ------------------------------------------------------ #

    def preprocess(self, tiles: list[np.ndarray]) -> torch.Tensor:
        """uint8 BGR tiles -> normalised float tensor (N, 3, S, S).

        Handles: BGR->RGB, grayscale->3-channel, resize to ``input_size``,
        scale to [0, 1], ImageNet normalisation.

        Raises:
            ValueError: if ``tiles`` is empty or a tile is not 2-D/3-D uint8.
        """
        if not tiles:
            raise ValueError("no tiles to preprocess")

        prepared: list[np.ndarray] = []
        for tile in tiles:
            array = np.asarray(tile)
            if array.ndim == 2:
                array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
            elif array.ndim == 3 and array.shape[2] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            elif array.ndim == 3 and array.shape[2] == 1:
                array = cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"expected (H, W) or (H, W, 3) tile, got {array.shape}")

            if array.shape[:2] != (self.input_size, self.input_size):
                # INTER_AREA downsamples without aliasing; tiles are usually
                # larger than the backbone input, and aliasing here would look
                # exactly like fabric texture to the model.
                interp = (
                    cv2.INTER_AREA
                    if array.shape[0] > self.input_size
                    else cv2.INTER_LINEAR
                )
                array = cv2.resize(
                    array, (self.input_size, self.input_size), interpolation=interp
                )
            prepared.append(array)

        stacked = np.stack(prepared).astype(np.float32) / 255.0
        batch = torch.from_numpy(stacked).permute(0, 3, 1, 2).to(self.device)
        return (batch - self._mean) / self._std

    # -- forward ------------------------------------------------------------ #

    @torch.no_grad()
    def embed(self, tiles: list[np.ndarray]) -> torch.Tensor:
        """Tiles -> patch-embedding grids.

        Returns:
            float32 (N, G, G, D) where G is ``grid_size`` and D is ``embed_dim``.
            Every hooked layer is bilinearly resampled to the target grid before
            concatenation along channels.
        """
        batch = self.preprocess(tiles)
        self._forward(batch)

        resampled: list[torch.Tensor] = []
        for name in self.layers:
            feature = self._features[name]
            if feature.shape[-1] != self.grid_size:
                feature = F.interpolate(
                    feature,
                    size=(self.grid_size, self.grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
            resampled.append(feature)

        grid = torch.cat(resampled, dim=1)  # (N, D, G, G)
        return grid.permute(0, 2, 3, 1).contiguous().float()

    @torch.no_grad()
    def embed_flat(self, tiles: list[np.ndarray]) -> torch.Tensor:
        """Same as ``embed`` but flattened to (N*G*G, D) - memory-bank shape."""
        grid = self.embed(tiles)
        return grid.reshape(-1, grid.shape[-1])

    # -- PatchCore's local-context aggregation ------------------------------ #

    @staticmethod
    def neighbourhood_pool(grid: torch.Tensor, k: int = 3) -> torch.Tensor:
        """kxk average pool over the spatial grid, stride 1, padding k//2.

        This is PatchCore's local-context step: each embedding is smeared over
        its neighbours so a patch is described by its surroundings too, which is
        what makes the distance robust to exact patch alignment.

        Args:
            grid: (N, G, G, D).
        Returns:
            (N, G, G, D), same shape.
        """
        if k <= 1:
            return grid
        pooled = F.avg_pool2d(
            grid.permute(0, 3, 1, 2), kernel_size=k, stride=1, padding=k // 2
        )
        return pooled.permute(0, 2, 3, 1).contiguous()

    @property
    def n_params(self) -> int:
        """Parameter count - the model-size figure quoted in the results table."""
        return sum(p.numel() for p in self.model.parameters())
