"""PatchCore - the shipped model. Hand-written, no Anomalib (ADR-001).

Three steps, no gradients:

  1. **Fit.**   Every defect-free tile -> frozen backbone -> layer2+layer3 grid
                -> 3x3 neighbourhood pool -> random-project 384->128 -> pool all
                embeddings into one bank -> greedy k-center coreset to 10%.
  2. **Score.** New tile -> same grid -> each location's L2 distance to its
                nearest neighbour in the bank.
  3. **Upsample.** Bilinear to tile size, Gaussian smooth (sigma ~4 px).

This is backprop-free, not data-free, and the distinction matters when reading
the code: three things are fitted from data - the frozen ImageNet ResNet
(fitted long ago, by someone else), the per-SKU memory bank built in ``fit``,
and the threshold derived downstream in ``calibrate`` from held-out clean
fabric. What is absent is a training loop, not the data.

The free functions at the top (coreset selection, random projection) are kept
out of the class so each can be tested and profiled on its own; ``PatchCore``
itself is mostly the wiring between them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from ..config import DetectConfig
from .backbone import FeatureExtractor, resolve_device
from .base import ScorerMeta

__all__ = ["PatchCore", "gaussian_random_projection", "greedy_coreset", "random_coreset"]


# --------------------------------------------------------------------------- #
# Free functions - independently testable, independently probe-able
# --------------------------------------------------------------------------- #


def greedy_coreset(x: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """k-center greedy: iteratively pick the point farthest from all picks so far.

    Returns indices. O(N*k), done in one pass by maintaining a running
    min-distance vector.

    12,000 -> 1,200 in ~2 s on GPU, ~20 s on CPU. Cost grows quadratically with
    the fit set, so ``random_coreset`` is the documented fallback for large fit
    sets (ADR-010); the accuracy it costs is measured, not assumed - see the
    third configuration in ``probes/p11_mvtec_reproduction.py``.

    Args:
        x: (N, D) float32 embeddings.
        k: number of points to keep.
        seed: seeds the single random choice of the first centre.
    Returns:
        int64 tensor of ``k`` indices into ``x``.

    Raises:
        ValueError: if ``k`` is not in ``1..len(x)``.
    """
    n = len(x)
    if k < 1 or k > n:
        raise ValueError(f"coreset size {k} out of range for {n} embeddings")

    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = [torch.randint(n, (1,), generator=g, device=x.device).item()]
    min_d = torch.cdist(x, x[idx[-1] : idx[-1] + 1]).squeeze(1)
    for _ in range(k - 1):
        idx.append(int(min_d.argmax()))
        d = torch.cdist(x, x[idx[-1] : idx[-1] + 1]).squeeze(1)
        min_d = torch.minimum(min_d, d)
    return torch.tensor(idx, dtype=torch.int64, device=x.device)


def random_coreset(x: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Uniform random subsample. The fallback of ADR-010, benchmarked not assumed."""
    n = len(x)
    if k < 1 or k > n:
        raise ValueError(f"coreset size {k} out of range for {n} embeddings")
    g = torch.Generator(device=x.device).manual_seed(seed)
    return torch.randperm(n, generator=g, device=x.device)[:k].to(torch.int64)


def gaussian_random_projection(
    dim_in: int, dim_out: int, seed: int = 0, device: torch.device | None = None
) -> torch.Tensor:
    """Johnson-Lindenstrauss projection matrix, (dim_in, dim_out).

    Entries ~ N(0, 1/dim_out) so that expected squared distance is preserved.
    Distances survive within a few percent and NN search gets ~3x faster.
    The matrix is saved with the bank - a bank projected by a different matrix
    is meaningless, so it is part of the artifact, not a re-derivable constant.

    Raises:
        ValueError: if the projection would not reduce dimension.
    """
    if dim_out > dim_in:
        raise ValueError(f"projection {dim_in} -> {dim_out} would not reduce dimension")
    g = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dim_in, dim_out, generator=g) / (dim_out**0.5)
    return matrix.to(device) if device is not None else matrix


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #


class PatchCore:
    """Memory-bank anomaly scorer implementing ``AnomalyScorer``.

    Attributes:
        bank: (K, projection_dim) float32 - the coreset. ~1,200 x 128.
        projection: (embed_dim, projection_dim) float32.
        meta: ``ScorerMeta`` provenance, written into the .npz.
    """

    def __init__(self, config: DetectConfig | None = None) -> None:
        """Construct an unfitted scorer. The backbone loads lazily on first use."""
        self.config = config or DetectConfig()
        self.device = resolve_device(self.config.device)
        self.bank: torch.Tensor | None = None
        self.projection: torch.Tensor | None = None
        self._backbone: FeatureExtractor | None = None
        self._meta: ScorerMeta | None = None

    # -- L4 protocol -------------------------------------------------------- #

    def fit(self, normal_tiles: list[np.ndarray]) -> None:
        """Build the SKU normality profile from defect-free tiles only.

        Steps: embed -> neighbourhood pool -> flatten -> project -> coreset.
        Target wall-clock: ~90 s for 30 tiles on CPU.

        Raises:
            ValueError: if fewer than 2 tiles are given.
        """
        if len(normal_tiles) < 2:
            raise ValueError(
                f"need at least 2 defect-free tiles to fit a bank, got {len(normal_tiles)}"
            )

        backbone = self._ensure_backbone()
        self.projection = gaussian_random_projection(
            backbone.embed_dim,
            self.config.projection_dim,
            seed=self.config.coreset_seed,
            device=self.device,
        )

        chunks: list[torch.Tensor] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(normal_tiles), batch_size):
            grids = self._embed_grids(normal_tiles[start : start + batch_size])
            chunks.append(grids.reshape(-1, grids.shape[-1]))
        embeddings = torch.cat(chunks, dim=0)

        k = max(1, round(self.config.coreset_frac * len(embeddings)))
        select = greedy_coreset if self.config.coreset_method == "greedy" else random_coreset
        indices = select(embeddings, k, seed=self.config.coreset_seed)
        self.bank = embeddings[indices].contiguous()

        self._meta = ScorerMeta(
            backbone_name=self.config.backbone,
            input_size=self.config.input_size,
            layers=tuple(self.config.layers),
            projection_dim=self.config.projection_dim,
            coreset_frac=self.config.coreset_frac,
            n_normal_tiles=len(normal_tiles),
            fit_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            linesight_version=_version(),
        )
        self._meta.validate()

    def score(self, tile: np.ndarray) -> np.ndarray:
        """One tile -> float32 (H, W) unbounded anomaly map. See the protocol."""
        return self.score_batch([tile])[0]

    def score_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """All tiles of a frame in one forward pass. The path the pipeline uses.

        Raises:
            RuntimeError: if the scorer has not been fitted or loaded.
        """
        if not self.is_fitted:
            raise RuntimeError("PatchCore is not fitted - call fit() or load() first")
        if not tiles:
            return []

        # Chunked by config.batch_size. A frame is only ~12 tiles so this never
        # mattered on the live path, but bulk scoring - calibration over a few
        # hundred clean tiles - asked for one forward pass over all of them and
        # tried to allocate 3.3 GB. fit() already chunked; this did not.
        step = max(1, self.config.batch_size)
        maps: list[np.ndarray] = []
        for start in range(0, len(tiles), step):
            batch = tiles[start : start + step]
            grids = self._embed_grids(batch)  # (N, G, G, P)
            n, grid, _, dim = grids.shape
            distances = self._nn_distance(grids.reshape(-1, dim)).reshape(n, grid, grid)
            maps.extend(
                self._upsample_and_smooth(
                    distances[i],
                    np.asarray(batch[i]).shape[:2],
                    self.config.smooth_sigma_px,
                )
                for i in range(n)
            )
        return maps

    def save(self, path: str | Path) -> None:
        """Write ``banks/{sku}.npz``: bank, projection, and every meta key.

        ~600 KB - small enough to commit beside the SKU's config, which is the
        point: the deployable artifact of this system is a memory bank and a
        threshold, not a checkpoint.
        """
        if not self.is_fitted:
            raise RuntimeError("nothing to save - PatchCore is not fitted")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            bank=self.bank.cpu().numpy().astype(np.float32),
            projection=self.projection.cpu().numpy().astype(np.float32),
            **{key: np.asarray(value) for key, value in self.meta.items()},
        )

    @classmethod
    def load(cls, path: str | Path, config: DetectConfig | None = None) -> PatchCore:
        """Restore a fitted scorer.

        Raises:
            FileNotFoundError: if the bank does not exist - with the exact
                ``linesight fit`` command that would create it.
            ValueError: if the bank's metadata is incompatible with ``config``
                (different backbone, input size, layers, or projection dim).
                Loading a mismatched bank would produce plausible-looking
                nonsense, so it fails loudly instead.
        """
        bank_path = Path(path)
        if not bank_path.exists():
            raise FileNotFoundError(
                f"no memory bank at {bank_path}. Fit one first:\n"
                f"  python -m linesight fit --sku {bank_path.stem} --normal <folder>"
            )

        scorer = cls(config)
        with np.load(bank_path, allow_pickle=False) as data:
            stored = ScorerMeta(
                backbone_name=str(data["backbone_name"]),
                input_size=int(data["input_size"]),
                layers=tuple(str(v) for v in np.atleast_1d(data["layers"])),
                projection_dim=int(data["projection_dim"]),
                coreset_frac=float(data["coreset_frac"]),
                n_normal_tiles=int(data["n_normal_tiles"]),
                fit_timestamp=str(data["fit_timestamp"]),
                linesight_version=str(data["linesight_version"]),
            )
            stored.validate()
            stored.assert_compatible(
                ScorerMeta(
                    backbone_name=scorer.config.backbone,
                    input_size=scorer.config.input_size,
                    layers=tuple(scorer.config.layers),
                    projection_dim=scorer.config.projection_dim,
                )
            )
            scorer.bank = torch.from_numpy(data["bank"]).to(scorer.device)
            scorer.projection = torch.from_numpy(data["projection"]).to(scorer.device)
        scorer._meta = stored
        return scorer

    # -- internals ---------------------------------------------------------- #

    def _ensure_backbone(self) -> FeatureExtractor:
        """Lazily construct the feature extractor. Keeps import cheap for the CLI."""
        if self._backbone is None:
            self._backbone = FeatureExtractor(
                backbone=self.config.backbone,
                layers=tuple(self.config.layers),
                input_size=self.config.input_size,
                device=self.device,
                grid_align=self.config.grid_align,
            )
        return self._backbone

    @torch.no_grad()
    def _embed_grids(self, tiles: list[np.ndarray]) -> torch.Tensor:
        """Tiles -> pooled, projected grids (N, G, G, projection_dim)."""
        backbone = self._ensure_backbone()
        grids = backbone.embed(tiles)
        grids = backbone.neighbourhood_pool(grids, self.config.neighbourhood_pool)
        return grids @ self.projection

    @torch.no_grad()
    def _nn_distance(self, queries: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbour L2 distance from each query to the bank.

        Brute force ``torch.cdist``: 400 x 1,200 x 128 = 61 MFLOP, sub-millisecond.
        FAISS is unnecessary at this scale - ADR-011 says so explicitly rather
        than leaving its absence to look like an oversight.

        Args:
            queries: (M, projection_dim).
        Returns:
            (M,) float32 distances.
        """
        return torch.cdist(queries, self.bank).min(dim=1).values

    @staticmethod
    def _upsample_and_smooth(
        grid_scores: torch.Tensor, out_hw: tuple[int, int], sigma: float
    ) -> np.ndarray:
        """(G, G) patch scores -> (H, W) pixel map: bilinear resize + Gaussian blur.

        The blur is not cosmetic: patch scores are piecewise-constant over a
        stride-16 grid, and thresholding that staircase directly would produce
        blocky components whose measured lengths snap to the grid.
        """
        resized = F.interpolate(
            grid_scores[None, None],
            size=(int(out_hw[0]), int(out_hw[1])),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        scores = resized.cpu().numpy().astype(np.float32)
        if sigma > 0:
            scores = gaussian_filter(scores, sigma=sigma)
        return scores.astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self.bank is not None and self.projection is not None

    @property
    def meta(self) -> ScorerMeta:
        """Provenance of the current bank. Raises if unfitted."""
        if self._meta is None:
            raise RuntimeError("PatchCore is not fitted - no metadata to report")
        return self._meta

    @property
    def bank_size(self) -> int:
        """Points in the coreset - the bank size the CLI reports after a fit."""
        return 0 if self.bank is None else int(self.bank.shape[0])


def _version() -> str:
    """Imported lazily to avoid a circular import at module load."""
    from .. import __version__

    return __version__
