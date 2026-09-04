"""The supervised baseline - measured against the shipped detector (ADR-004).

Supervised segmentation is the obvious approach to defect detection, so it is
measured here rather than dismissed: a small U-Net for binary segmentation on
AITEX's 105 pixel-annotated defective images, 256x256 crops, Dice + BCE, Adam,
~30 epochs on a GPU.

It implements ``AnomalyScorer``, so it drops into the same evaluation harness
as PatchCore and the comparison is one config key rather than a second
pipeline. That is also the strongest evidence that the seam in
``detect/base.py`` is real rather than decorative.

The comparison runs in two conditions, and the second is the informative one:
in-distribution (the same fabric constructions in train and test), and
leave-one-fabric-out (a construction the model has never seen). A mill onboards
new constructions continuously, so the held-out case is the one that decides
whether an approach is deployable there at all.

**Not shipped.** A supervised model needs pixel-annotated defects for every
construction, and that corpus does not exist in a mill - which is exactly what
``UNetScorer.fit`` says by raising. This module fixes the architecture, the loss
and the training protocol; ``probes/p18_unet_baseline.py`` trains it and writes
one row of the results table. The output is a comparison number, not an
artifact in ``banks/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = ["DoubleConv", "UNet", "UNetScorer", "dice_bce_loss", "train_unet"]


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2. The U-Net block, nothing exotic."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small U-Net: 4 down, 4 up, skip connections, 1 logit channel out.

    Deliberately small (~2 M params). A bigger one would overfit 105 images
    harder, flattering the in-distribution result of the very baseline this
    exists to measure honestly.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        channels = in_ch
        widths = [base_ch * 2**i for i in range(depth)]
        for width in widths:
            self.downs.append(DoubleConv(channels, width))
            channels = width

        self.bottleneck = DoubleConv(channels, channels * 2)
        channels *= 2

        for width in reversed(widths):
            self.upconvs.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
            self.ups.append(DoubleConv(width * 2, width))
            channels = width

        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, 1, H, W) logits. No sigmoid - the loss applies it."""
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, up, skip in zip(self.upconvs, self.ups, reversed(skips), strict=True):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = up(torch.cat([skip, x], dim=1))

        return self.head(x)


def dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    """BCE-with-logits plus soft Dice.

    Defect pixels are perhaps 1% of an AITEX crop; plain BCE would happily
    predict all-background and call it 99% accurate. Dice is what forces it to
    actually segment.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice_weight) * bce + dice_weight * dice.mean()


def _sample_crops(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    crop: int,
    count: int,
    rng: np.random.Generator,
    defect_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``count`` crops, half of them centred on an annotated defect pixel.

    A uniformly random crop of a 4096x256 strip almost never contains the
    defect, so uniform sampling would train the model on background and produce
    a baseline too weak to be worth comparing against. Balancing the draw is
    what makes this a fair opponent rather than a strawman - and it is a choice
    that flatters the baseline, which is the right direction for a comparison
    the shipped method is meant to survive.
    """
    xs, ys = [], []
    for _ in range(count):
        idx = int(rng.integers(len(images)))
        image, mask = images[idx], masks[idx]
        h, w = mask.shape[:2]
        ch, cw = min(crop, h), min(crop, w)

        defect_pixels = np.argwhere(mask > 0)
        if len(defect_pixels) and rng.random() < defect_fraction:
            cy, cx = defect_pixels[int(rng.integers(len(defect_pixels)))]
            top = int(np.clip(cy - ch // 2, 0, max(h - ch, 0)))
            left = int(np.clip(cx - cw // 2, 0, max(w - cw, 0)))
        else:
            top = int(rng.integers(0, max(h - ch, 0) + 1))
            left = int(rng.integers(0, max(w - cw, 0) + 1))

        tile = image[top : top + ch, left : left + cw]
        tile_mask = mask[top : top + ch, left : left + cw]
        if tile.ndim == 2:
            tile = np.repeat(tile[:, :, None], 3, axis=2)
        xs.append(tile.astype(np.float32) / 255.0)
        ys.append((tile_mask > 0).astype(np.float32))

    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(np.stack(ys)).unsqueeze(1).contiguous()
    return x, y


def train_unet(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    crop: int = 256,
    device: str = "cuda",
    seed: int = 0,
    log_every: int = 1,
) -> tuple[UNet, list[dict]]:
    """Train the baseline. Returns the model and a per-epoch history.

    History is returned rather than printed so the caller can write it to
    ``results/unet_history.csv`` as soon as training ends: a hosted session can
    be reclaimed at any moment, and a metric that only reached stdout is lost
    with it.
    """
    if not images:
        raise ValueError("no training images - a supervised baseline needs labels")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = UNet().to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    steps = max(1, len(images) // batch_size)

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps):
            x, y = _sample_crops(images, masks, crop, batch_size, rng)
            x, y = x.to(dev), y.to(dev)
            optimiser.zero_grad(set_to_none=True)
            loss = dice_bce_loss(model(x), y)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        record = {"epoch": epoch, "loss": round(total / steps, 5), "steps": steps}
        history.append(record)
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:>3}/{epochs}  loss {record['loss']:.5f}", flush=True)

    return model, history


class UNetScorer:
    """``AnomalyScorer`` adapter around a trained U-Net.

    ``score`` returns the pre-sigmoid logit map - unbounded and uncalibrated,
    exactly like PatchCore's distance map, so the same
    ``threshold_from_budget`` calibration applies to both and the comparison is
    like for like.
    """

    def __init__(
        self, model: UNet | None = None, device: str = "cuda", input_size: int = 256
    ) -> None:
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.input_size = input_size
        self.model = model.to(self.device).eval() if model is not None else None

    def fit(self, normal_tiles: list[np.ndarray]) -> None:
        """Raises ``NotImplementedError`` - and the raise is the point.

        A supervised model cannot be fitted from defect-free tiles alone: it
        needs ``train_unet`` and pixel-annotated defects for this construction,
        a corpus that does not exist in a mill. The cold-start constraint the
        whole project is built around is therefore recorded as executable
        behaviour rather than prose - anything that tries to cold-start this
        scorer fails immediately, and says why.
        """
        raise NotImplementedError(
            "UNetScorer cannot be fitted from defect-free tiles. A supervised "
            "model needs pixel-annotated defects for this construction, which is "
            "the corpus a mill does not have. Use train_unet() with labelled "
            "data, or use PatchCore, which does cold-start. See ADR-004."
        )

    def score(self, tile: np.ndarray) -> np.ndarray:
        return self.score_batch([tile])[0]

    def score_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        if self.model is None:
            raise RuntimeError("UNetScorer has no model - train_unet() or load() first")

        batch = []
        for tile in tiles:
            array = tile
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            batch.append(array.astype(np.float32) / 255.0)

        shapes = [b.shape[:2] for b in batch]
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            logits = self.model(x)

        maps = []
        for i, (h, w) in enumerate(shapes):
            single = logits[i : i + 1]
            if single.shape[-2:] != (h, w):
                single = nn.functional.interpolate(
                    single, size=(h, w), mode="bilinear", align_corners=False
                )
            maps.append(single[0, 0].cpu().numpy().astype(np.float32))
        return maps

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save - UNetScorer has no model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "input_size": self.input_size},
            destination,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> UNetScorer:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = UNet()
        model.load_state_dict(payload["state_dict"])
        return cls(model=model, device=device, input_size=int(payload["input_size"]))
