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
and the training protocol; the bodies raise ``NotImplementedError`` because the
training needs a GPU and AITEX's 105 pixel-annotated defective images, and has
not been run. Its output would be one row of the results table rather than an
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
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UNet(nn.Module):
    """Small U-Net: 4 down, 4 up, skip connections, 1 logit channel out.

    Deliberately small (~2 M params). A bigger one would overfit 105 images
    harder, flattering the in-distribution result of the very baseline this
    exists to measure honestly.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32, depth: int = 4) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) -> (N, 1, H, W) logits. No sigmoid - the loss applies it."""
        raise NotImplementedError


def dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    """BCE-with-logits plus soft Dice.

    Defect pixels are perhaps 1% of an AITEX crop; plain BCE would happily
    predict all-background and call it 99% accurate. Dice is what forces it to
    actually segment.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError

    def fit(self, normal_tiles: list[np.ndarray]) -> None:
        """Raises ``NotImplementedError`` - and the raise is the point.

        A supervised model cannot be fitted from defect-free tiles alone: it
        needs ``train_unet`` and pixel-annotated defects for this construction,
        a corpus that does not exist in a mill. The cold-start constraint the
        whole project is built around is therefore recorded as executable
        behaviour rather than prose - anything that tries to cold-start this
        scorer fails immediately, and says why.
        """
        raise NotImplementedError

    def score(self, tile: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> UNetScorer:
        raise NotImplementedError
