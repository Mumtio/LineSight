"""Shared fixtures.

Two rules the suite lives by:

  * **No test requires a dataset.** Anything touching ``data/`` is marked
    ``@pytest.mark.data`` and skips when the folder is absent, so ``make test``
    is green on a fresh clone before ``fetch_data.sh`` has ever run.
  * **No test requires the model.** The pipeline is exercised against
    ``StubScorer``, which is the whole point of the seam.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Skips the whole test if the datasets have not been fetched."""
    if not DATA_ROOT.exists():
        pytest.skip("data/ not populated - run scripts/fetch_data.sh")
    return DATA_ROOT


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded. Every stochastic test must be reproducible or it is not a test."""
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_fabric(rng: np.random.Generator) -> np.ndarray:
    """A 512x512 texture that looks enough like plain-weave fabric.

    A regular warp/weft grid plus noise. Not a substitute for real fabric, but
    enough to exercise tiling, flat-field, and the frame loop deterministically.
    """
    base = np.full((512, 512), 128, dtype=np.int16)
    base[::4, :] += 18
    base[:, ::4] += 18
    base += rng.normal(0, 6, size=base.shape).astype(np.int16)
    grey = np.clip(base, 0, 255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


@pytest.fixture
def synthetic_defect(synthetic_fabric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same fabric with a bright vertical slub, plus its ground-truth mask.

    Returns ``(image, mask)``. A defect with a known mask is what lets a
    detector test assert overlap rather than merely "something lit up".
    """
    image = synthetic_fabric.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    image[180:300, 250:258] = 235
    mask[180:300, 250:258] = 1
    return image, mask
