"""A synthetic scorer, so the other seven layers can run without a model.

``StubScorer`` implements ``AnomalyScorer`` exactly and returns a plausible
heatmap - background noise with one bright Gaussian blob - with no backbone, no
memory bank and no fitted parameters. That makes it the scorer the structural
tests run against: acquisition through to the PDF can be exercised end to end
on a machine that has neither torch nor a dataset, and any failure is then a
failure of the layer under test rather than of the model.

Deliberately dependency-free: numpy only, no torch, no timm, no cv2.
``tests/test_skeleton.py`` asserts exactly that, because the moment it stops
being true, so does the guarantee.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

__all__ = ["StubScorer"]


class StubScorer:
    """Background noise plus a bright Gaussian blob at a deterministic spot.

    Implements ``AnomalyScorer`` exactly. ``fit`` records the tile statistics it
    was shown so that round-tripping through ``save``/``load`` is a real test of
    the artifact contract rather than a no-op.
    """

    def __init__(self, seed: int = 0, blob_sigma_px: float = 12.0, blob_gain: float = 4.0) -> None:
        """Args:
        seed: makes the blob position and noise reproducible per tile.
        blob_sigma_px: radius of the fake defect.
        blob_gain: peak height above the noise floor.
        """
        self.seed = int(seed)
        self.blob_sigma_px = float(blob_sigma_px)
        self.blob_gain = float(blob_gain)
        self.tile_mean: float = 0.0
        self.tile_std: float = 0.0
        self.n_normal_tiles: int = 0

    # -- L4 protocol -------------------------------------------------------- #

    def fit(self, normal_tiles: list[np.ndarray]) -> None:
        """Record mean/std of the given tiles. No model is built.

        Raises:
            ValueError: if no tiles are supplied, matching the real scorer's
                behaviour so that a caller's error handling is exercised too.
        """
        if not normal_tiles:
            raise ValueError("no normal tiles supplied")
        values = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in normal_tiles])
        self.tile_mean = float(values.mean())
        self.tile_std = float(values.std())
        self.n_normal_tiles = len(normal_tiles)

    def score(self, tile: np.ndarray) -> np.ndarray:
        """Return float32 (H, W): N(0, 1) noise plus one Gaussian bump.

        The bump's centre is a hash of the tile's contents, so the same tile
        always scores the same way and downstream tests are deterministic.
        """
        array = np.asarray(tile)
        height, width = array.shape[:2]
        rng = np.random.default_rng(self._tile_seed(array))

        scores = rng.normal(0.0, 1.0, size=(height, width))

        cy = rng.integers(height // 4, max(height // 4 + 1, 3 * height // 4))
        cx = rng.integers(width // 4, max(width // 4 + 1, 3 * width // 4))
        ys = np.arange(height, dtype=np.float64)[:, None]
        xs = np.arange(width, dtype=np.float64)[None, :]
        sq_dist = (ys - float(cy)) ** 2 + (xs - float(cx)) ** 2
        scores += self.blob_gain * np.exp(-sq_dist / (2.0 * self.blob_sigma_px**2))

        return scores.astype(np.float32)

    def score_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """No real batching to do here - the interface is what matters."""
        return [self.score(tile) for tile in tiles]

    def save(self, path: str | Path) -> None:
        """Persist the recorded statistics. Same .npz shape as a real bank."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            kind="stub",
            seed=self.seed,
            blob_sigma_px=self.blob_sigma_px,
            blob_gain=self.blob_gain,
            tile_mean=self.tile_mean,
            tile_std=self.tile_std,
            n_normal_tiles=self.n_normal_tiles,
        )

    @classmethod
    def load(cls, path: str | Path) -> StubScorer:
        """Restore a scorer saved by ``save``.

        Raises:
            ValueError: if the file is not a stub artifact - loading a real
                memory bank into the stub would produce confident nonsense.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["kind"]) != "stub":
                raise ValueError(f"{path} is not a StubScorer artifact")
            scorer = cls(
                seed=int(data["seed"]),
                blob_sigma_px=float(data["blob_sigma_px"]),
                blob_gain=float(data["blob_gain"]),
            )
            scorer.tile_mean = float(data["tile_mean"])
            scorer.tile_std = float(data["tile_std"])
            scorer.n_normal_tiles = int(data["n_normal_tiles"])
        return scorer

    # -- internals ---------------------------------------------------------- #

    def _tile_seed(self, tile: np.ndarray) -> int:
        """A stable seed derived from the tile's bytes and this scorer's seed.

        Content-derived rather than call-counted, so scoring the same tile twice
        - or scoring tiles in a different order - gives identical output.
        """
        digest = hashlib.blake2b(
            np.ascontiguousarray(tile).tobytes(), digest_size=8, key=str(self.seed).encode()
        )
        return int.from_bytes(digest.digest(), "little")

    @property
    def is_fitted(self) -> bool:
        return self.n_normal_tiles > 0
