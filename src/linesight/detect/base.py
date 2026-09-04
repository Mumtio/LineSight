"""THE SEAM - the only interface L4 exposes to the rest of the system.

Everything upstream produces tiles; everything downstream consumes score maps.
Nothing else crosses the boundary. The protocol is reproduced verbatim in
``docs/contracts.md``, and ``tests/test_skeleton.py`` locks its method list so
the contract cannot drift silently.

Three implementations exist deliberately, and all three are selected by the
same config key, ``detect.scorer``:

  * ``stub.StubScorer``       - noise plus a bright blob, numpy only. Lets the
                                other seven layers run, and be tested in CI,
                                with no torch, no memory bank and no dataset.
  * ``patchcore.PatchCore``   - the shipped model.
  * ``baselines.unet.UNetScorer`` - the supervised baseline behind the same
                                protocol, so comparing the two is a config
                                change rather than a second pipeline.

Contract notes that are easy to get wrong:

  * ``score`` returns **unbounded** float32. Do not normalise to [0, 1], do not
    threshold. Calibration is a separate layer for a reason: one threshold
    derived from a false-alarm budget must be comparable across scorers.
  * ``fit`` sees **defect-free tiles only**. A scorer that needs labelled
    defects (the U-Net) declares that by raising in ``fit`` and offering its own
    training entry point instead - which is itself the finding the supervised
    baseline exists to record (ADR-004).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["AnomalyScorer", "ScorerMeta"]


@runtime_checkable
class AnomalyScorer(Protocol):
    """Anything that turns a fabric tile into a per-pixel anomaly score."""

    def fit(self, normal_tiles: list[np.ndarray]) -> None:
        """Build the SKU normality profile from defect-free tiles only.

        Args:
            normal_tiles: uint8 arrays, (H, W, 3) or (H, W). Need not be the
                same size as each other; the scorer resizes internally.
        """
        ...

    def score(self, tile: np.ndarray) -> np.ndarray:
        """Score one tile.

        Args:
            tile: uint8, (H, W, 3) or (H, W), any size >= 224.

        Returns:
            float32, (H, W), unbounded, higher = more anomalous.
            No thresholding, no normalisation. Calibration lives downstream.
        """
        ...

    def score_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """Score many tiles in one forward pass.

        The pipeline always calls this, never ``score`` in a loop - batching
        all tiles of a frame into one forward pass is the single biggest CPU
        win available. A scorer with no batching of its own may implement it as
        a list comprehension over ``score``.
        """
        ...

    def save(self, path: str) -> None:
        """Persist the fitted profile. Must round-trip through ``load``."""
        ...

    @classmethod
    def load(cls, path: str) -> AnomalyScorer:
        """Restore a scorer saved by ``save``."""
        ...


class ScorerMeta(dict):
    """Provenance stored alongside every fitted artifact.

    Keys (all present, no exceptions): ``backbone_name``, ``input_size``,
    ``layers``, ``projection_dim``, ``coreset_frac``, ``n_normal_tiles``,
    ``fit_timestamp``, ``linesight_version``.

    A bank whose metadata does not match the running config is a silent
    correctness bug - ``PatchCore.load`` refuses it rather than guessing.
    """

    REQUIRED: tuple[str, ...] = (
        "backbone_name",
        "input_size",
        "layers",
        "projection_dim",
        "coreset_frac",
        "n_normal_tiles",
        "fit_timestamp",
        "linesight_version",
    )

    #: Keys that must agree between a saved bank and the running config. A
    #: difference in any of these makes the bank's distances meaningless.
    COMPARED: tuple[str, ...] = (
        "backbone_name",
        "input_size",
        "layers",
        "projection_dim",
    )

    def validate(self) -> None:
        """Raise ``ValueError`` if any required key is missing."""
        missing = [key for key in self.REQUIRED if key not in self]
        if missing:
            raise ValueError(f"artifact metadata is missing {missing}")

    def assert_compatible(self, other: ScorerMeta) -> None:
        """Raise if a loaded bank was fitted under an incompatible config.

        Compared: backbone, input size, layers, projection dim. Not compared:
        timestamps, tile counts - those are informational.

        Loading a mismatched bank does not crash; it produces plausible-looking
        nonsense, which is worse. So this fails loudly instead.
        """
        differences = [
            f"{key}: bank has {self.get(key)!r}, config wants {other.get(key)!r}"
            for key in self.COMPARED
            if _normalise(self.get(key)) != _normalise(other.get(key))
        ]
        if differences:
            joined = "\n  ".join(differences)
            raise ValueError(
                f"this memory bank was fitted under a different configuration:\n  {joined}\n"
                "Refit the SKU, or run with the configuration it was fitted under."
            )


def _normalise(value: object) -> object:
    """Compare sequences by content, so a tuple and a list of layers agree.

    A bank round-trips through .npz, which turns tuples into arrays; without
    this, every reloaded bank would look incompatible with itself.
    """
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return value
