# Contracts

Every inter-module interface, verbatim. These signatures are the contract.

This file plus the repo link is enough to build against any layer without
reading the layer on the other side of it.

---

## The seam — `AnomalyScorer`

```python
# src/linesight/detect/base.py
from typing import Protocol
import numpy as np

class AnomalyScorer(Protocol):
    """Anything that turns a fabric tile into a per-pixel anomaly score."""

    def fit(self, normal_tiles: list[np.ndarray]) -> None:
        """Build the SKU normality profile from defect-free tiles only."""

    def score(self, tile: np.ndarray) -> np.ndarray:
        """
        tile:    uint8, (H, W, 3) or (H, W), any size >= 224
        returns: float32, (H, W), unbounded, higher = more anomalous.
                 No thresholding, no normalisation. Calibration lives downstream.
        """

    def score_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "AnomalyScorer": ...
```

Three rules that are easy to break and expensive to discover:

1. **`score` returns unbounded float32.** Do not normalise to `[0, 1]`. Do not
   threshold. One threshold derived from a false-alarm budget must be comparable
   across scorers, and normalising per-tile destroys exactly that comparability.
2. **`fit` sees defect-free tiles only.** A scorer that needs labelled defects
   declares this by raising in `fit` and offering its own training entry point.
   `UNetScorer.fit` does precisely that, and that raise is an argument, not an
   omission.
3. **The pipeline always calls `score_batch`**, never `score` in a loop.
   Batching all tiles of a frame into one forward pass is the single biggest CPU
   win available.

Get a working heatmap with no bank, no torch, and no model:

```python
from linesight.detect.stub import StubScorer

scorer = StubScorer()
scorer.fit(normal_tiles)          # records statistics, builds nothing
heatmap = scorer.score(tile)      # float32 (H, W): noise + one bright blob
```

---

## Layer boundaries

### L1 → L2 — `Frame`

```python
@dataclass(slots=True)
class Frame:
    image: np.ndarray      # uint8, (H, W, 3), BGR
    index: int             # monotonic, 0-based, per run
    timestamp: float       # seconds
    source_id: str = ""    # filename / camera URL
```

Produced by anything satisfying `FrameSource`: `__iter__`, `__len__`, `open`,
`close`, `fps`. Three implementations, in increasing order of what can fail at
run time (ADR-006): `DirectorySource`, `VideoSource`, `MjpegSource`.

### L2 → L3 — `FrameGeometry`

```python
@dataclass(slots=True)
class FrameGeometry:
    along_mm: float                        # ROI top edge, down the roll
    mm_per_px: float                       # from ArUco corner spacing
    roi: tuple[int, int, int, int]         # (x, y, w, h) in the raw frame
    marker_ids: tuple[int, ...] = ()
    interpolated: bool = False             # position extrapolated, not measured
    gap_warning: bool = False              # continuity cannot be vouched for
```

`along_mm` is **absolute** (`marker_id * marker_pitch_mm`), never an integrated
speed. Absolute position cannot drift; an integrated one always does.

### L3 → L4 — `Tile`

```python
@dataclass(slots=True)
class Tile:
    image: np.ndarray      # uint8, (S, S, 3) or (S, S)
    x: int                 # top-left in FABRIC-ROI pixel coords
    y: int
    frame_index: int
    tile_index: int
```

`(x, y)` is relative to the fabric ROI, not the raw camera frame. Nothing
downstream of L2 ever sees the raw frame.

### L4 → L5 — `ScoreMap`

```python
@dataclass(slots=True)
class ScoreMap:
    scores: np.ndarray     # float32, (S, S), unbounded
    tile: Tile
```

### L5 — `Calibration`

```python
def threshold_from_budget(
    clean_scores: np.ndarray,
    budget_fa_per_100m: float,
    metres_per_tile: float,
) -> tuple[float, float]:
    """Returns (threshold, abstain_low)."""
```

`clean_scores` is the **max score per held-out clean tile**, and held-out means
disjoint from the fit set. Raises `ValueError` when the sample is too small to
resolve the requested budget.

### L5 → L6 — `Detection`

Per frame, per connected component, already converted to millimetres. Carries an
`Assertion` of `ASSERTED` or `UNCERTAIN`.

### L6 → L7 — `Event`

One physical defect, tracked across every frame it appears in. `length_mm` is
the machine-direction extent — the quantity ASTM scores. Only events that are
asserted and not operator-rejected contribute points.

### L7 → L8 — `RollReport`

```python
total_points, points_per_100yd2, verdict = score_roll(
    events, roll_length_m, width_m
)
```

---

## Coordinate conventions

| Space | Axes | Origin |
|---|---|---|
| Raw frame pixels | `(col, row)` | Top-left of the camera image. Used only inside L2. |
| Fabric-ROI pixels | `(x, y)` | Top-left of the cropped fabric. Everything L3–L6. |
| Machine millimetres | `(along_mm, across_mm)` | `along` = down the roll from its start; `across` = from the left selvedge. Everything L6–L8. |

`mm_per_px` is per-frame, not global: on a hand-pulled bench rig the camera
distance is not guaranteed constant, and pretending otherwise puts a systematic
error into every reported length.

---

## Artifact formats

### `banks/{sku}.npz`

| Key | Type | Notes |
|---|---|---|
| `bank` | float32 (K, 128) | The coreset. K ≈ 1,200. |
| `projection` | float32 (384, 128) | The Johnson–Lindenstrauss matrix. **Saved with the bank** — a bank projected by a different matrix is meaningless. |
| `backbone_name` | str | e.g. `resnet18` |
| `input_size` | int | e.g. 320 |
| `layers` | str list | e.g. `["layer2", "layer3"]` |
| `projection_dim` | int | 128 |
| `coreset_frac` | float | 0.10 |
| `n_normal_tiles` | int | 30 |
| `fit_timestamp` | str | ISO 8601 |
| `linesight_version` | str | |

`PatchCore.load` compares backbone, input size, layers, and projection dim
against the running config and **refuses** on a mismatch. A mismatched bank does
not crash — it produces plausible-looking nonsense, which is worse.

### `linesight.db`

`rolls`, `events`, `decisions`. `decisions` is append-only: a QA disposition that
changed and left no trace is worse than no disposition at all.

---

## Errors, and what they mean

| Raised by | Condition | Why it is not a warning |
|---|---|---|
| `threshold_from_budget` | Too few clean tiles for the budget | Returning a number built on one order statistic would be the single dishonest line in the system. |
| `PatchCore.load` | Bank metadata mismatch | Silent wrong answers beat loud failures only for people who are not accountable for the result. |
| `DirectorySource` | Unreadable frame file | A skipped frame is a stretch of roll nobody inspected. |
| `points_per_100yd2` | Zero inspected area | Otherwise it reports a spectacular defect rate for a roll nobody measured. |
| `Pipeline.iter_frames` | No calibration set | Running without a threshold means inventing one. |
| `UNetScorer.fit` | Called at all | Supervised models cannot be fitted from defect-free tiles. That is the cold-start problem, as executable code. |
