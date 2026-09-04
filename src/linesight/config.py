"""Configuration: one YAML file per SKU, validated on load.

Nothing in the pipeline reads a magic number from its own module. Every tunable
lives here, so two runs are compared by diffing two files, and a fitted bank can
record the exact config it was built under (``dump_config`` writes it beside the
bank). The section classes below follow the layer order of the pipeline.

Layering: ``configs/default.yaml`` is the base; a SKU file overrides only the
keys it cares about. ``load_config`` merges them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AcquisitionConfig",
    "ApiConfig",
    "CalibrationConfig",
    "Config",
    "DetectConfig",
    "EventConfig",
    "GeometryConfig",
    "PreprocessConfig",
    "ScoringConfig",
    "dump_config",
    "load_config",
    "merge_dicts",
]


class _Section(BaseModel):
    """Base for every config section.

    ``extra="forbid"`` belongs on the SECTIONS, not just on the root. It was on
    the root alone, which meant a key misfiled into the wrong section - the easy
    mistake, since the sections all look alike - was silently ignored. The run
    then used the default and reported it as though the override had applied.
    A typo that changes behaviour and says nothing is the worst kind.
    """

    model_config = ConfigDict(extra="forbid")


class AcquisitionConfig(_Section):
    """L1. Which frame source, and how fast to consume it."""

    source_type: str = "directory"        # directory | video | mjpeg
    uri: str = ""                         # folder path, video path, or stream URL
    frame_stride: int = 1                 # process every Nth frame; state it, do not hide it
    glob: str = "*.png"                   # directory source only
    loop: bool = False
    synthetic_fps: float = 30.0           # directory source: timestamps = index / fps
    drop_stale: bool = True               # mjpeg: keep only the newest frame


class GeometryConfig(_Section):
    """L2. ArUco decoding and the fallback when markers go missing."""

    aruco_dict: str = "DICT_4X4_50"
    marker_length_mm: float = 40.0        # physical edge length of a printed marker
    marker_pitch_mm: float = 100.0        # spacing between marker centres on the tape
    roi: tuple[int, int, int, int] | None = None   # fixed ROI; None = full frame
    fallback_speed_mm_per_s: float | None = None   # bench simplification, must be declared
    max_gap_mm: float = 250.0             # beyond this without a marker -> gap_warning
    max_gap_s: float = 5.0                # ... and this, when no speed is known at all


class PreprocessConfig(_Section):
    """L3. Flat-field correction and tiling."""

    flatfield: bool = True
    flatfield_sigma_px: float = 101.0     # Gaussian blur radius estimating illumination
    tile_size: int = 512
    tile_overlap: int = 64
    drop_partial_edge_tiles: bool = False  # False = pad; True = discard remainder


class DetectConfig(_Section):
    """L4. The detector's specification; the algorithm is in ``detect/patchcore.py``."""

    scorer: str = "patchcore"             # patchcore | stub | unet
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    input_size: int = 320                 # backbone input; tiles are resized to this
    grid_align: str = "coarsest"          # coarsest | finest -- see backbone.py
    neighbourhood_pool: int = 3           # 3x3 avg, stride 1
    projection_dim: int = 128             # Gaussian random projection 384 -> 128
    coreset_frac: float = 0.10            # keep 10%; ADR-010 records the deviation from 1%
    coreset_method: str = "greedy"        # greedy | random (documented fallback)
    coreset_seed: int = 0
    n_normal_tiles: int = 30              # cold-start target: one SKU from ~30 tiles
    smooth_sigma_px: float = 4.0          # Gaussian smoothing of the upsampled map
    batch_size: int = 12                  # all tiles of one frame in one forward pass
    device: str = "auto"                  # auto | cpu | cuda
    bank_dir: Path = Path("banks")


class CalibrationConfig(_Section):
    """L5. The false-alarm budget - the number the operator actually sets."""

    budget_fa_per_100m: float = 1.0
    abstain_multiplier: float = 5.0       # abstain band spans 5x the allowed tail
    stability_margin: int = 10            # order statistics required above the threshold
    clean_scores_path: Path | None = None


class EventConfig(_Section):
    """L6. Component filtering and cross-frame tracking."""

    min_area_px: int = 64                 # kill single-pixel speckle
    min_length_mm: float = 1.0            # below the stated resolution -> not a claim
    extent_rule: str = "half_max"         # half_max | threshold -- see events/assemble.py
    extent_fraction: float = 0.5          # half_max: fraction of the peak defining the edge
    track_iou_threshold: float = 0.2      # IoU on position-corrected boxes
    track_join_mm: float = 50.0           # machine-direction gap a line may span
    track_max_gap_frames: int = 2         # frames an event may vanish for and survive


class ScoringConfig(_Section):
    """L7. ASTM D5430 and the pass/hold/reject bands."""

    roll_width_m: float = 1.5
    hold_points_per_100yd2: float = 20.0
    reject_points_per_100yd2: float = 40.0


class ApiConfig(_Section):
    """L8. Where the service listens, and where it keeps rolls and evidence."""

    host: str = "127.0.0.1"
    port: int = 8000
    db_path: Path = Path("linesight.db")
    evidence_dir: Path = Path("results/evidence")


class Config(_Section):
    """The whole run, in one validated object."""

    sku: str = "default"
    seed: int = 0
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    @property
    def bank_path(self) -> Path:
        """Where this SKU's memory bank lives: ``banks/{sku}.npz``."""
        return Path(self.detect.bank_dir) / f"{self.sku}.npz"

    def metres_per_tile(
        self, mm_per_px: float, roi_hw: tuple[int, int] | None = None
    ) -> float:
        """Machine-direction fabric length one scored tile accounts for.

        This is the conversion the false-alarm budget hangs on: a per-tile tail
        fraction becomes a per-100-metre rate by way of this number, so getting
        it wrong scales the guarantee by exactly the error.

        **Tiles cover the web in two dimensions, not one.** A 256 x 1024 frame
        cut into 256 px tiles yields five tiles side by side across the width -
        all at the *same* position down the roll. Charging each of them a full
        tile-stride of fabric would claim five times more inspected length than
        exists, and hand back a threshold five times looser than the operator
        asked for. So when the ROI shape is known, the length one frame covers
        is divided by the number of tiles that frame actually produces.

        Args:
            mm_per_px: measured scale for the frame.
            roi_hw: ``(height, width)`` of the fabric ROI in pixels. Omitting it
                falls back to the along-axis stride alone, which is correct only
                for a web exactly one tile wide.

        Raises:
            ValueError: on a non-positive scale.
        """
        if mm_per_px <= 0.0:
            raise ValueError(f"mm_per_px must be positive, got {mm_per_px}")

        tile_size = self.preprocess.tile_size
        overlap = self.preprocess.tile_overlap

        if roi_hw is None:
            return (tile_size - overlap) * mm_per_px / 1000.0

        # Imported here rather than at module scope: config is a leaf that every
        # layer imports, and one source of truth for the tiling rule beats two.
        from .preprocess.tiling import tile_positions

        height, width = int(roi_hw[0]), int(roi_hw[1])
        n_tiles = len(
            tile_positions(
                height, width, tile_size, overlap, self.preprocess.drop_partial_edge_tiles
            )
        )
        frame_along_m = height * mm_per_px / 1000.0
        return frame_along_m / max(1, n_tiles)


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested mappings merge key-by-key; every other type is replaced wholesale.
    A SKU file therefore states only what differs from the default, and a diff
    of two SKU files is a diff of two fabrics.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_config(sku: str | None = None, config_dir: Path | str = "configs") -> Config:
    """Load ``default.yaml``, overlay ``sku_{sku}.yaml`` if present, validate.

    Args:
        sku: SKU name; looks for ``configs/sku_{sku}.yaml``. None loads defaults.
        config_dir: directory holding the YAML files.

    Raises:
        FileNotFoundError: if ``default.yaml`` is missing, or if a SKU was named
            explicitly and its file does not exist. Silently falling back to
            defaults for a named SKU would run the wrong fabric's geometry and
            report the result as if it were right.
        pydantic.ValidationError: if any key fails its type or bound, including
            a key that does not exist - ``extra="forbid"`` turns a typo in a
            YAML file into an error rather than a setting that never applies.
    """
    directory = Path(config_dir)
    default_path = directory / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"missing base config: {default_path}")

    data = _read_yaml(default_path)

    if sku is not None:
        sku_path = directory / f"sku_{sku}.yaml"
        if not sku_path.exists():
            raise FileNotFoundError(
                f"no config for SKU {sku!r}: expected {sku_path}. "
                f"Create it, or run without --sku to use defaults."
            )
        data = merge_dicts(data, _read_yaml(sku_path))
        data["sku"] = sku

    return Config.model_validate(data)


def dump_config(config: Config, path: Path | str) -> None:
    """Write the fully-resolved config beside a run's artifacts.

    Every run records the exact configuration that produced it - otherwise the
    results table is not reproducible and should not be believed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
