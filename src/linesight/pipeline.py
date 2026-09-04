"""THE ORCHESTRATOR - L1 through L7, wired together.

Every arrow in the layer diagram is a typed function, and this file is where
they are composed. It owns no algorithm of its own: if logic appears here that
belongs to a layer, it has been put in the wrong place.

    L1 ACQUISITION      FrameSource -> BGR frame + timestamp
    L2 GEOMETRY         frame -> (position_mm, mm_per_px, fabric_roi)
    L3 PREPROCESS       fabric_roi -> List[Tile] with global coords
    L4 DETECTION        Tile -> ScoreMap                        <-- the seam
    L5 CALIBRATION      ScoreMap + threshold -> mask + confidence
    L6 EVENT ASSEMBLY   masks across tiles/frames -> List[Event] in mm
    L7 SCORING          Events -> ASTM D5430 points -> pass/hold/reject

Three procedures are composed out of those layers, and they are the three
entry points worth reading in order:

  * ``fit`` / ``fit_from_source``              learn one SKU from clean fabric
  * ``calibrate`` / ``calibrate_from_source``  derive the threshold from a budget
  * ``run`` / ``iter_frames``                  inspect a roll and score it

``teach_calibrate_inspect`` runs all three against a single continuous stream,
which is what a moving line requires. Every one of them prepares pixels through
``prepare`` and cuts them through ``tiles_of`` - one path, so the fit set and
the inspected frames cannot end up on different distributions.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .acquisition.base import open_source
from .calibrate.threshold import calibrate, confidence_from_score
from .config import Config
from .events.assemble import assemble_frame
from .events.track import EventTracker, PerFrameTracker
from .geometry.aruco import ArucoReader
from .preprocess.flatfield import flat_field
from .preprocess.tiling import assemble_score_map, tile_frame
from .scoring.astm_d5430 import fill_report
from .types import (
    Calibration,
    Detection,
    Event,
    Frame,
    FrameGeometry,
    LatencyRecord,
    RollReport,
    ScoreMap,
)

__all__ = ["FrameResult", "Pipeline", "build_scorer"]


def build_scorer(config: Config, load_bank: bool = True) -> object:
    """Construct the configured ``AnomalyScorer``.

    ``"patchcore"`` loads ``banks/{sku}.npz``; ``"stub"`` needs nothing, which
    is how the whole pipeline runs with no bank, no dataset and no torch.

    Raises:
        FileNotFoundError: if a bank is required and missing - with the exact
            ``linesight fit`` command that would create it.
        ValueError: on an unknown scorer name.
    """
    name = config.detect.scorer
    if name == "stub":
        from .detect.stub import StubScorer

        return StubScorer(seed=config.seed)
    if name == "patchcore":
        from .detect.patchcore import PatchCore

        if not load_bank:
            return PatchCore(config.detect)
        return PatchCore.load(config.bank_path, config.detect)
    if name == "unet":
        from .baselines.unet import UNetScorer

        return UNetScorer.load(config.bank_path, device=config.detect.device)
    raise ValueError(f"unknown scorer {name!r}; expected patchcore, stub, or unet")


class FrameResult:
    """Everything one frame produced. Emitted per frame so a UI can stream."""

    def __init__(
        self,
        frame: Frame,
        geometry: FrameGeometry,
        score_map: np.ndarray,
        detections: list[Detection],
        latency: LatencyRecord,
    ) -> None:
        self.frame = frame
        self.geometry = geometry
        self.score_map = score_map
        self.detections = detections
        self.latency = latency

    def overlay(self, alpha: float = 0.45) -> np.ndarray:
        """Heatmap composited over the frame - the operator UI's main image.

        Normalised per frame **for display only**. The underlying scores stay
        unbounded and uncalibrated; this is the one place a normalisation is
        acceptable, because nothing downstream reads it.
        """
        roi = _crop_roi(self.frame.image, self.geometry.roi)
        scores = self.score_map
        span = float(scores.max() - scores.min())
        normalised = (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)
        heat = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if heat.shape[:2] != roi.shape[:2]:
            heat = cv2.resize(heat, (roi.shape[1], roi.shape[0]))
        base = roi if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0.0)

    def __repr__(self) -> str:
        return (
            f"FrameResult(frame={self.frame.index}, "
            f"along_mm={self.geometry.along_mm:.1f}, "
            f"detections={len(self.detections)})"
        )


def _crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Apply a ``(x, y, w, h)`` ROI, clamped to the image."""
    x, y, w, h = roi
    height, width = image.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, x0 + int(w)), min(height, y0 + int(h))
    return image[y0:y1, x0:x1]


class Pipeline:
    """Fit a SKU, calibrate a threshold, and inspect a roll."""

    def __init__(self, config: Config, scorer: object | None = None) -> None:
        """Args:
        scorer: inject one to bypass ``build_scorer`` - how tests run the
            whole pipeline against ``StubScorer`` with no bank and no torch.
        """
        self.config = config
        self._scorer = scorer
        self.calibration: Calibration | None = None
        self.geometry = ArucoReader(config.geometry)
        self.tracker = (
            EventTracker(config.events)
            if config.events.track_iou_threshold > 0
            else PerFrameTracker(config.events)
        )
        self._latencies: list[LatencyRecord] = []
        self._gap_warnings = 0
        self._roi_hw: tuple[int, int] | None = None

    @property
    def scorer(self) -> object:
        """The configured scorer, built on first use so ``--help`` stays cheap."""
        if self._scorer is None:
            self._scorer = build_scorer(self.config)
        return self._scorer

    # -- the single preprocessing path -------------------------------------- #

    def prepare(
        self, image: np.ndarray, geometry: FrameGeometry | None = None
    ) -> np.ndarray:
        """Raw frame -> the fabric ROI the model actually sees.

        **Every path goes through here: fit, calibrate, and inspect.** They used
        not to, and the consequences were exactly what you would predict and
        nobody noticed for days: the bank was fitted on raw images while
        inference scored flat-fielded, ROI-cropped frames, so the model had
        literally never seen the kind of picture it was being asked to judge.
        Normal fabric scored as anomalous because "normal" had been defined on a
        different distribution. That is a false-alarm generator, and no amount
        of threshold tuning fixes it.

        Keeping it in one function makes the three paths structurally incapable
        of diverging again.
        """
        roi = _crop_roi(image, geometry.roi) if geometry is not None else image
        if self.config.preprocess.flatfield:
            roi = flat_field(roi, sigma_px=self.config.preprocess.flatfield_sigma_px)
        return roi

    def tiles_of(
        self, image: np.ndarray, geometry: FrameGeometry | None = None, frame_index: int = 0
    ) -> list:
        """Raw frame -> prepared tiles, ready for the scorer."""
        roi = self.prepare(image, geometry)
        size = self.config.preprocess.tile_size
        if roi.shape[0] < size or roi.shape[1] < size:
            return []
        return tile_frame(
            roi,
            tile_size=size,
            overlap=self.config.preprocess.tile_overlap,
            frame_index=frame_index,
            drop_partial=self.config.preprocess.drop_partial_edge_tiles,
        )

    # -- fit ---------------------------------------------------------------- #

    def fit(self, normal_dir: Path | str, save: bool = True) -> object:
        """Build this SKU's memory bank from a folder of defect-free fabric.

        Takes roughly 90 s for 30 tiles on CPU. Writes ``banks/{sku}.npz``
        plus the fully-resolved config beside it, so the bank records the
        configuration it was fitted under.

        Raises:
            FileNotFoundError: if the folder holds no usable images.
        """
        tiles = self._load_tiles(normal_dir, limit=self.config.detect.n_normal_tiles)
        scorer = build_scorer(self.config, load_bank=False)
        scorer.fit(tiles)  # type: ignore[attr-defined]

        if save:
            from .config import dump_config

            bank_path = self.config.bank_path
            scorer.save(bank_path)  # type: ignore[attr-defined]
            dump_config(self.config, bank_path.with_suffix(".config.yaml"))

        self._scorer = scorer
        return scorer

    def calibrate(self, clean_dir: Path | str) -> Calibration:
        """Derive the threshold from held-out clean fabric and a stated budget.

        ``clean_dir`` must be disjoint from the fit set. This is the step that
        makes the threshold derived rather than chosen: the operator states a
        false-alarm budget and this returns the score that meets it.

        **Clean fabric is scored through the same path a roll is**, not tile by
        tile in isolation - see ``_clean_tile_maxima``. The conformal guarantee
        is an exchangeability argument, and it only holds if the scores the
        threshold is fitted on come from the same process as the scores it is
        later applied to.

        Raises:
            ValueError: via ``threshold_from_budget`` if the clean sample is too
                small to resolve the requested budget.
        """
        scores = self._clean_tile_maxima(clean_dir)
        self.calibration = calibrate(
            scores,
            self.config.calibration.budget_fa_per_100m,
            self.metres_per_tile,
            self.config.calibration.abstain_multiplier,
            self.config.calibration.stability_margin,
            sku=self.config.sku,
        )
        return self.calibration

    def _clean_tile_maxima(self, clean_dir: Path | str) -> np.ndarray:
        """Per-tile peak scores on clean fabric, measured the way a roll is.

        Inference does not threshold a tile's own score map. It pastes every
        tile of a frame into one mosaic, taking the **max** across the 64 px
        overlaps, and thresholds that. Max-combining two overlapping views of
        the same fabric can only push a pixel's score up, so the mosaic's tail
        sits measurably above any single tile's - about +0.16 at the 99th
        percentile on the bench data. Calibrating on isolated tiles and
        inferring on mosaics therefore under-sets the threshold and blows the
        false-alarm budget, which is the one number this system stakes its
        argument on.

        So clean frames go through ROI -> flat-field -> tile -> score_batch ->
        assemble, exactly as ``_process`` does, and the maxima are read back per
        tile footprint off the assembled mosaic.

        Frames already cut to one tile are scored as-is - there is no mosaic to
        build - and that case is flagged, because its guarantee is the weaker
        one this method exists to avoid.
        """
        images = self._read_images(clean_dir)
        return self._maxima_from_images(images)

    def _maxima_from_images(
        self, images: list[np.ndarray], geometries: list | None = None
    ) -> np.ndarray:
        """Per-tile peak scores over prepared frames, read off the mosaic."""
        tile_size = self.config.preprocess.tile_size
        maxima: list[float] = []
        single_tile_frames = 0

        for index, image in enumerate(images):
            geometry = geometries[index] if geometries else None
            tiles = self.tiles_of(image, geometry, frame_index=index)
            if not tiles:
                continue
            roi = self.prepare(image, geometry)
            if roi.shape[0] == tile_size and roi.shape[1] == tile_size:
                single_tile_frames += 1

            raw = self.scorer.score_batch([t.image for t in tiles])  # type: ignore[attr-defined]
            mosaic = assemble_score_map(
                [ScoreMap(scores=m, tile=t) for m, t in zip(raw, tiles, strict=True)],
                roi.shape[:2],
            )
            maxima.extend(
                float(mosaic[t.y : t.y + tile_size, t.x : t.x + tile_size].max())
                for t in tiles
            )

        if single_tile_frames and single_tile_frames == len(images):
            print(
                "  note: the clean set is single-tile images, so no overlap "
                "mosaic was formed. The threshold is fitted on a slightly "
                "lighter tail than inference sees; supply full frames for the "
                "exchangeable guarantee."
            )

        if not maxima:
            raise ValueError("no usable clean frames")
        return np.asarray(maxima, dtype=np.float32)

    def _read_images(self, folder: Path | str) -> list[np.ndarray]:
        """Every readable image in a folder, in natural order."""
        directory = Path(folder)
        if not directory.is_dir():
            raise FileNotFoundError(f"not a directory: {directory}")
        paths = sorted(
            p
            for p in directory.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            and not p.stem.endswith("_mask")
        )
        images = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"could not decode {path}")
            images.append(image)
        if not images:
            raise FileNotFoundError(f"no readable images in {directory}")
        return images

    # -- learning from the line itself --------------------------------------- #

    def _collect_frames(
        self, source_uri: str | None, n_frames: int, skip: int = 0, label: str = "collecting"
    ) -> tuple[list[np.ndarray], list]:
        """Pull ``n_frames`` off a live source, after discarding ``skip`` of them.

        ``skip`` exists because the fit and calibration sets must be disjoint
        stretches of fabric, and on a moving line the only way to get that is to
        let some fabric go past between them.
        """
        acquisition = self.config.acquisition
        source = open_source(
            acquisition.source_type,
            source_uri or acquisition.uri,
            stride=acquisition.frame_stride,
            **(
                {"glob": acquisition.glob, "synthetic_fps": acquisition.synthetic_fps}
                if acquisition.source_type == "directory"
                else {}
            ),
        )

        images: list[np.ndarray] = []
        geometries: list = []
        seen = 0
        with source:
            for frame in source:
                seen += 1
                if seen <= skip:
                    continue
                try:
                    geometry = self.geometry.read(frame.image, frame.timestamp)
                except RuntimeError:
                    geometry = None
                images.append(frame.image)
                geometries.append(geometry)
                if len(images) % 10 == 0:
                    print(f"  {label}: {len(images)}/{n_frames} frames", end="\r")
                if len(images) >= n_frames:
                    break
        print(f"  {label}: {len(images)} frames collected      ")
        return images, geometries

    def fit_from_source(
        self, source_uri: str | None = None, n_frames: int = 40, save: bool = True
    ) -> object:
        """Learn this fabric by watching clean cloth run past the camera.

        This is the workflow a mill would actually use: thread clean fabric,
        run it, and let the system watch. It is also the only way to guarantee
        the bank is fitted on exactly what inference will see - same lens, same
        lighting, same JPEG compression, same ROI crop, same flat-field. Fitting
        from a folder of photos taken some other way is a distribution mismatch
        wearing a convincing disguise.

        Raises:
            ValueError: if the stream yields too few usable tiles.
        """
        images, geometries = self._collect_frames(
            source_uri, n_frames, label="learning clean fabric"
        )

        tiles: list[np.ndarray] = []
        for index, image in enumerate(images):
            tiles.extend(t.image for t in self.tiles_of(image, geometries[index], index))
        if len(tiles) < 2:
            raise ValueError(
                f"{len(images)} frames yielded only {len(tiles)} tiles - is the "
                "fabric ROI smaller than one tile?"
            )

        limit = self.config.detect.n_normal_tiles
        if limit and len(tiles) > limit:
            rng = np.random.default_rng(self.config.seed)
            picked = rng.choice(len(tiles), size=limit, replace=False)
            tiles = [tiles[i] for i in sorted(picked)]

        print(f"  fitting on {len(tiles)} tiles from {len(images)} frames")
        scorer = build_scorer(self.config, load_bank=False)
        scorer.fit(tiles)  # type: ignore[attr-defined]

        if save:
            from .config import dump_config

            bank_path = self.config.bank_path
            scorer.save(bank_path)  # type: ignore[attr-defined]
            dump_config(self.config, bank_path.with_suffix(".config.yaml"))

        self._scorer = scorer
        return scorer

    def calibrate_from_source(
        self, source_uri: str | None = None, n_frames: int = 60, skip: int = 0
    ) -> Calibration:
        """Derive the threshold from clean fabric still running past the camera.

        ``skip`` frames are discarded first so this stretch of cloth is disjoint
        from the one the bank was fitted on - held-out means held out, and on a
        moving line that means later fabric, not the same fabric again.
        """
        images, geometries = self._collect_frames(
            source_uri, n_frames, skip=skip, label="calibrating on clean fabric"
        )
        scores = self._maxima_from_images(images, geometries)
        self.calibration = calibrate(
            scores,
            self.config.calibration.budget_fa_per_100m,
            self.metres_per_tile,
            self.config.calibration.abstain_multiplier,
            self.config.calibration.stability_margin,
            sku=self.config.sku,
        )
        return self.calibration

    def teach_calibrate_inspect(
        self,
        source_uri: str | None = None,
        n_fit_frames: int = 40,
        n_calibration_frames: int = 60,
        on_phase: object = None,
        on_frame: object = None,
        roll_id: str | None = None,
    ):
        """The whole procedure against ONE continuous stream: learn, calibrate, inspect.

        This is what a mill actually does. Thread clean cloth and start the line.
        The system watches the first stretch and learns what this fabric is;
        watches the next stretch and derives a threshold from the operator's
        false-alarm budget; then inspects everything after that.

        **One connection, consumed in three stages.** Opening the stream three
        times would be simpler and wrong: each connection restarts the camera at
        the same fabric, so the "held-out" calibration set would be the very
        cloth the bank was fitted on. On a moving line, held out means *later*,
        and the only way to guarantee that is to never let go of the stream.

        Args:
            n_fit_frames: clean frames to learn from.
            n_calibration_frames: clean frames after those, for the threshold.
            on_phase: optional ``callable(name, detail)`` for progress.
            on_frame: optional ``callable(FrameResult)`` per inspected frame.

        Returns:
            The scored ``RollReport`` for the inspected stretch.

        Raises:
            ValueError: if the clean stretch is too short to fit or calibrate on.
        """
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        roll_id = roll_id or f"{self.config.sku}-{int(time.time())}"
        announce = on_phase if callable(on_phase) else (lambda *a: None)

        source = self._open_source(source_uri)
        self.geometry.reset()
        self.tracker.reset()
        self._latencies = []
        self._gap_warnings = 0

        required_tiles = math.ceil(
            self.config.calibration.stability_margin
            * 100.0
            / (self.config.calibration.budget_fa_per_100m * max(self.metres_per_tile, 1e-9))
        )
        announce(
            "plan",
            f"budget {self.config.calibration.budget_fa_per_100m} FA/100 m needs "
            f"~{required_tiles} held-out clean tiles",
        )

        fit_images: list[np.ndarray] = []
        fit_geoms: list = []
        cal_images: list[np.ndarray] = []
        cal_geoms: list = []
        furthest_mm = 0.0
        phase = "learn"

        with source:
            for frame in source:
                try:
                    geometry = self.geometry.read(frame.image, frame.timestamp)
                except RuntimeError:
                    geometry = None
                if geometry is not None:
                    furthest_mm = max(furthest_mm, geometry.along_mm)
                if self._roi_hw is None:
                    # Learn the ROI shape from this frame rather than opening a
                    # second connection to the camera to go and measure it.
                    self._roi_hw = self.prepare(frame.image, geometry).shape[:2]

                if phase == "learn":
                    fit_images.append(frame.image)
                    fit_geoms.append(geometry)
                    announce("learn", f"{len(fit_images)}/{n_fit_frames} clean frames")
                    if len(fit_images) >= n_fit_frames:
                        self._fit_from_frames(fit_images, fit_geoms)
                        announce("learn-done", f"bank fitted on {len(fit_images)} frames")
                        phase = "calibrate"
                    continue

                if phase == "calibrate":
                    cal_images.append(frame.image)
                    cal_geoms.append(geometry)
                    announce(
                        "calibrate",
                        f"{len(cal_images)}/{n_calibration_frames} clean frames",
                    )
                    if len(cal_images) >= n_calibration_frames:
                        scores = self._maxima_from_images(cal_images, cal_geoms)
                        self.calibration = calibrate(
                            scores,
                            self.config.calibration.budget_fa_per_100m,
                            self.metres_per_tile,
                            self.config.calibration.abstain_multiplier,
                            self.config.calibration.stability_margin,
                            sku=self.config.sku,
                        )
                        announce(
                            "calibrate-done",
                            f"threshold {self.calibration.threshold:.4f} from "
                            f"{self.calibration.n_clean_tiles} held-out clean tiles",
                        )
                        phase = "inspect"
                    continue

                result = self._process_with_geometry(frame, geometry)
                self.tracker.update(result.detections)
                if callable(on_frame):
                    on_frame(result)

        if phase != "inspect":
            raise ValueError(
                f"the stream ended during the {phase!r} phase - the clean lead-in "
                f"is too short. It needs {n_fit_frames} + {n_calibration_frames} "
                "clean frames before the defects start."
            )

        report = self.finalise(roll_id, self.tracker.finalise(), furthest_mm)
        report.started_at = started
        report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return report

    def _fit_from_frames(self, images: list[np.ndarray], geometries: list) -> None:
        """Fit the bank from frames already pulled off a stream."""
        tiles: list[np.ndarray] = []
        for index, image in enumerate(images):
            tiles.extend(t.image for t in self.tiles_of(image, geometries[index], index))
        if len(tiles) < 2:
            raise ValueError(
                f"{len(images)} clean frames yielded only {len(tiles)} tiles - is the "
                "fabric ROI smaller than one tile?"
            )

        limit = self.config.detect.n_normal_tiles
        if limit and len(tiles) > limit:
            rng = np.random.default_rng(self.config.seed)
            picked = rng.choice(len(tiles), size=limit, replace=False)
            tiles = [tiles[i] for i in sorted(picked)]

        scorer = build_scorer(self.config, load_bank=False)
        scorer.fit(tiles)
        self._scorer = scorer

    def _open_source(self, source_uri: str | None):
        """Construct the configured frame source. One place, so the kwargs agree."""
        acquisition = self.config.acquisition
        if acquisition.source_type == "directory":
            extra = {
                "glob": acquisition.glob,
                "synthetic_fps": acquisition.synthetic_fps,
            }
        elif acquisition.source_type == "mjpeg":
            extra = {"drop_stale": acquisition.drop_stale}
        else:
            extra = {}
        return open_source(
            acquisition.source_type,
            source_uri or acquisition.uri,
            stride=acquisition.frame_stride,
            **extra,
        )

    # -- run ---------------------------------------------------------------- #

    def run(self, source_uri: str | None = None, roll_id: str | None = None) -> RollReport:
        """Inspect a whole roll and return its scored report.

        Consumes ``iter_frames`` to exhaustion, then finalises the tracker and
        applies ASTM scoring. The offline equivalent of a whole shift:
        ``python -m linesight run --source data/aitex/roll_02/``.
        """
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        roll_id = roll_id or f"{self.config.sku}-{int(time.time())}"

        furthest_mm = 0.0
        for result in self.iter_frames(source_uri):
            self.tracker.update(result.detections)
            furthest_mm = max(furthest_mm, result.geometry.along_mm)

        report = self.finalise(roll_id, self.tracker.finalise(), furthest_mm)
        report.started_at = started
        report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return report

    def iter_frames(self, source_uri: str | None = None) -> Iterator[FrameResult]:
        """Stream per-frame results. The generator the operator UI consumes.

        Per frame: geometry -> ROI -> flat-field -> tile -> score_batch (one
        forward pass for all tiles) -> assemble -> track.

        Raises:
            RuntimeError: if no calibration has been set. Running without a
                threshold would mean inventing one, which is the one thing this
                system is built not to do.
        """
        if self.calibration is None:
            raise RuntimeError(
                "no calibration - run `linesight calibrate` first, or set "
                "Pipeline.calibration. Inspecting without a threshold would mean "
                "inventing one."
            )

        acquisition = self.config.acquisition
        source = open_source(
            acquisition.source_type,
            source_uri or acquisition.uri,
            stride=acquisition.frame_stride,
            **(
                {"glob": acquisition.glob, "synthetic_fps": acquisition.synthetic_fps}
                if acquisition.source_type == "directory"
                else {}
            ),
        )

        self.geometry.reset()
        self.tracker.reset()
        self._latencies = []
        self._gap_warnings = 0

        with source:
            for frame in source:
                yield self._process(frame)

    def finalise(
        self, roll_id: str, events: list[Event], roll_length_mm: float = 0.0
    ) -> RollReport:
        """Close open tracks, apply ASTM scoring, and assemble the report."""
        roll_length_m = max(roll_length_mm / 1000.0, 1e-6)
        width_m = self.config.scoring.roll_width_m

        if self.calibration is not None:
            for event in events:
                event.confidence = confidence_from_score(event.max_score, self.calibration)

        report = RollReport(
            roll_id=roll_id,
            sku=self.config.sku,
            roll_length_m=roll_length_m,
            width_m=width_m,
            events=events,
            calibration=self.calibration,
            gap_warnings=self._gap_warnings,
            meta={
                "frame_stride": self.config.acquisition.frame_stride,
                "scorer": self.config.detect.scorer,
                "n_frames": len(self._latencies),
                "latency_ms": self.latency_table,
                "spatial_resolution_mm": self.spatial_resolution_mm,
                "extent_rule": self.config.events.extent_rule,
            },
        )
        return fill_report(
            report,
            self.config.scoring.hold_points_per_100yd2,
            self.config.scoring.reject_points_per_100yd2,
        )

    # -- internals ---------------------------------------------------------- #

    def _process(self, frame: Frame) -> FrameResult:
        """One frame, all the way from pixels to detections, timed per stage."""
        t0 = time.perf_counter()
        geometry = self.geometry.read(frame.image, frame.timestamp)
        geometry_ms = (time.perf_counter() - t0) * 1000
        return self._process_with_geometry(frame, geometry, geometry_ms)

    def _process_with_geometry(
        self, frame: Frame, geometry: FrameGeometry, geometry_ms: float = 0.0
    ) -> FrameResult:
        """The rest of the frame pipeline, given geometry that is already read.

        Split out so the workflow driver can read geometry once per frame across
        all three phases, rather than reading it twice on inspected frames.
        """
        latency = LatencyRecord(frame_index=frame.index)
        latency.geometry_ms = geometry_ms
        if geometry.gap_warning:
            self._gap_warnings += 1

        t0 = time.perf_counter()
        roi = self.prepare(frame.image, geometry)
        tiles = self.tiles_of(frame.image, geometry, frame_index=frame.index)
        latency.preprocess_ms = (time.perf_counter() - t0) * 1000

        # One forward pass for every tile in the frame - the single biggest CPU
        # win available, and the reason the protocol has score_batch at all.
        t0 = time.perf_counter()
        raw_maps = self.scorer.score_batch([t.image for t in tiles])  # type: ignore[attr-defined]
        latency.backbone_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        score_maps = [ScoreMap(scores=m, tile=t) for m, t in zip(raw_maps, tiles, strict=True)]
        detections = assemble_frame(
            score_maps,
            roi.shape[:2],
            geometry,
            self.calibration,
            self.config.events,
            frame.index,
        )
        latency.assemble_ms = (time.perf_counter() - t0) * 1000

        self._latencies.append(latency)
        return FrameResult(
            frame=frame,
            geometry=geometry,
            score_map=assemble_score_map(score_maps, roi.shape[:2]),
            detections=detections,
            latency=latency,
        )

    def _load_tiles(self, folder: Path | str, limit: int | None = None) -> list[np.ndarray]:
        """Read a folder of images and cut them to the configured tile size.

        Runs every image through ``tiles_of``, so a bank fitted from a folder
        sees the same flat-fielding and tiling as one fitted from a live stream.
        """
        tiles: list[np.ndarray] = []
        for image in self._read_images(folder):
            tiles.extend(t.image for t in self.tiles_of(image))
            if limit is not None and len(tiles) >= limit:
                return tiles[:limit]
        if not tiles:
            raise FileNotFoundError(f"no usable tiles in {folder}")
        return tiles

    # -- introspection ------------------------------------------------------ #

    @property
    def latency_table(self) -> dict[str, float]:
        """Median ms per stage over the run - the published latency table.

        Reported together with ``frame_stride``: a per-frame latency that hides
        a sampling factor of 3 is not a measurement, it is a claim. Median
        rather than mean, so one cold first frame does not set the headline.
        """
        if not self._latencies:
            return {}
        stages = ("geometry", "preprocess", "backbone", "nn_search", "assemble")
        table = {
            stage: float(np.median([getattr(r, f"{stage}_ms") for r in self._latencies]))
            for stage in stages
        }
        table["total"] = float(np.median([r.total_ms for r in self._latencies]))
        table["frame_stride"] = float(self.config.acquisition.frame_stride)
        return table

    @property
    def spatial_resolution_mm(self) -> float:
        """The finest extent this configuration can honestly resolve, in mm.

        Three things blur a point defect into a broader response, and they
        compound: each patch embedding describes a ``tile_size / grid_size``
        square of fabric; the 3x3 neighbourhood pool smears every patch across
        its neighbours; and the score map is Gaussian-smoothed before
        thresholding. A defect narrower than the sum is detected and located
        correctly but **measured** as roughly this wide.

        Reported so nobody reads a 46 mm extent as a measurement of a 6 mm slub.
        An extent at or below this figure means "at the resolution limit", not
        "this defect is 46 mm".
        """
        detect = self.config.detect
        scorer = self._scorer
        grid = getattr(getattr(scorer, "_backbone", None), "grid_size", 0) or 20
        patch_px = self.config.preprocess.tile_size / grid
        footprint_px = patch_px * detect.neighbourhood_pool + 2 * detect.smooth_sigma_px
        return footprint_px * (self.geometry._last_mm_per_px or 1.0)

    @property
    def metres_per_tile(self) -> float:
        """Machine-direction fabric length one scored tile accounts for.

        The false-alarm budget is converted through this number, so it has to
        reflect how the roll is *actually* tiled - including the tiles that sit
        side by side across the web. It therefore peeks at one frame of the
        configured source to learn the ROI shape, because calibration runs
        before the roll does and the shape is not knowable from config alone.

        Falls back to the nominal 1 mm/px of scanned imagery, and to the
        along-axis stride, when no frame can be read.
        """
        mm_per_px = self.geometry._last_mm_per_px or 1.0
        return self.config.metres_per_tile(mm_per_px, self._probe_roi_hw())

    def _probe_roi_hw(self) -> tuple[int, int] | None:
        """Shape of one fabric ROI, from a single frame of the configured source.

        Cached: it is a property of the rig, not of the frame. Returns None if
        the source cannot be read, so calibration degrades to the along-axis
        figure rather than failing.
        """
        if self._roi_hw is not None:
            return self._roi_hw
        acquisition = self.config.acquisition
        if not acquisition.uri:
            return None
        try:
            source = open_source(
                acquisition.source_type,
                acquisition.uri,
                stride=acquisition.frame_stride,
                **(
                    {"glob": acquisition.glob, "synthetic_fps": acquisition.synthetic_fps}
                    if acquisition.source_type == "directory"
                    else {}
                ),
            )
            with source:
                for frame in source:
                    roi = self.config.geometry.roi
                    if roi is not None:
                        self._roi_hw = (int(roi[3]), int(roi[2]))
                    else:
                        self._roi_hw = frame.image.shape[:2]
                    break
        except (OSError, ValueError, TypeError):
            return None
        return self._roi_hw
