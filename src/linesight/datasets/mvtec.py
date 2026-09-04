"""MVTec AD loader - the reproduction anchor (ADR-003).

Normally mounted read-only at ``/kaggle/input`` rather than downloaded, which is
what ``kaggle_root`` is for. Every function below takes an explicit ``root``, so
a local copy of a single category works identically - which is what a one-off
validation run wants, without pulling the other fourteen categories.

Role: fitting PatchCore on ``carpet`` and landing near the published image AUROC
is what licenses every other number in the results table to be believed. Landing
far from it means there is a bug, and no later table is worth producing until it
is found.

Layout, which every function here assumes:

    {root}/{category}/train/good/000.png          defect-free by construction
    {root}/{category}/test/good/000.png           clean test images
    {root}/{category}/test/{defect}/000.png       anomalous test images
    {root}/{category}/ground_truth/{defect}/000_mask.png

**Licence: CC BY-NC-SA 4.0 - non-commercial.** Used for validation only;
nothing derived from it ships, and the deployed system fits on the mill's own
fabric. Recorded here and in ``docs/datasets.md``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["TEXTURE_CATEGORIES", "MvtecSplit", "kaggle_root", "list_split", "load_pair"]

#: The texture classes - the ones that resemble fabric. The object classes
#: (bottle, screw, ...) are irrelevant to this problem and are not evaluated.
TEXTURE_CATEGORIES: tuple[str, ...] = ("carpet", "grid", "leather")

#: The split subdirectory that is defect-free by construction. Fitting on it is
#: exactly the cold-start assumption being validated.
NORMAL_DIR: str = "good"

_SPLITS: tuple[str, ...] = ("train", "test")


def kaggle_root(dataset: str = "mvtec-ad") -> Path:
    """Locate the mounted dataset under ``/kaggle/input``.

    Kaggle unzips dataset archives, so the category directories can sit either
    directly under the input folder or one level down; both are checked.

    Raises:
        FileNotFoundError: with an explicit "attach the dataset as a notebook
            input" message, because that is always the actual cause.
    """
    base = Path("/kaggle/input") / dataset
    candidates = [base, base / "mvtec_ad", base / "MVTec_AD", base / dataset]
    for candidate in candidates:
        if candidate.is_dir() and any(
            (candidate / category).is_dir() for category in TEXTURE_CATEGORIES
        ):
            return candidate
    raise FileNotFoundError(
        f"MVTec AD is not mounted at {base}. Attach the dataset as a notebook "
        "input before running any code - a stalled input attach at minute forty "
        "is a lost session. Locally, pass an explicit root instead."
    )


class MvtecSplit:
    """One category's train (normal only) or test (normal + defect) split."""

    def __init__(self, root: Path, category: str, split: str) -> None:
        """Args:
        split: ``"train"`` or ``"test"``. Train is defect-free by construction,
            which is exactly the cold-start assumption being validated.

        Raises:
            ValueError: on an unknown split.
            FileNotFoundError: if the category or split directory is absent,
                naming what was actually there - a half-extracted download
                otherwise presents as "zero images", which reads like a code
                bug rather than a data one.
        """
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")

        self.root = Path(root)
        self.category = str(category)
        self.split = split
        self.directory = self.root / self.category / split

        if not self.directory.is_dir():
            present = (
                sorted(p.name for p in (self.root / self.category).iterdir())
                if (self.root / self.category).is_dir()
                else sorted(p.name for p in self.root.iterdir())
                if self.root.is_dir()
                else []
            )
            raise FileNotFoundError(
                f"no {split!r} split at {self.directory}. Present instead: {present}"
            )

    @property
    def image_paths(self) -> list[Path]:
        """Every image in the split, sorted by defect type then filename.

        Sorted so that two runs enumerate in the same order: an evaluation whose
        row order depends on the filesystem is one whose per-image results
        cannot be diffed between runs.
        """
        return sorted(
            (p for p in self.directory.rglob("*.png") if p.is_file()),
            key=lambda p: (p.parent.name, p.name),
        )

    @property
    def defect_types(self) -> list[str]:
        """Subdirectory names under ``test/``; ``good`` plus the anomaly classes."""
        return sorted(p.name for p in self.directory.iterdir() if p.is_dir())

    def mask_path_for(self, image_path: Path) -> Path | None:
        """``ground_truth/{defect}/{stem}_mask.png``, or None for ``good``.

        None means "defect-free", not "mask missing" - ``load_pair`` turns it
        into an all-zero mask so evaluation code needs no special case. A
        genuinely absent mask for an anomalous image returns None too, and that
        is a data problem the caller should notice rather than average away.
        """
        defect = image_path.parent.name
        if defect == NORMAL_DIR:
            return None
        mask = (
            self.root
            / self.category
            / "ground_truth"
            / defect
            / f"{image_path.stem}_mask.png"
        )
        return mask if mask.exists() else None

    def is_anomalous(self, image_path: Path) -> bool:
        """Label from the directory name. The only label MVTec gives at image level."""
        return image_path.parent.name != NORMAL_DIR

    def __len__(self) -> int:
        return len(self.image_paths)

    def __repr__(self) -> str:
        return (
            f"MvtecSplit({self.category!r}, {self.split!r}, "
            f"n={len(self)}, types={self.defect_types})"
        )


def list_split(root: Path | str, category: str, split: str) -> MvtecSplit:
    """Construct a split. The entry point the reproduction probe calls."""
    return MvtecSplit(Path(root), category, split)


def load_pair(
    image_path: Path, mask_path: Path | None, size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and its pixel-precise mask, resized together.

    Returns ``(image_uint8_bgr, mask_uint8_01)``; an all-zero mask stands in for
    a defect-free image so evaluation code needs no special case.

    The mask is resized with nearest-neighbour while the image gets area
    interpolation: a bilinear-resized binary mask acquires fractional values at
    every defect boundary, and thresholding those back to {0, 1} silently moves
    the boundary by a pixel or two in whichever direction rounding falls.

    Raises:
        FileNotFoundError: if the image cannot be read.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image {image_path}")

    if mask_path is None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"could not read mask {mask_path}")
        mask = (raw > 0).astype(np.uint8)

    if size is not None:
        image = cv2.resize(image, (int(size), int(size)), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (int(size), int(size)), interpolation=cv2.INTER_NEAREST)

    return image, mask.astype(np.uint8)
