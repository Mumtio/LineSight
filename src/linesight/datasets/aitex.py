"""AITEX Fabric Image Database - the local reference input, and the LOFO study.

245 images, 7 fabric structures, 4096x256 px. 140 defect-free (20 per fabric),
105 defective across 12 defect types, each with a ``_mask.png``.

The filename is the metadata: ``nnnn_ddd_ff.png``, where ``ff`` is the fabric
code and ``ddd`` the defect code (``000`` = defect-free). So splitting the data
by fabric construction - which is what measures cross-construction
generalisation - is a string parse rather than a labelling effort. See
``parse_name`` and ``lofo_splits``.

Two dataset quirks are handled here rather than at every call site, and both
would corrupt results silently if they were not: the clean-code spelling is not
consistent (``_is_clean_code``), and the strips carry scanner padding that
differs between the clean and the defective sets (``fabric_extent``).

Licence: cite the AFID paper. See ``docs/datasets.md``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "EDGE_MARGIN",
    "NO_DEFECT",
    "WHITE_LEVEL",
    "AitexImage",
    "crop_to_fabric",
    "cut_tiles",
    "fabric_extent",
    "list_images",
    "load_image",
    "load_mask",
    "lofo_splits",
    "normal_tiles_for_fabric",
    "parse_name",
]

#: Defect code meaning "no defect". Its images are the fit set.
#:
#: The dataset is not perfectly consistent: most clean images are ``_000_`` but
#: ``0018_00_01.png`` uses a two-digit code. Comparing against the literal
#: string would classify that image as DEFECTIVE, dropping it from the fit set
#: and counting it as a defect in every evaluation - so the test is "all zeros".
NO_DEFECT: str = "000"

_MASK_SUFFIX = "_mask"


def _is_clean_code(defect_code: str) -> bool:
    """True for ``000``, ``00``, or any other all-zero spelling."""
    return set(defect_code) == {"0"}


def _is_mask_name(stem: str) -> bool:
    """True for ``..._mask``, and also ``..._mask1`` / ``..._mask2``.

    Several AITEX defects carry more than one annotation, saved as numbered
    mask files. Matching only the exact ``_mask`` suffix lets ``_mask1`` parse
    as a fabric image - a binary annotation would then be scored as if it were
    fabric, which corrupts any evaluation it lands in.
    """
    return _MASK_SUFFIX in stem



#: Columns brighter than this on average are scanner padding, not fabric.
WHITE_LEVEL: int = 200

#: Pixels to inset past the padding boundary, dropping the scan-edge band.
EDGE_MARGIN: int = 24


def fabric_extent(image: np.ndarray, white_level: int = WHITE_LEVEL) -> tuple[int, int]:
    """Column range ``[x0, x1)`` holding actual fabric, excluding scanner padding.

    **AITEX strips are padded with blank white, and the padding is not
    consistent between the clean and defective sets.** Measured across the
    dataset: fabric 00's clean strips carry 1,229 white columns of 4,096 (30% of
    the strip) while its defective strips carry as few as 35; fabric 03 is
    similar, fabric 02 has ~240, and fabrics 04 and 06 have none.

    Left uncropped this poisons everything downstream. Padding tiles enter the
    fit set, so the memory bank spends its coreset learning that blank white is
    normal fabric. The padding boundary is a hard edge unlike any weave, so it
    scores as a defect on every strip that has one. And because clean and
    defective strips are padded differently, any comparison between them is
    partly a comparison of padding.

    Returns the widest contiguous run of non-padding columns.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    is_fabric = grey.mean(axis=0) <= white_level
    if not is_fabric.any():
        return 0, int(grey.shape[1])

    best_start = best_len = run_start = run_len = 0
    for x, fabric in enumerate(is_fabric):
        if fabric:
            if run_len == 0:
                run_start = x
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    return best_start, best_start + best_len


def crop_to_fabric(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    white_level: int = WHITE_LEVEL,
    margin_px: int = EDGE_MARGIN,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop a strip (and its mask, identically) to the fabric region.

    ``margin_px`` insets past the padding boundary. The boundary itself is a
    band of scan artefact - visibly unlike weave, and the model is quite right
    to score it as anomalous - but it is not a fabric defect, and leaving it in
    puts a false alarm at the same position on every single strip. This is the
    dataset's version of the selvedge that ``geometry.fabric_roi`` crops away on
    the rig.
    """
    x0, x1 = fabric_extent(image, white_level)
    x0 = min(x0 + margin_px, max(0, x1 - 1))
    x1 = max(x1 - margin_px, x0 + 1)
    return image[:, x0:x1], (None if mask is None else mask[:, x0:x1])


def parse_name(path: Path | str) -> tuple[str, str, str]:
    """``nnnn_ddd_ff.png`` -> ``(number, defect_code, fabric_code)``.

    Raises:
        ValueError: on a filename that does not match the pattern - including a
            ``_mask.png``, which callers should filter before parsing. Silently
            accepting a mask would put ground truth into the fit set.
    """
    stem = Path(path).stem
    if _is_mask_name(stem):
        raise ValueError(f"{stem} is a mask, not an image - filter masks before parsing")
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"{stem} does not match AITEX's nnnn_ddd_ff pattern")
    return parts[0], parts[1], parts[2]


class AitexImage:
    """One AITEX image and its parsed identity."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._number, self._defect, self._fabric = parse_name(self.path)

    @property
    def number(self) -> str:
        return self._number

    @property
    def fabric_code(self) -> str:
        """``ff`` - the fabric structure. The unit that LOFO holds out."""
        return self._fabric

    @property
    def defect_code(self) -> str:
        """``ddd``; ``"000"`` means defect-free."""
        return self._defect

    @property
    def is_defective(self) -> bool:
        return not _is_clean_code(self._defect)

    @property
    def mask_paths(self) -> list[Path]:
        """Every mask for this image: ``_mask.png`` plus any ``_mask1/2/...``.

        AITEX keeps masks in a sibling ``Mask_images/`` directory, so this looks
        beside the image first and then in that sibling. A few defects are
        annotated across two files; returning both is what lets ``load_mask``
        union them into the complete ground truth.
        """
        if not self.is_defective:
            return []
        for folder in (
            self.path.parent,
            self.path.parent.parent / "Mask_images",
            self.path.parent.parent / "mask_images",
        ):
            if not folder.is_dir():
                continue
            found = sorted(folder.glob(f"{self.path.stem}{_MASK_SUFFIX}*{self.path.suffix}"))
            if found:
                return found
        return []

    @property
    def mask_path(self) -> Path | None:
        """The first mask, or None. Prefer ``mask_paths`` when unioning."""
        masks = self.mask_paths
        return masks[0] if masks else None

    def __repr__(self) -> str:
        kind = f"defect {self._defect}" if self.is_defective else "clean"
        return f"AitexImage({self.path.name}, fabric={self._fabric}, {kind})"


def list_images(root: Path | str, include_masks: bool = False) -> list[AitexImage]:
    """Every AITEX image under ``root``, mask files excluded by default.

    Recursive, because AITEX ships as ``Defect_images/``, ``NODefect_images/``
    and ``Mask_images/`` and callers should be able to point at the parent.

    Raises:
        FileNotFoundError: if ``root`` does not exist.
    """
    directory = Path(root)
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} does not exist - run scripts/fetch_data.sh first"
        )

    images: list[AitexImage] = []
    for path in sorted(directory.rglob("*.png")):
        if _is_mask_name(path.stem):
            continue
        try:
            images.append(AitexImage(path))
        except ValueError:
            continue  # not an AITEX-named file; ignore rather than crash
    return images


def load_image(path: Path | str, grayscale: bool = False) -> np.ndarray:
    """Read one 4096x256 strip as uint8 BGR (or single-channel).

    Raises:
        OSError: if the file cannot be decoded. Loud, not skipped - a strip that
            silently fails to load is a stretch of fabric nobody inspected.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise OSError(f"could not decode {path}")
    return image


def load_mask(path: Path | str | list[Path]) -> np.ndarray:
    """Read one or more ``_mask*.png`` as a single uint8 {0, 1} array.

    Given a list, the masks are unioned: a defect annotated across two files is
    one defect, and evaluating against only half of it would understate recall.
    """
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if not paths:
        raise ValueError("no mask paths given")

    combined: np.ndarray | None = None
    for item in paths:
        mask = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"could not decode mask {item}")
        binary = (mask > 0).astype(np.uint8)
        combined = binary if combined is None else np.maximum(combined, binary)
    return combined


def cut_tiles(
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 0,
    mask: np.ndarray | None = None,
    crop_padding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray | None, tuple[int, int]]]:
    """Cut a 4096x256 strip into square tiles; masks come along unchanged.

    Returns ``(tile, mask_tile_or_None, (x, y))`` triples. Cut first, then
    score: putting a 4096-wide strip through a 320 px backbone would throw away
    the resolution the defects live in.

    Reuses ``preprocess.tiling`` so the strip cutter and the frame tiler cannot
    drift apart; a coordinate convention that holds in one and not the other is
    the kind of bug that only shows up in a report.

    ``crop_padding`` strips the scanner padding first - see ``fabric_extent``.
    Coordinates are then relative to the cropped fabric, not the raw file.
    """
    from ..preprocess.tiling import tile_positions

    if crop_padding:
        image, mask = crop_to_fabric(image, mask)

    height, width = image.shape[:2]
    size = min(tile_size, height, width)
    positions = tile_positions(height, width, size, min(overlap, size - 1))

    out: list[tuple[np.ndarray, np.ndarray | None, tuple[int, int]]] = []
    for x, y in positions:
        tile = image[y : y + size, x : x + size]
        mask_tile = None if mask is None else mask[y : y + size, x : x + size]
        out.append((tile, mask_tile, (x, y)))
    return out


def lofo_splits(paths: list[Path]) -> Iterator[tuple[str, list[Path], list[Path]]]:
    """Group AITEX images by fabric code.

    Yields ``(held_out_code, fit_normals, eval_defectives)``: the bank is fitted
    on the held-out fabric's own defect-free images and evaluated against every
    *other* fabric's defective images. This measures what happens when a mill
    onboards a construction the system has never seen.
    """
    by_fabric: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: {"normal": [], "defect": []}
    )
    for path in paths:
        try:
            _, defect, fabric = parse_name(path)
        except ValueError:
            continue
        key = "normal" if _is_clean_code(defect) else "defect"
        by_fabric[fabric][key].append(Path(path))

    for code in sorted(by_fabric):
        normals = sorted(by_fabric[code]["normal"])
        if not normals:
            # Fabric 08 has one defective image and no clean ones. A fold with
            # nothing to fit a bank on is not a fold; yielding it would fail
            # inside the caller's loop with a much less obvious message.
            continue
        others = [q for c, v in by_fabric.items() if c != code for q in v["defect"]]
        yield code, normals, sorted(others)


def normal_tiles_for_fabric(
    root: Path | str, fabric_code: str, n_tiles: int = 30, tile_size: int = 256, seed: int = 0
) -> list[np.ndarray]:
    """Defect-free tiles for one fabric code - the input to ``PatchCore.fit``.

    Used by ``probes/p02_memory_bank.py`` and by ``linesight fit`` when the
    SKU's fit folder is an AITEX directory.
    Tiles are drawn without replacement across all that fabric's clean strips,
    so the bank sees the whole fabric rather than one strip's local quirks.

    Raises:
        ValueError: if the fabric code has no defect-free images, or if fewer
            tiles exist than were asked for.
    """
    clean = [
        img
        for img in list_images(root)
        if img.fabric_code == fabric_code and not img.is_defective
    ]
    if not clean:
        raise ValueError(f"no defect-free AITEX images for fabric code {fabric_code!r}")

    pool: list[np.ndarray] = []
    for img in clean:
        strip = load_image(img.path)
        pool.extend(tile for tile, _, _ in cut_tiles(strip, tile_size))

    if len(pool) < n_tiles:
        raise ValueError(
            f"fabric {fabric_code} yields only {len(pool)} tiles of {tile_size} px, "
            f"but {n_tiles} were requested"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pool), size=n_tiles, replace=False)
    return [pool[i] for i in sorted(chosen)]
