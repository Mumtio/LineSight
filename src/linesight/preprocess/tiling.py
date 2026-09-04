"""Frame -> overlapping tiles, and per-tile results -> back to frame.

Tiling exists because the backbone eats a fixed 320 px input while a frame is
megapixels: cutting first preserves the resolution a 1 mm defect lives in.
Tiles overlap (64 px by default) so a defect straddling a boundary is whole in
at least one of them, and overlaps are max-combined on the way back - a defect
seen in either of two views is a defect.

The invariant ``tests/test_tiling.py`` locks: **reassembling the tiles
reproduces the input exactly, and every tile's stored (x, y) maps back to its
true origin.** Everything downstream reports positions in millimetres, so an
off-by-one here becomes a wrong defect location on a QA report - which is worse
than no report.
"""

from __future__ import annotations

import numpy as np

from ..types import ScoreMap, Tile

__all__ = ["assemble_score_map", "paste_tiles", "tile_frame", "tile_positions"]

_REDUCERS: tuple[str, ...] = ("max", "mean", "last")


def _axis_positions(length: int, tile_size: int, overlap: int, drop_partial: bool) -> list[int]:
    """Tile origins along one axis.

    The last position is pulled back flush with the edge rather than left short,
    so coverage is complete and no tile is ever partially out of bounds. That
    makes the final stride smaller than the rest - deliberate, and the reason
    overlaps must be max-combined rather than averaged.
    """
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if not drop_partial and positions[-1] != last:
        positions.append(last)
    return positions


def tile_positions(
    height: int, width: int, tile_size: int, overlap: int, drop_partial: bool = False
) -> list[tuple[int, int]]:
    """Top-left corners covering a (height, width) image.

    Stride is ``tile_size - overlap``. The last row and column are pulled back
    flush with the edge rather than left short, so coverage is complete and no
    tile is ever partially out of bounds.

    Args:
        drop_partial: if True, discard a trailing tile instead of pulling it
            flush. Loses edge coverage; only for ablations.

    Returns:
        ``(x, y)`` pairs in row-major order.

    Raises:
        ValueError: if ``overlap >= tile_size``, if ``overlap`` is negative, or
            if the image is smaller than one tile on either axis.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap {overlap} must be smaller than tile_size {tile_size}")
    if height < tile_size or width < tile_size:
        raise ValueError(
            f"image {height}x{width} is smaller than one {tile_size}x{tile_size} tile"
        )

    xs = _axis_positions(width, tile_size, overlap, drop_partial)
    ys = _axis_positions(height, tile_size, overlap, drop_partial)
    return [(x, y) for y in ys for x in xs]


def tile_frame(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    frame_index: int = 0,
    drop_partial: bool = False,
) -> list[Tile]:
    """Cut a fabric ROI into ``Tile`` objects carrying their global coords."""
    height, width = image.shape[:2]
    positions = tile_positions(height, width, tile_size, overlap, drop_partial)
    return [
        Tile(
            image=image[y : y + tile_size, x : x + tile_size],
            x=x,
            y=y,
            frame_index=frame_index,
            tile_index=index,
        )
        for index, (x, y) in enumerate(positions)
    ]


def paste_tiles(
    tiles: list[Tile],
    patches: list[np.ndarray],
    out_hw: tuple[int, int],
    reduce: str = "max",
) -> np.ndarray:
    """Paste per-tile arrays back into one frame-sized array.

    Args:
        reduce: how to combine overlapping regions. ``"max"`` for score maps and
            masks (a defect seen in either tile is a defect); ``"mean"`` for
            reconstruction checks; ``"last"`` for the exactness probe.

    Returns:
        An array of shape ``out_hw + patch.shape[2:]``, dtype matching the
        patches (``mean`` promotes to float32 while accumulating, then casts
        back).

    Raises:
        ValueError: if ``len(tiles) != len(patches)``, if a patch does not match
            its tile's spatial shape, or on an unknown ``reduce``.
    """
    if reduce not in _REDUCERS:
        raise ValueError(f"reduce must be one of {_REDUCERS}, got {reduce!r}")
    if len(tiles) != len(patches):
        raise ValueError(f"got {len(tiles)} tiles but {len(patches)} patches")
    if not tiles:
        raise ValueError("nothing to paste: tiles is empty")

    for tile, patch in zip(tiles, patches, strict=True):
        if patch.shape[:2] != tile.image.shape[:2]:
            raise ValueError(
                f"patch {patch.shape[:2]} does not match tile {tile.image.shape[:2]} "
                f"at (x={tile.x}, y={tile.y})"
            )

    dtype = patches[0].dtype
    trailing = patches[0].shape[2:]
    shape = (int(out_hw[0]), int(out_hw[1]), *trailing)

    if reduce == "mean":
        total = np.zeros(shape, dtype=np.float32)
        count = np.zeros(shape[:2], dtype=np.float32)
        for tile, patch in zip(tiles, patches, strict=True):
            h, w = patch.shape[:2]
            total[tile.y : tile.y + h, tile.x : tile.x + w] += patch
            count[tile.y : tile.y + h, tile.x : tile.x + w] += 1.0
        count = np.maximum(count, 1.0)
        if trailing:
            count = count.reshape(count.shape + (1,) * len(trailing))
        return (total / count).astype(dtype)

    if reduce == "max":
        # Start at -inf (or the dtype minimum) so a genuine zero still wins over
        # "never written", and untouched pixels end up at zero rather than at a
        # sentinel that would read as a defect.
        out = np.zeros(shape, dtype=dtype)
        written = np.zeros(shape[:2], dtype=bool)
        for tile, patch in zip(tiles, patches, strict=True):
            h, w = patch.shape[:2]
            ys, xs = slice(tile.y, tile.y + h), slice(tile.x, tile.x + w)
            region = out[ys, xs]
            seen = written[ys, xs]
            mask = seen if not trailing else seen.reshape(seen.shape + (1,) * len(trailing))
            out[ys, xs] = np.where(mask, np.maximum(region, patch), patch)
            written[ys, xs] = True
        return out

    out = np.zeros(shape, dtype=dtype)
    for tile, patch in zip(tiles, patches, strict=True):
        h, w = patch.shape[:2]
        out[tile.y : tile.y + h, tile.x : tile.x + w] = patch
    return out


def assemble_score_map(score_maps: list[ScoreMap], out_hw: tuple[int, int]) -> np.ndarray:
    """Frame-sized float32 score map from this frame's tile score maps.

    Max-combines the 64 px overlap bands. This is the first step of event
    assembly (``events/assemble.py``), kept here because pasting by stored tile
    coordinates is a tiling concern rather than an event one.
    """
    if not score_maps:
        return np.zeros((int(out_hw[0]), int(out_hw[1])), dtype=np.float32)
    tiles = [sm.tile for sm in score_maps]
    patches = [sm.scores.astype(np.float32, copy=False) for sm in score_maps]
    return paste_tiles(tiles, patches, out_hw, reduce="max")
