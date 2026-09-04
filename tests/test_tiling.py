"""Tiling invariants. An off-by-one here becomes a wrong defect position on a
QA report, which is worse than no report at all.

The probe question (``probes/p04_tiling.py``): does the tiler produce 512x512
tiles with 64 px overlap and correct global coordinates for a 1920x1080 frame?
PASS IF reassembling the tiles reproduces the input exactly, and every tile's
stored (x, y) maps back to its true origin.
"""

from __future__ import annotations

import numpy as np
import pytest

from linesight.preprocess.tiling import paste_tiles, tile_frame, tile_positions


@pytest.fixture
def frame() -> np.ndarray:
    """A 1920x1080 frame whose every pixel encodes its own coordinates.

    Reconstruction bugs that a flat or random image would hide - a swapped x/y,
    a tile pasted one stride off - become exact mismatches against this.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)


class TestTilePositions:
    def test_stride_is_size_minus_overlap(self) -> None:
        positions = tile_positions(1080, 1920, tile_size=512, overlap=64)
        xs = sorted({x for x, _ in positions})
        assert xs[1] - xs[0] == 512 - 64

    def test_coverage_is_complete(self) -> None:
        # The last row and column are pulled flush with the edge rather than
        # left short, so no fabric goes uninspected.
        positions = tile_positions(1080, 1920, tile_size=512, overlap=64)
        assert max(x for x, _ in positions) + 512 == 1920
        assert max(y for _, y in positions) + 512 == 1080

    def test_no_tile_leaves_the_frame(self) -> None:
        for x, y in tile_positions(1080, 1920, tile_size=512, overlap=64):
            assert x >= 0 and x + 512 <= 1920
            assert y >= 0 and y + 512 <= 1080

    def test_exact_fit_produces_no_duplicate_flush_tile(self) -> None:
        # 512 + 3*(512-64) = 1856; a 1856-wide image tiles exactly.
        positions = tile_positions(512, 1856, tile_size=512, overlap=64)
        assert len(positions) == len({p for p in positions})
        assert len(positions) == 4

    def test_overlap_must_be_smaller_than_the_tile(self) -> None:
        with pytest.raises(ValueError):
            tile_positions(1080, 1920, tile_size=512, overlap=512)

    def test_image_smaller_than_one_tile_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            tile_positions(100, 100, tile_size=512, overlap=64)


class TestTileFrame:
    def test_tiles_are_square_and_the_requested_size(self, frame: np.ndarray) -> None:
        for tile in tile_frame(frame, tile_size=512, overlap=64):
            assert tile.image.shape[:2] == (512, 512)

    def test_stored_coords_map_back_to_the_true_origin(self, frame: np.ndarray) -> None:
        # The core claim of the probe: (x, y) is not decorative.
        for tile in tile_frame(frame, tile_size=512, overlap=64):
            expected = frame[tile.y : tile.y + 512, tile.x : tile.x + 512]
            np.testing.assert_array_equal(tile.image, expected)

    def test_tile_indices_are_dense_and_ordered(self, frame: np.ndarray) -> None:
        tiles = tile_frame(frame, tile_size=512, overlap=64, frame_index=7)
        assert [t.tile_index for t in tiles] == list(range(len(tiles)))
        assert all(t.frame_index == 7 for t in tiles)


class TestRoundTrip:
    def test_reassembly_reproduces_the_input_exactly(self, frame: np.ndarray) -> None:
        # PASS IF of the probe, locked as a test.
        tiles = tile_frame(frame, tile_size=512, overlap=64)
        rebuilt = paste_tiles(tiles, [t.image for t in tiles], frame.shape[:2], reduce="last")
        np.testing.assert_array_equal(rebuilt, frame)

    def test_max_reduce_wins_in_overlaps(self) -> None:
        # A defect seen in either of two overlapping tiles is a defect.
        image = np.zeros((512, 1024), dtype=np.uint8)
        tiles = tile_frame(image, tile_size=512, overlap=64)
        patches = [np.full((512, 512), i * 10, dtype=np.uint8) for i in range(len(tiles))]
        rebuilt = paste_tiles(tiles, patches, image.shape[:2], reduce="max")
        assert rebuilt.max() == (len(tiles) - 1) * 10

    def test_mismatched_patch_count_is_an_error(self, frame: np.ndarray) -> None:
        tiles = tile_frame(frame, tile_size=512, overlap=64)
        with pytest.raises(ValueError):
            paste_tiles(tiles, [tiles[0].image], frame.shape[:2])
