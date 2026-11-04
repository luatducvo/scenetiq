"""TSDF fusion on Open3D itself; skipped where the GPU stack (open3d) is not installed.

Run it on a CPU machine with: uv run --with open3d==0.20.0 pytest tests/unit/test_fusion.py
"""

import numpy as np
import pytest

from scenetiq.config import FusionOptions
from scenetiq.domain import Camera
from scenetiq.pipeline.fusion import fuse
from scenetiq.pipeline.view_data import ViewData

pytest.importorskip("open3d")

HEIGHT, WIDTH = 120, 160
FOCAL = 100.0
WALL_Z = 3.01
OPTIONS = FusionOptions(voxel_size=0.05)
# A pinhole View at the origin sees x = +-z * (W/2) / f and y = +-z * (H/2) / f.
WALL_X = WALL_Z * 80 / FOCAL  # 2.408 m
WALL_Y = WALL_Z * 60 / FOCAL  # 1.806 m


def _camera() -> Camera:
    intrinsics = np.array(
        [[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]], dtype=np.float32
    )
    return Camera(
        cam_to_world=np.eye(4, dtype=np.float32), intrinsics=intrinsics, width=WIDTH, height=HEIGHT
    )


def _view_data(depths: list[np.ndarray]) -> ViewData:
    return ViewData(
        images=np.full((len(depths), HEIGHT, WIDTH, 3), 128, dtype=np.uint8),
        depth=np.stack(depths).astype(np.float32),
        cameras=[_camera() for _ in depths],
    )


def _wall() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), WALL_Z, dtype=np.float32)


def test_a_wall_three_metres_away_is_fused_whole():
    # Open3D 0.20's ScalableTSDFVolume keeps nothing this far from the camera (#7592).
    geometry = fuse(_view_data([_wall(), _wall()]), OPTIONS)

    # One vertex per voxel over the 4.816 x 3.612 m wall is about 96 x 72.
    assert len(geometry.vertices) > 5000
    assert len(geometry.colors) == len(geometry.vertices)
    assert len(geometry.triangles) > 0
    assert np.all(np.abs(geometry.vertices[:, 2] - WALL_Z) < OPTIONS.voxel_size)
    extent = geometry.vertices.max(axis=0) - geometry.vertices.min(axis=0)
    assert extent[0] == pytest.approx(2 * WALL_X, abs=4 * OPTIONS.voxel_size)
    assert extent[1] == pytest.approx(2 * WALL_Y, abs=4 * OPTIONS.voxel_size)


def test_a_single_view_is_enough_for_a_surface():
    assert len(fuse(_view_data([_wall()]), OPTIONS).vertices) > 5000


def test_vertex_colours_come_from_the_images():
    colors = fuse(_view_data([_wall()]), OPTIONS).colors.astype(int)
    # Open3D leaves a few vertices on the outline of a surface black.
    assert np.mean(np.abs(colors - 128) <= 1) > 0.95


def test_pixels_without_depth_add_no_surface():
    depth = _wall()
    depth[:, : WIDTH // 2] = 0.0  # The half with x < 0 has no depth.

    vertices = fuse(_view_data([depth]), OPTIONS).vertices

    assert len(vertices) > 2000
    assert vertices[:, 0].min() > -3 * OPTIONS.voxel_size


def test_a_view_without_any_depth_is_skipped():
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    assert len(fuse(_view_data([empty, _wall(), empty]), OPTIONS).vertices) > 5000


def test_a_view_with_depth_only_between_the_sampled_pixels_is_skipped():
    # Open3D samples every 4th pixel when it activates blocks; (1, 1) is not one of them.
    sparse = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    sparse[1, 1] = WALL_Z
    assert len(fuse(_view_data([sparse, _wall()]), OPTIONS).vertices) > 5000


def test_a_capture_without_any_depth_is_refused():
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    with pytest.raises(RuntimeError, match="empty mesh"):
        fuse(_view_data([empty, empty]), OPTIONS)


def test_the_block_capacity_grows_past_its_initial_value():
    small = FusionOptions(voxel_size=OPTIONS.voxel_size, initial_block_count=4)
    grown = fuse(_view_data([_wall()]), small)
    assert len(grown.vertices) == len(fuse(_view_data([_wall()]), OPTIONS).vertices)
