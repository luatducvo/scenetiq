"""Plane search on Open3D itself; skipped where the GPU stack (open3d) is not installed.

Run it on a CPU machine with: uv run --with open3d==0.20.0 pytest tests/unit/test_plane_search.py
"""

import numpy as np
import pytest

from scenetiq.config import AlignmentOptions
from scenetiq.pipeline.alignment import find_planes

pytest.importorskip("open3d")


def _floor_and_wall() -> np.ndarray:
    """A 2 x 2 m floor at y = 0 with 900 points, and a wall at x = 0 with 2500 points."""
    a = np.linspace(0.1, 2.0, 30)
    floor = np.array([[x, 0.0, z] for x in a for z in a])
    b = np.linspace(0.1, 2.0, 50)
    wall = np.array([[0.0, y, z] for y in b for z in b])
    return np.concatenate([wall, floor]).astype(np.float32)


def test_planes_are_found_largest_first_with_their_share_of_the_points():
    candidates = find_planes(_floor_and_wall(), AlignmentOptions())

    assert len(candidates) == 2
    assert abs(candidates[0].normal[0]) == pytest.approx(1.0, abs=1e-3)
    assert candidates[0].fraction == pytest.approx(2500 / 3400, abs=0.01)
    assert abs(candidates[1].normal[1]) == pytest.approx(1.0, abs=1e-3)
    assert candidates[1].fraction == pytest.approx(900 / 3400, abs=0.01)


def test_the_search_stops_at_the_plane_limit():
    candidates = find_planes(_floor_and_wall(), AlignmentOptions(max_planes=1))
    assert len(candidates) == 1


def test_the_search_leaves_the_points_alone():
    points = _floor_and_wall()
    before = points.copy()

    find_planes(points, AlignmentOptions(plane_max_points=1000))

    np.testing.assert_array_equal(points, before)


def test_too_few_points_yield_no_planes():
    assert find_planes(np.zeros((2, 3), dtype=np.float32), AlignmentOptions()) == []
