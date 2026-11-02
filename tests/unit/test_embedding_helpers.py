import numpy as np
import pytest

from scenetiq.domain import Camera
from scenetiq.pipeline.embedding import crop_box, select_views, visible_pixels
from scenetiq.pipeline.siglip import sigmoid_score

K = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=np.float32)


def _camera_at_origin() -> Camera:
    return Camera(cam_to_world=np.eye(4, dtype=np.float32), intrinsics=K, width=100, height=80)


def test_point_in_front_of_camera_projects_to_expected_pixel():
    points = np.array([[0.2, -0.1, 2.0]], dtype=np.float32)
    depth = np.full((80, 100), 2.0, dtype=np.float32)
    u, v = visible_pixels(points, _camera_at_origin(), depth, rtol=0.05)
    # u = 100 * 0.2 / 2 + 50 = 60, v = 100 * -0.1 / 2 + 40 = 35
    assert u.tolist() == [60]
    assert v.tolist() == [35]


def test_occluded_behind_and_out_of_frame_points_are_not_visible():
    points = np.array(
        [
            [0.0, 0.0, 3.0],  # behind the surface seen at 2 m: occluded
            [0.0, 0.0, -2.0],  # behind the camera
            [5.0, 0.0, 2.0],  # outside the image
            [0.0, 0.0, 2.05],  # within 5 % of the depth: visible
        ],
        dtype=np.float32,
    )
    depth = np.full((80, 100), 2.0, dtype=np.float32)
    u, v = visible_pixels(points, _camera_at_origin(), depth, rtol=0.05)
    assert list(zip(u.tolist(), v.tolist(), strict=True)) == [(50, 40)]


def test_pixels_without_depth_do_not_count_as_visible():
    points = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    u, _ = visible_pixels(points, _camera_at_origin(), np.zeros((80, 100), np.float32), 0.05)
    assert len(u) == 0


def test_select_views_keeps_the_views_seeing_most_points():
    counts = [3, 50, 0, 120, 49]
    assert select_views(counts, top_k=2, min_pixels=10) == [3, 1]
    assert select_views(counts, top_k=10, min_pixels=50) == [3, 1]
    assert select_views([1, 2], top_k=3, min_pixels=10) == []


def test_crop_box_expands_around_pixels_and_stays_inside_the_image():
    # Pixels 10..30 span 21 px; each side grows by 0.5 * 21 = 10.5 px:
    # left floor(10 - 10.5) = -1 -> 0, right ceil(31 + 10.5) = 42;
    # top floor(20 - 10.5) = 9, bottom ceil(41 + 10.5) = 52.
    u = np.array([10, 30])
    v = np.array([20, 40])
    assert crop_box(u, v, expand=0.5, width=100, height=80) == (0, 9, 42, 52)
    # Expanding a box touching the border is clipped to the image.
    assert crop_box(np.array([90, 99]), np.array([70, 79]), 1.0, 100, 80) == (80, 60, 100, 80)


def test_crop_box_has_a_minimum_size():
    left, top, right, bottom = crop_box(np.array([50]), np.array([40]), 0.1, 100, 80, min_size=8)
    assert right - left >= 8
    assert bottom - top >= 8


def test_sigmoid_score_maps_cosine_to_probability():
    scores = sigmoid_score(np.array([0.0, 0.1]), logit_scale=10.0, logit_bias=-1.0)
    assert scores[0] == pytest.approx(1 / (1 + np.exp(1.0)))
    assert scores[1] == pytest.approx(0.5)
