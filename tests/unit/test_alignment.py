import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scenetiq
from scenetiq.config import AlignmentOptions
from scenetiq.domain import Camera
from scenetiq.pipeline.alignment import (
    ALIGNMENT_LOGGER,
    apply_rotation,
    evenly_spaced_views,
    find_up_consensus,
    gravity_align,
    rotation_between,
)
from scenetiq.pipeline.embedding import visible_pixels
from scenetiq.pipeline.view_data import FusedGeometry, ViewData

QUARTER_TURN_ABOUT_X = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _view_data(cam_to_world: np.ndarray) -> ViewData:
    camera = Camera(
        cam_to_world=cam_to_world.astype(np.float32),
        intrinsics=np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32),
        width=100,
        height=100,
    )
    return ViewData(
        images=np.zeros((1, 100, 100, 3), dtype=np.uint8),
        depth=np.full((1, 100, 100), 2.0, dtype=np.float32),
        cameras=[camera],
    )


def _geometry(vertices: list[list[float]]) -> FusedGeometry:
    return FusedGeometry(
        vertices=np.array(vertices, dtype=np.float32),
        colors=np.arange(3 * len(vertices), dtype=np.uint8).reshape(-1, 3),
        triangles=np.array([[0, 1, 0]], dtype=np.int32),
    )


def test_applying_a_rotation_moves_points_and_keeps_their_order():
    geometry = _geometry([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    aligned, _ = apply_rotation(geometry, _view_data(np.eye(4)), QUARTER_TURN_ABOUT_X)

    np.testing.assert_allclose(aligned.vertices, [[1.0, -3.0, 2.0], [4.0, -6.0, 5.0]], atol=1e-6)
    assert aligned.vertices.dtype == np.float32
    np.testing.assert_array_equal(aligned.colors, geometry.colors)
    np.testing.assert_array_equal(aligned.triangles, geometry.triangles)


def test_a_point_lands_on_the_same_pixel_before_and_after_rotation():
    # With f = 100 and centre (50, 50), the point (0.5, 0, 2) projects to pixel (75, 50).
    geometry = _geometry([[0.5, 0.0, 2.0]])
    view_data = _view_data(np.eye(4))

    aligned, moved = apply_rotation(geometry, view_data, QUARTER_TURN_ABOUT_X)

    u, v = visible_pixels(aligned.vertices, moved.cameras[0], moved.depth[0], rtol=0.05)
    assert (u.tolist(), v.tolist()) == ([75], [50])


def test_applying_a_rotation_leaves_depth_and_images_alone():
    view_data = _view_data(np.eye(4))

    _, moved = apply_rotation(_geometry([[0.0, 0.0, 2.0]]), view_data, QUARTER_TURN_ABOUT_X)

    np.testing.assert_array_equal(moved.depth, view_data.depth)
    np.testing.assert_array_equal(moved.images, view_data.images)


def _up_at_degrees(degrees: float) -> list[float]:
    """A unit vector tilted from -Y towards +X."""
    rad = np.deg2rad(degrees)
    return [float(np.sin(rad)), float(-np.cos(rad)), 0.0]


def test_a_single_estimate_is_its_own_consensus():
    consensus = find_up_consensus(np.array([[0.0, -1.0, 0.0]]), 10.0, 0.5)
    assert consensus.fraction == 1.0
    assert consensus.direction is not None
    np.testing.assert_allclose(consensus.direction, [0.0, -1.0, 0.0])


def test_two_estimates_far_apart_have_no_consensus():
    consensus = find_up_consensus(np.array([_up_at_degrees(0), _up_at_degrees(40)]), 10.0, 0.5)
    assert consensus.direction is None
    assert consensus.fraction == 0.5


def test_two_of_three_close_estimates_reach_consensus_on_their_mean():
    ups = np.array([_up_at_degrees(0), _up_at_degrees(6), _up_at_degrees(90)])
    consensus = find_up_consensus(ups, 10.0, 0.5)
    assert consensus.fraction == pytest.approx(2 / 3)
    assert consensus.direction is not None
    # The mean of 0 and 6 degrees is 3 degrees: (sin 3, -cos 3, 0).
    np.testing.assert_allclose(consensus.direction, [0.052336, -0.998630, 0.0], atol=1e-5)


def test_exactly_half_is_not_a_consensus():
    ups = np.array([_up_at_degrees(0), _up_at_degrees(5), _up_at_degrees(60), _up_at_degrees(65)])
    assert find_up_consensus(ups, 10.0, 0.5).direction is None


def test_a_tie_between_groups_goes_to_the_earlier_view():
    # Three groups of one: the first View's estimate decides the reported group.
    ups = np.array([_up_at_degrees(0), _up_at_degrees(60), _up_at_degrees(120)])
    consensus = find_up_consensus(ups, 10.0, 0.3)
    assert consensus.direction is not None
    np.testing.assert_allclose(consensus.direction, _up_at_degrees(0), atol=1e-9)


def test_no_estimates_have_no_consensus():
    consensus = find_up_consensus(np.zeros((0, 3)), 10.0, 0.5)
    assert consensus.direction is None
    assert consensus.fraction == 0.0


class _FixedUps:
    """An up estimator that answers with given vectors and records what it was asked."""

    def __init__(self, ups: list[list[float]]) -> None:
        self.ups = np.array(ups, dtype=np.float32)
        self.images: np.ndarray | None = None
        self.focals: np.ndarray | None = None

    def estimate(self, images: np.ndarray, focals: np.ndarray | None) -> np.ndarray:
        self.images = images
        self.focals = focals
        return self.ups


def _views(count: int) -> ViewData:
    """``count`` level Views; each View's pixels hold its index, so a test can tell them apart."""
    base = _view_data(np.eye(4))
    return ViewData(
        images=np.stack([np.full((100, 100, 3), i, dtype=np.uint8) for i in range(count)]),
        depth=np.full((count, 100, 100), 2.0, dtype=np.float32),
        cameras=base.cameras * count,
    )


def test_views_are_spaced_evenly_through_the_capture():
    assert evenly_spaced_views(5, 3) == [0, 2, 4]
    assert evenly_spaced_views(2, 12) == [0, 1]
    assert evenly_spaced_views(10, 1) == [0]


def test_a_capture_with_consensus_is_rotated_so_up_becomes_minus_y(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger=ALIGNMENT_LOGGER)

    aligned, moved = gravity_align(
        _geometry([[1.0, 2.0, 3.0]]), _views(1), _FixedUps([[0.0, 0.0, 1.0]]), AlignmentOptions()
    )

    # Up (0, 0, 1) to (0, -1, 0) is a quarter turn about +X: (x, y, z) to (x, -z, y).
    np.testing.assert_allclose(aligned.vertices, [[1.0, -3.0, 2.0]], atol=1e-6)
    np.testing.assert_allclose(moved.cameras[0].cam_to_world[:3, :3], QUARTER_TURN_ABOUT_X)
    assert caplog.messages == [
        "gravity-alignment method=geocalib angle=90.0deg views=1 consensus=1.00"
    ]


def test_up_estimates_are_read_in_the_frame_of_their_camera():
    # The camera is turned so its (0, -1, 0) is world (0, 0, -1); up must become -Y anyway.
    # That is a quarter turn the other way: (x, y, z) to (x, z, -y).
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = QUARTER_TURN_ABOUT_X
    view_data = _view_data(cam_to_world)

    aligned, moved = gravity_align(
        _geometry([[1.0, 2.0, 3.0]]), view_data, _FixedUps([[0.0, -1.0, 0.0]]), AlignmentOptions()
    )

    np.testing.assert_allclose(aligned.vertices, [[1.0, 3.0, -2.0]], atol=1e-6)
    np.testing.assert_allclose(moved.cameras[0].cam_to_world[:3, :3], np.eye(3), atol=1e-6)


def test_without_consensus_nothing_moves_and_the_log_says_so(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=ALIGNMENT_LOGGER)
    geometry = _geometry([[1.0, 2.0, 3.0]])
    view_data = _views(2)
    estimator = _FixedUps([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])

    aligned, moved = gravity_align(geometry, view_data, estimator, AlignmentOptions())

    np.testing.assert_array_equal(aligned.vertices, geometry.vertices)
    np.testing.assert_array_equal(moved.cameras[0].cam_to_world, view_data.cameras[0].cam_to_world)
    assert caplog.messages == ["gravity-alignment method=none angle=0.0deg views=2 consensus=0.50"]


def test_only_the_selected_views_are_estimated():
    estimator = _FixedUps([[0.0, -1.0, 0.0]] * 3)

    gravity_align(_geometry([[0.0, 0.0, 1.0]]), _views(5), estimator, AlignmentOptions(max_views=3))

    assert estimator.images is not None
    assert estimator.images[:, 0, 0, 0].tolist() == [0, 2, 4]


def test_the_focal_prior_is_the_mean_of_fx_and_fy():
    camera = replace(
        _view_data(np.eye(4)).cameras[0],
        intrinsics=np.array([[100, 0, 50], [0, 120, 50], [0, 0, 1]], dtype=np.float32),
    )
    view_data = replace(_views(1), cameras=[camera])
    estimator = _FixedUps([[0.0, -1.0, 0.0]])

    gravity_align(_geometry([[0.0, 0.0, 1.0]]), view_data, estimator, AlignmentOptions())

    assert estimator.focals is not None
    assert estimator.focals.tolist() == [110.0]


def test_the_focal_prior_can_be_turned_off():
    estimator = _FixedUps([[0.0, -1.0, 0.0]])

    gravity_align(
        _geometry([[0.0, 0.0, 1.0]]),
        _views(1),
        estimator,
        AlignmentOptions(use_focal_prior=False),
    )

    assert estimator.focals is None


def test_a_failing_estimator_is_not_swallowed():
    class Broken:
        def estimate(self, images: np.ndarray, focals: np.ndarray | None) -> np.ndarray:
            raise RuntimeError("weights could not be downloaded")

    with pytest.raises(RuntimeError, match="weights could not be downloaded"):
        gravity_align(_geometry([[0.0, 0.0, 1.0]]), _views(1), Broken(), AlignmentOptions())


def test_colab_logging_lets_the_alignment_line_reach_stdout():
    path = Path(scenetiq.__file__).with_name("colab_logging.json")
    logger = json.loads(path.read_text(encoding="utf-8"))["loggers"][ALIGNMENT_LOGGER]
    assert logger["level"] == "INFO"
    assert logger["handlers"] == ["progress"]
    assert logger["propagate"] is False


def test_rotation_takes_one_direction_to_the_other():
    rotation = rotation_between(np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))
    # A quarter turn about +X sends (x, y, z) to (x, -z, y).
    np.testing.assert_allclose(rotation @ np.array([1.0, 2.0, 3.0]), [1.0, -3.0, 2.0], atol=1e-9)


def test_rotation_between_opposite_directions_is_still_a_rotation():
    rotation = rotation_between(np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.linalg.norm(rotation @ np.array([3.0, 0.0, 4.0])) == pytest.approx(5.0)


def test_rotation_between_equal_directions_changes_nothing():
    rotation = rotation_between(np.array([0.0, -1.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)
