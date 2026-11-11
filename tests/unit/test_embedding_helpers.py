from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scenetiq.config import EmbeddingOptions
from scenetiq.domain import Camera
from scenetiq.pipeline.embedding import (
    PlannedView,
    crop_box,
    embed_instances,
    instance_mask,
    plan_views,
    select_views,
    splat_radius,
    visible_pixels,
)
from scenetiq.pipeline.view_data import ViewData

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


def _two_views() -> ViewData:
    # Both Views look down +Z from the origin. View 0 sees a surface at 2 m everywhere;
    # View 1 only has depth at pixel (50, 40).
    depth = np.zeros((2, 80, 100), dtype=np.float32)
    depth[0] = 2.0
    depth[1, 40, 50] = 2.0
    return ViewData(
        images=np.zeros((2, 80, 100, 3), dtype=np.uint8),
        depth=depth,
        cameras=[_camera_at_origin(), _camera_at_origin()],
    )


def test_plan_views_lists_each_instances_best_views_with_their_pixels():
    points = np.array(
        [[0.0, 0.0, 2.0], [0.2, -0.1, 2.0], [0.0, 0.0, -2.0], [9.0, 9.0, 9.0]], dtype=np.float32
    )
    instance_ids = np.array([0, 0, 1, -1], dtype=np.int32)
    options = EmbeddingOptions(views_per_instance=5, min_visible_pixels=1)

    plans = plan_views(points, instance_ids, _two_views(), options)

    assert sorted(plans) == [0, 1]
    # View 0 sees both points of Instance 0, View 1 only the one at pixel (50, 40).
    assert [p.view_index for p in plans[0]] == [0, 1]
    assert plans[0][0].u.tolist() == [50, 60]
    assert plans[0][0].v.tolist() == [40, 35]
    assert plans[0][1].u.tolist() == [50]
    # Instance 1 is behind both cameras.
    assert plans[1] == []


def test_plan_views_drops_views_seeing_too_few_points():
    points = np.array([[0.0, 0.0, 2.0], [0.2, -0.1, 2.0]], dtype=np.float32)
    options = EmbeddingOptions(views_per_instance=5, min_visible_pixels=2)
    plans = plan_views(points, np.array([0, 0], dtype=np.int32), _two_views(), options)
    assert [p.view_index for p in plans[0]] == [0]


def test_a_point_is_splatted_over_half_the_spacing_of_its_neighbours():
    # Points 0.02 m apart seen with a focal length of 500 px: 5 px apart at 2 m, so a radius
    # of ceil(2.5) = 3 closes the gaps; 1 px apart at 10 m, radius ceil(0.5) = 1.
    radius = splat_radius(np.array([2.0, 10.0]), focal=500.0, spacing=0.02, max_radius=12)
    assert radius.tolist() == [3, 1]


def test_the_splat_radius_is_capped_for_points_close_to_the_camera():
    # 100 px apart at 0.1 m would need a radius of 50.
    assert splat_radius(np.array([0.1]), 500.0, 0.02, max_radius=12).tolist() == [12]


def test_an_instance_mask_is_the_union_of_its_points_squares():
    u = np.array([2, 7, 0])
    v = np.array([2, 5, 7])
    mask = instance_mask(u, v, np.array([1, 2, 1]), width=10, height=8)
    assert mask.shape == (8, 10)
    assert mask.dtype == np.bool_
    expected = np.zeros((8, 10), dtype=bool)
    expected[1:4, 1:4] = True  # (2, 2) with radius 1
    expected[3:8, 5:10] = True  # (7, 5) with radius 2, to the right and bottom borders
    expected[6:8, 0:2] = True  # (0, 7) with radius 1, clipped by the corner
    assert np.array_equal(mask, expected)


def test_plan_views_gives_each_point_a_splat_radius_when_the_point_spacing_is_known():
    points = np.array([[0.0, 0.0, 2.0], [0.2, -0.1, 2.0]], dtype=np.float32)
    ids = np.array([0, 0], dtype=np.int32)
    options = EmbeddingOptions(views_per_instance=1, min_visible_pixels=1, max_splat_radius=9)
    # Focal length 100 px, points at 2 m and 0.1 m apart: 5 px, radius ceil(2.5) = 3.
    plans = plan_views(points, ids, _two_views(), options, point_spacing=0.1)
    assert plans[0][0].radius is not None
    assert plans[0][0].radius.tolist() == [3, 3]
    # Without a spacing no mask can be drawn.
    assert plan_views(points, ids, _two_views(), options)[0][0].radius is None


GREY = (127, 127, 127)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


class _CropEncoder:
    """Encodes a crop with a grey corner (a masked one) as (0, 1), any other as (1, 0)."""

    def __init__(self) -> None:
        self.crops: list[Image.Image] = []

    def encode_images(self, images: list) -> np.ndarray:
        self.crops += images
        masked = [img.getpixel((0, 0)) == GREY for img in images]
        return np.array([[0.0, 1.0] if m else [1.0, 0.0] for m in masked], dtype=np.float32)


def _red_square_plans(tmp_path: Path) -> tuple[dict, ViewData, list[Path]]:
    # A blue photo with a red square over columns 40..60 and rows 30..50, at the depth-map
    # resolution. The Instance's points fall every 4 px on columns 42..58 and rows 32..48;
    # a radius of 2 makes their squares touch and cover exactly the red square.
    photo = Image.new("RGB", (100, 80), BLUE)
    photo.paste(RED, (40, 30, 61, 51))
    path = tmp_path / "view.png"
    photo.save(path)
    u, v = np.meshgrid(np.arange(42, 59, 4), np.arange(32, 49, 4))
    u, v = u.ravel(), v.ravel()
    plans = {3: [PlannedView(0, u, v, np.full(len(u), 2))]}
    view_data = ViewData(
        images=np.zeros((1, 80, 100, 3), dtype=np.uint8),
        depth=np.zeros((1, 80, 100), dtype=np.float32),
        cameras=[_camera_at_origin()],
    )
    return plans, view_data, [path]


def test_an_instance_is_embedded_from_context_crops_and_a_crop_with_its_background_greyed(
    tmp_path: Path,
):
    plans, view_data, paths = _red_square_plans(tmp_path)
    encoder = _CropEncoder()
    options = EmbeddingOptions(expansions=(1.0,), masked_weight=0.8)

    embeddings = embed_instances(plans, view_data, paths, encoder, options)

    # The points span columns 42..58 and rows 32..48, 17 px; growing each side by 17 px
    # gives the crop (25, 15)..(76, 66), 51 px square, with the red square at (15, 15)..(35, 35).
    context, masked = encoder.crops
    assert context.size == masked.size == (51, 51)
    assert context.getpixel((0, 0)) == BLUE
    assert context.getpixel((25, 25)) == RED
    assert masked.getpixel((0, 0)) == GREY
    assert masked.getpixel((14, 25)) == GREY  # just left of the Instance
    assert masked.getpixel((15, 25)) == RED
    assert masked.getpixel((35, 35)) == RED
    assert masked.getpixel((36, 36)) == GREY
    # 0.2 * (1, 0) + 0.8 * (0, 1) = (0.2, 0.8), normalised by sqrt(0.68) = 0.824621.
    assert embeddings[3].tolist() == pytest.approx([0.242536, 0.970143], abs=1e-6)


def test_no_masked_crop_is_made_without_a_weight_or_without_splat_radii(tmp_path: Path):
    plans, view_data, paths = _red_square_plans(tmp_path)
    encoder = _CropEncoder()
    options = EmbeddingOptions(expansions=(1.0,), masked_weight=0.0)
    assert embed_instances(plans, view_data, paths, encoder, options)[3].tolist() == [1.0, 0.0]
    assert len(encoder.crops) == 1

    unsplatted = {3: [PlannedView(0, plans[3][0].u, plans[3][0].v)]}
    encoder = _CropEncoder()
    options = EmbeddingOptions(expansions=(1.0,), masked_weight=0.8)
    assert embed_instances(unsplatted, view_data, paths, encoder, options)[3].tolist() == [1.0, 0.0]
    assert len(encoder.crops) == 1
