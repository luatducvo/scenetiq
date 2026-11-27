import numpy as np
import pytest

from scenetiq.domain import Camera
from scenetiq.pipeline.masking import ViewMasks
from scenetiq.pipeline.merging import merge_instances
from scenetiq.pipeline.view_data import ViewData

HEIGHT, WIDTH = 4, 4


def _inputs() -> tuple[ViewData, list[ViewMasks]]:
    camera = Camera(
        cam_to_world=np.eye(4, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        width=WIDTH,
        height=HEIGHT,
    )
    view_data = ViewData(
        images=np.zeros((1, HEIGHT, WIDTH, 3), dtype=np.uint8),
        depth=np.ones((1, HEIGHT, WIDTH), dtype=np.float32),
        cameras=[camera],
    )
    masks = ViewMasks(
        masks=np.ones((1, HEIGHT, WIDTH), dtype=np.uint8),
        scores=np.ones(1, dtype=np.float32),
    )
    return view_data, [masks]


def _points(count: int) -> np.ndarray:
    return np.arange(count * 3, dtype=np.float32).reshape(count, 3)


def test_a_scene_too_small_for_the_neighbour_graph_is_refused_by_name():
    # The traceback this replaces: IndexError: index 3 is out of bounds for axis 0 with size 3.
    view_data, view_masks = _inputs()
    with pytest.raises(RuntimeError, match=r"only 3 points"):
        merge_instances(_points(3), np.array([0, 1, 2]), view_data, view_masks)


def test_a_single_primitive_is_refused_by_name():
    # MV3DIS compares neighbouring primitives, so one alone ends in a bare AssertionError.
    view_data, view_masks = _inputs()
    with pytest.raises(RuntimeError, match=r"only 1 primitive"):
        merge_instances(_points(8), np.zeros(8, dtype=np.int64), view_data, view_masks)
