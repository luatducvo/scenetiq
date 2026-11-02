import numpy as np
import pytest

from scenetiq.pipeline.adapters import (
    c2w_to_w2c,
    camera_from_da3,
    confident_depth,
    w2c_to_c2w,
)


def _rot_z_90() -> np.ndarray:
    return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)


def test_world_to_camera_3x4_becomes_camera_to_world_4x4():
    # Camera sits at world (1, 2, 3), rotated 90 deg about z.
    r_c2w = _rot_z_90()
    centre = np.array([1, 2, 3], dtype=np.float32)
    w2c = np.hstack([r_c2w.T, (-r_c2w.T @ centre)[:, None]])

    c2w = w2c_to_c2w(w2c)

    assert c2w.shape == (4, 4)
    np.testing.assert_allclose(c2w[:3, :3], r_c2w, atol=1e-6)
    np.testing.assert_allclose(c2w[:3, 3], [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(c2w[3], [0, 0, 0, 1])


def test_camera_origin_maps_to_camera_centre_in_world():
    w2c = np.array([[1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 2]], dtype=np.float32)
    c2w = w2c_to_c2w(w2c)
    np.testing.assert_allclose(c2w @ [0, 0, 0, 1], [5, 0, -2, 1], atol=1e-6)


def test_c2w_to_w2c_round_trips():
    w2c = np.array([[0, -1, 0, 0.5], [1, 0, 0, -1], [0, 0, 1, 4]], dtype=np.float32)
    back = c2w_to_w2c(w2c_to_c2w(w2c))
    np.testing.assert_allclose(back[:3], w2c, atol=1e-6)


def test_camera_from_da3_records_size_of_depth_map():
    w2c = np.eye(4, dtype=np.float32)[:3]
    k = np.array([[300, 0, 252], [0, 300, 189], [0, 0, 1]], dtype=np.float32)
    cam = camera_from_da3(w2c, k, height=378, width=504)
    assert (cam.width, cam.height) == (504, 378)
    np.testing.assert_allclose(cam.intrinsics, k)
    np.testing.assert_allclose(cam.cam_to_world, np.eye(4))


def test_confident_depth_zeroes_low_confidence_and_invalid_pixels():
    depth = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    conf = np.array([[0.1, 0.9], [0.9, 0.95]], dtype=np.float32)

    out = confident_depth(depth, conf, min_conf=0.5, max_depth=3.5)

    # 1.0 has low confidence, NaN is invalid, 4.0 is beyond max_depth.
    np.testing.assert_array_equal(out, [[0.0, 2.0], [0.0, 0.0]])
    assert out.dtype == np.float32


def test_confident_depth_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        confident_depth(np.zeros((2, 2)), np.zeros((2, 3)), min_conf=0.0, max_depth=1.0)
