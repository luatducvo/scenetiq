"""Pure conversions between model output conventions and SceneTIQ's domain types."""

import numpy as np
import numpy.typing as npt

from scenetiq.domain import Camera, FloatArray


def w2c_to_c2w(w2c: npt.ArrayLike) -> FloatArray:
    """Invert a world-to-camera 3x4 (or 4x4) rigid transform into camera-to-world 4x4."""
    m = np.asarray(w2c, dtype=np.float64)
    rot, trans = m[:3, :3], m[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out.astype(np.float32)


def c2w_to_w2c(c2w: npt.ArrayLike) -> FloatArray:
    """Invert a camera-to-world 4x4 rigid transform into world-to-camera 4x4."""
    return w2c_to_c2w(c2w)


def camera_from_da3(
    w2c: npt.ArrayLike, intrinsics: npt.ArrayLike, height: int, width: int
) -> Camera:
    """Build a Camera from DA3's world-to-camera extrinsics and intrinsics.

    Args:
        w2c: DA3 extrinsics for one View, 3x4 world-to-camera (OpenCV convention).
        intrinsics: DA3 3x3 intrinsics, valid for the depth map's resolution.
        height: Depth map height in pixels.
        width: Depth map width in pixels.
    """
    return Camera(
        cam_to_world=w2c_to_c2w(w2c),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        width=width,
        height=height,
    )


def confident_depth(
    depth: npt.ArrayLike, conf: npt.ArrayLike, min_conf: float, max_depth: float
) -> FloatArray:
    """Return depth in float32 metres with unreliable pixels set to 0 (= no depth).

    A pixel is dropped when its confidence is below ``min_conf``, its depth is
    not finite or not positive, or its depth exceeds ``max_depth``.

    Raises:
        ValueError: If ``depth`` and ``conf`` differ in shape.
    """
    d = np.asarray(depth, dtype=np.float32)
    c = np.asarray(conf, dtype=np.float32)
    if d.shape != c.shape:
        raise ValueError(f"depth shape {d.shape} != confidence shape {c.shape}")
    keep = np.isfinite(d) & (d > 0) & (d <= max_depth) & (c >= min_conf)
    return np.where(keep, d, 0.0).astype(np.float32)
