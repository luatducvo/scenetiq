import numpy as np
import numpy.typing as npt

from scenetiq.domain import Camera, FloatArray


def w2c_to_c2w(w2c: npt.ArrayLike) -> FloatArray:
    m = np.asarray(w2c, dtype=np.float64)
    rot, trans = m[:3, :3], m[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out.astype(np.float32)


def c2w_to_w2c(c2w: npt.ArrayLike) -> FloatArray:
    return w2c_to_c2w(c2w)


def camera_from_da3(
    w2c: npt.ArrayLike, intrinsics: npt.ArrayLike, height: int, width: int
) -> Camera:
    return Camera(
        cam_to_world=w2c_to_c2w(w2c),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        width=width,
        height=height,
    )


def confident_depth(
    depth: npt.ArrayLike, conf: npt.ArrayLike, min_conf: float, max_depth: float
) -> FloatArray:
    d = np.asarray(depth, dtype=np.float32)
    c = np.asarray(conf, dtype=np.float32)
    if d.shape != c.shape:
        raise ValueError(f"depth shape {d.shape} != confidence shape {c.shape}")
    keep = np.isfinite(d) & (d > 0) & (d <= max_depth) & (c >= min_conf)
    return np.where(keep, d, 0.0).astype(np.float32)
