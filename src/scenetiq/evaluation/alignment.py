"""Similarity alignment between a Reconstruction and the ScanNet world frame."""

import numpy as np
import numpy.typing as npt


def umeyama(
    src: npt.ArrayLike, dst: npt.ArrayLike
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Least-squares similarity transform with ``dst ~= scale * R @ src + t`` (Umeyama 1991).

    Args:
        src: ``(N, 3)`` points, e.g. DA3 camera centres.
        dst: ``(N, 3)`` corresponding points, e.g. ScanNet camera centres.

    Returns:
        ``(scale, R, t)``.

    Raises:
        ValueError: With fewer than 3 correspondences or degenerate ``src``.
    """
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    if a.shape != b.shape or len(a) < 3:
        raise ValueError("need >= 3 matching correspondences")
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - mu_a, b - mu_b
    var_a = (ac**2).sum() / len(a)
    if var_a == 0:
        raise ValueError("source points are degenerate")
    cov = bc.T @ ac / len(a)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    scale = float(np.trace(np.diag(d) @ s) / var_a)
    t = mu_b - scale * rot @ mu_a
    return scale, rot, t


def apply_similarity(
    points: npt.ArrayLike, scale: float, rot: npt.ArrayLike, t: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Map ``points`` with ``scale * R @ p + t``."""
    p = np.asarray(points, dtype=np.float64)
    return scale * p @ np.asarray(rot).T + np.asarray(t)
