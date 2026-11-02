"""Primitives: groups of nearby points that MV3DIS merges into Instances."""

import numpy as np
import numpy.typing as npt

from scenetiq.domain import FloatArray


def voxel_primitives(points: FloatArray, voxel_size: float) -> npt.NDArray[np.int64]:
    """Group points by the voxel of a regular grid they fall in.

    The fallback primitive when no mesh superpoints are available: MV3DIS needs
    a few thousand primitives, not one per point (see the vendored PROVENANCE).

    Args:
        points: ``(N, 3)`` positions in metres.
        voxel_size: Grid cell edge in metres.

    Returns:
        ``(N,)`` primitive ids, consecutive from 0.
    """
    cells = np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_inverse=True)
    return ids.reshape(-1).astype(np.int64)
