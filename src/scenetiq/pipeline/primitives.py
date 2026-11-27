import numpy as np
import numpy.typing as npt

from scenetiq.domain import FloatArray


def voxel_primitives(points: FloatArray, voxel_size: float) -> npt.NDArray[np.int64]:
    cells = np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_inverse=True)
    return ids.reshape(-1).astype(np.int64)
