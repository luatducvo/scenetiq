"""Transfer predicted Instances onto ground-truth points."""

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


def transfer_instances(
    pred_points: npt.ArrayLike,
    pred_ids: npt.ArrayLike,
    gt_points: npt.ArrayLike,
    max_distance: float,
) -> npt.NDArray[np.int32]:
    """Give every ground-truth point the Instance id of its nearest predicted point.

    Points farther than ``max_distance`` metres from any prediction get -1.
    """
    ids = np.asarray(pred_ids, dtype=np.int32)
    dist, idx = KDTree(np.asarray(pred_points)).query(np.asarray(gt_points), k=1)
    return np.where(dist <= max_distance, ids[idx], -1).astype(np.int32)
