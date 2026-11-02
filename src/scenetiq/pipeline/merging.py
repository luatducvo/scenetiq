"""Merging stage: MV3DIS 3D-guided mask matching turns Masks into Instances.

The vendored MV3DIS core (``scenetiq.vendor.mv3dis``) runs unchanged on arrays.
This module adapts SceneTIQ's ViewData and Masks to it and its labels back.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scenetiq.domain import FloatArray
from scenetiq.pipeline.masking import ViewMasks
from scenetiq.pipeline.view_data import ViewData


@dataclass(frozen=True)
class MergingOptions:
    """Settings of the MV3DIS Merging stage.

    Attributes:
        visibility_rtol: Relative depth tolerance for a point to count as seen.
        merge_small_points: MV3DIS merges or drops regions smaller than this.
        min_instance_points: Instances with fewer points are dropped.
    """

    visibility_rtol: float = 0.05
    merge_small_points: int = 200
    min_instance_points: int = 50


def labels_to_instance_ids(labels: npt.ArrayLike, min_points: int) -> npt.NDArray[np.int32]:
    """Turn MV3DIS point labels into Instance ids.

    Args:
        labels: ``(N,)`` MV3DIS labels; 0 means no instance.
        min_points: Regions with fewer points get no Instance.

    Returns:
        ``(N,)`` int32 ids, consecutive from 0 by descending size; -1 for none.
    """
    lab = np.asarray(labels)
    out = np.full(lab.shape, -1, dtype=np.int32)
    values, counts = np.unique(lab[lab != 0], return_counts=True)
    keep = counts >= min_points
    values, counts = values[keep], counts[keep]
    for new_id, value in enumerate(values[np.argsort(-counts, kind="stable")]):
        out[lab == value] = new_id
    return out


def merge_instances(
    points: FloatArray,
    primitive_ids: npt.ArrayLike,
    view_data: ViewData,
    view_masks: list[ViewMasks],
    options: MergingOptions | None = None,
) -> npt.NDArray[np.int32]:
    """Run MV3DIS and return the Instance id of every point (-1 for none).

    Views without any Mask are left out, as upstream does.

    Raises:
        RuntimeError: If no View has a Mask.
    """
    from scenetiq.vendor.mv3dis.in_memory import InMemoryMV3DIS

    opts = options or MergingOptions()
    used = [i for i, vm in enumerate(view_masks) if len(vm.masks) > 0]
    if not used:
        raise RuntimeError("no View produced any Mask; nothing to merge into Instances")
    agent = InMemoryMV3DIS(
        points=np.asarray(points, dtype=np.float32),
        seg_ids=np.asarray(primitive_ids),
        poses=np.stack([view_data.cameras[i].cam_to_world for i in used]).astype(np.float32),
        intrinsics=np.stack([view_data.cameras[i].intrinsics for i in used]).astype(np.float32),
        depths=view_data.depth[used],
        mask_list=[view_masks[i].masks for i in used],
        pred_score=[view_masks[i].scores for i in used],
        thres_dis=opts.visibility_rtol,
        thres_merge=opts.merge_small_points,
    )
    labels = agent.run()
    return labels_to_instance_ids(labels, opts.min_instance_points)
