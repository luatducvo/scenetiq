"""In-memory MV3DIS adapter (replaces upstream ``ScanNetMV3DIS``).

Upstream ``mv3dis_main.ScanNetMV3DIS`` reads ScanNet files (``color/*.jpg``,
millimetre depth PNGs, ``pose/*.txt``, ``*.segs.json``, Grounded-SAM ``.pth``).
This class takes the same data as arrays instead. ``get_seg_data`` is copied
from upstream with the file-loading branches and the broken ``point_level``
branch removed; the algorithm is unchanged. See PROVENANCE.md.
"""

from types import SimpleNamespace

import numpy as np
import scipy.spatial

from . import mv3dis_utils as utils
from .mv3dis_base import MV3DISBase

DEFAULT_ARGS = dict(
    max_neighbor_distance=2,
    similar_metric="2-norm",
    view_freq=1,
    dis_decay=0.5,
    scans_dir="",
    thres_trunc=0.0,
    thres_merge=200,
    thres_dis=0.05,
)
"""Upstream CLI defaults (``mv3dis_main.build_argument_parser``), except ``view_freq``:
SceneTIQ hands over exactly the Views to use, so none are skipped here."""


class InMemoryMV3DIS(MV3DISBase):
    """MV3DIS on arrays.

    Args:
        points: (N, 3) float32 points to label.
        seg_ids: (N,) int primitive (superpoint) id of every point.
        poses: (M, 4, 4) camera-to-world.
        intrinsics: (M, 3, 3) intrinsics shared by colour, depth and masks.
        depths: (M, H, W) float32 metres, 0 = unknown.
        mask_list: M arrays (K_i, H, W) of binary masks (K_i >= 1, <= 255).
        pred_score: M arrays (K_i,) of mask confidences.
        **overrides: Overrides of :data:`DEFAULT_ARGS`.
    """

    def __init__(self, points, seg_ids, poses, intrinsics, depths, mask_list, pred_score, **overrides):
        args = SimpleNamespace(**{**DEFAULT_ARGS, **overrides})
        super().__init__(points, args)
        self.base_seg_ids = np.asarray(seg_ids)
        self.poses = poses
        self.color_intrinsics = intrinsics
        self.depth_intrinsics = intrinsics
        self.depths = depths
        self.mask_list = mask_list
        self.pred_score = pred_score
        self.masks_2D = None
        self.M = poses.shape[0]
        self.CH, self.CW = mask_list[0].shape[-2:]
        self.DH, self.DW = depths.shape[-2:]
        self.img_dim = [self.CW, self.CH]
        self.scene_id = "in-memory"

    def run(self):
        """Return (N,) labels: 0 = no instance, other values = instance region ids."""
        return self.assign_label(
            self.points,
            thres_connect=None,
            vis_dis=self.args.thres_dis,
            max_neighbor_distance=self.args.max_neighbor_distance,
            similar_metric=self.args.similar_metric,
        )

    # Copied from upstream ScanNetMV3DIS.get_seg_data (file branches removed).
    def get_seg_data(
        self,
        scans_dir,
        scene_id,
        max_neighbor_distance,
        seg_ids=None,
        points_obj_labels_path=None,
        k_graph=8,
        point_level=False,
        points_clipfeature=None,
        ssp_split_sum=None,
        members_only=False,
    ):
        if seg_ids is None:
            seg_ids = self.base_seg_ids

        seg_ids, mapping, unique_values = utils.num_to_natural(seg_ids)
        unique_seg_ids, counts = np.unique(seg_ids, return_counts=True)
        seg_num = unique_seg_ids.shape[0]

        if ssp_split_sum is not None:
            new_seg_split_sum = np.zeros(len(unique_values))

            valid_old_ids = np.arange(len(ssp_split_sum))
            valid_new_ids = mapping[valid_old_ids + 1]
            mask = valid_new_ids != -1
            np.add.at(new_seg_split_sum, valid_new_ids[mask], ssp_split_sum[mask])

        seg_members = {}
        for id in unique_seg_ids:
            seg_members[id] = np.where(seg_ids == id)[0]

        if ssp_split_sum is not None:
            for seg_id, point_indices in seg_members.items():
                seg_members[seg_id] = {
                    "point_indices": point_indices,
                    "seg_num": new_seg_split_sum[seg_id],
                }
        else:
            for seg_id, point_indices in seg_members.items():
                seg_members[seg_id] = {"point_indices": point_indices}
        if members_only:
            return seg_ids, seg_num, seg_members, None, counts, None

        knn_cache = getattr(self, "_point_knn_cache", None)
        if knn_cache is None or knn_cache[0] != k_graph:
            points_kdtree = scipy.spatial.KDTree(self.points)
            points_neighbors = points_kdtree.query(self.points, k_graph, workers=-1)[1]
            self._point_knn_cache = (k_graph, points_neighbors)
        else:
            points_neighbors = knn_cache[1]

        seg_direct_neighbors = np.zeros((seg_num, seg_num), dtype=bool)
        for id, members in seg_members.items():
            neighbors = points_neighbors[members["point_indices"]]
            neighbor_seg_ids = seg_ids[neighbors]
            seg_direct_neighbors[id][neighbor_seg_ids] = 1
        seg_direct_neighbors[np.eye(seg_num, dtype=bool)] = 0  # exclude self
        # make neighboring matrix symmetric
        seg_direct_neighbors[seg_direct_neighbors.T] = 1

        centroids = np.zeros((seg_num, self.points.shape[1]))
        for i, members_i in seg_members.items():
            points_i = self.points[members_i["point_indices"]]
            centroids[i] = np.mean(points_i, axis=0)

        distance_matrix = scipy.spatial.distance.cdist(centroids, centroids)

        seg_neineighbors = np.zeros(
            (max_neighbor_distance, seg_num, seg_num), dtype=bool
        )
        seg_neineighbors[0] = seg_direct_neighbors
        for i in range(1, max_neighbor_distance):
            for seg_id in range(seg_num):
                last_layer_neighbors = seg_neineighbors[i - 1, seg_id]
                this_layer_neighbors = (
                    seg_neineighbors[i - 1, last_layer_neighbors].sum(0) > 0
                )
                seg_neineighbors[i, seg_id] = this_layer_neighbors
            # exclude self
            seg_neineighbors[i, np.eye(seg_num, dtype=bool)] = 0
            # include closer neighbors
            seg_neineighbors[i, seg_neineighbors[i - 1]] = 1

        return seg_ids, seg_num, seg_members, seg_neineighbors, counts, distance_matrix
