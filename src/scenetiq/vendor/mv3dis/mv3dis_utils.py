"""Subset of MV3DIS ``helpers/mv3dis_utils.py`` used by the inference core.

Vendored verbatim from https://github.com/zybjn/MV3DIS (MIT); see PROVENANCE.md.
Only the functions reached from ``MV3DISBase.assign_label`` are kept; the
header imports were reduced to what they need.
"""

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def num_to_natural(group_ids, void_number=-1):
    """
    code credit: SAM3D
    """
    if void_number == -1:
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids == -1):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)

        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]

    elif void_number == 0:
        if np.all(group_ids == 0):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != 0])
        mapping = np.full(np.max(unique_values) + 2, 0)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")
    print("array.min()", array.min())
    return array, mapping, unique_values


def _seglabels_from_bincount(plabels, gpu_seg_ids_long, seg_num, label_range):
    """Count (segment, point-label) pairs through a dense linear index."""
    labels_long = plabels.to(torch.long)
    linear_ids = gpu_seg_ids_long * label_range + labels_long
    counts = torch.bincount(linear_ids, minlength=seg_num * label_range)
    return counts.reshape(seg_num, label_range).to(torch.float32)


def _label_ranges_from_points_label(points_label):
    """Return the exact per-frame max label plus one on the CPU."""
    return np.max(np.asarray(points_label), axis=0).astype(np.int64) + 1


@torch.inference_mode()
def old_torch_calcu_all_similar(
    labels1, labels2, similar_metric="2-norm", thres_trunc=0
):
    """Production pairwise similarity implementation using broadcasting."""
    if similar_metric == "2-norm":
        dis = torch.norm(labels1[:, None, :] - labels2[None, :, :], dim=-1)

        def similar_func(x):
            return (2 - x**2) / 2.0
    else:
        raise NotImplementedError

    similar = similar_func(dis)
    if thres_trunc > 0:
        similar[similar < thres_trunc] = 0

    return similar


torch_calcu_all_similar = old_torch_calcu_all_similar


@torch.inference_mode()
def _torch_get_similar_confidence_matrix_impl(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    thres_trunc,
    weight_points,
    seglabel_builder,
    upload_points_label,
):
    """Optimized equivalent of ``old_torch_get_similar_confidence_matrix``.

    The pair semantics are unchanged: each 200-row batch is compared against
    the ordered union of graph-neighbor columns for that batch.
    """
    view_num = seg_seen.shape[1]
    seg_num = seg_seen.shape[0]

    print("preparing data on gpu")
    gpu_seg_ids = torch.tensor(seg_ids, device=device, dtype=torch.int32)
    gpu_seg_ids_long = (
        gpu_seg_ids.to(torch.long)
        if seglabel_builder is _seglabels_from_bincount
        else None
    )
    if upload_points_label:
        label_ranges = _label_ranges_from_points_label(points_label)
        gpu_points_label = torch.as_tensor(
            points_label, device=device, dtype=torch.float32
        )
    else:
        label_ranges = None
        gpu_points_label = torch.tensor(points_label, device="cpu", dtype=torch.float32)
    gpu_seg_neighbors = torch.tensor(seg_neighbors, device=device, dtype=torch.bool)
    gpu_seg_seen = torch.tensor(seg_seen, device=device, dtype=torch.float32)

    similar_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    confidence_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    one_view_similar = torch.zeros(
        [seg_num, seg_num], device=device, dtype=torch.float32
    )

    batch_size = 200
    batch_neighbors = []
    for batch_start in range(0, seg_num, batch_size):
        batch_end = min(batch_start + batch_size, seg_num)
        if seg_neighbors[batch_start:batch_end].nonzero()[0].size == 0:
            continue
        all_neighbors_mask = (
            torch.sum(gpu_seg_neighbors[batch_start:batch_end], dim=0) > 0
        )
        neighbor_indices = torch.nonzero(all_neighbors_mask, as_tuple=False).flatten()
        batch_neighbors.append(
            (batch_start, batch_end, all_neighbors_mask, neighbor_indices)
        )

    for view_index in tqdm(range(view_num)):
        if upload_points_label:
            plabels = gpu_points_label[:, view_index]
            label_range = int(label_ranges[view_index])
        else:
            plabels = gpu_points_label[:, view_index].to(device)
            label_range = int(torch.max(plabels) + 1)
        one_view_similar.fill_(0.0)

        if label_range < 2:
            continue

        if seglabel_builder is _seglabels_from_bincount:
            seglabels = seglabel_builder(
                plabels, gpu_seg_ids_long, seg_num, label_range
            )
        else:
            seglabels = seglabel_builder(plabels, gpu_seg_ids, seg_num, label_range)

        nonzero_seglabels = seglabels[:, 1:]
        if similar_metric == "2-norm":
            nonzero_seglabels = torch.divide(
                nonzero_seglabels,
                torch.clamp(torch.norm(nonzero_seglabels, dim=-1), 1e-8)[:, None],
            )
        else:
            raise NotImplementedError
        del plabels

        for (
            batch_start,
            batch_end,
            all_neighbors_mask,
            neighbor_indices,
        ) in batch_neighbors:
            neighbors_labels = nonzero_seglabels[neighbor_indices]
            one_view_similar[batch_start:batch_end, all_neighbors_mask] = (
                torch_calcu_all_similar(
                    nonzero_seglabels[batch_start:batch_end],
                    neighbors_labels,
                    similar_metric=similar_metric,
                    thres_trunc=thres_trunc,
                )
            )

        view_visibility = gpu_seg_seen[:, view_index]
        one_view_confidence = view_visibility[:, None] @ view_visibility[None, :]

        del seglabels, nonzero_seglabels

        confidence_sum += one_view_confidence
        similar_sum += one_view_similar * one_view_confidence

    return [similar_sum.cpu().numpy(), confidence_sum.cpu().numpy()]


def torch_get_similar_confidence_matrix(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    thres_trunc=0,
    weight_points=None,
):
    """Production implementation using exact linear-index pair counts.

    Point labels are uploaded once, while the historical 200-row batch and
    neighbor-union pair semantics remain unchanged.
    """
    return _torch_get_similar_confidence_matrix_impl(
        seg_neighbors,
        seg_ids,
        seg_seen,
        points_label,
        similar_metric,
        thres_trunc,
        weight_points,
        _seglabels_from_bincount,
        True,
    )


@torch.inference_mode()
def torch_world2cam_pixel(
    points_world_all: np.array,
    color_intrinsic: np.array,
    depth_intrinsic: np.array,
    pose: np.array,
):
    """project N (1e6) points to M (1e3) images

    :param points_world: (N, 3)
    :param color_intrinsic, depth_intrinsic: (M, 3, 3) the intrinsics of color and depth camera
    :param pose: (M, 4, 4)
    :return: points_cam(n,M,3), color_points_pixel(n,M,2), depth_points_pixel(n,M,2)
    """
    batch_size = 10000

    color_intrinsic = torch.tensor(color_intrinsic, device=device, dtype=torch.float32)
    depth_intrinsic = torch.tensor(depth_intrinsic, device=device, dtype=torch.float32)
    pose = torch.tensor(pose, device=device, dtype=torch.float32)
    pose_inv = torch.linalg.inv(pose)
    del pose

    N = points_world_all.shape[0]
    M = pose_inv.shape[0]

    final_points_cam = np.zeros((N, M, 3), dtype=np.float32)
    final_color_points_pixel = np.zeros((N, M, 2), dtype=np.int32)
    final_depth_points_pixel = np.zeros((N, M, 2), dtype=np.int32)

    for batch_start in tqdm(range(0, points_world_all.shape[0], batch_size)):
        points_world = torch.tensor(
            points_world_all[batch_start : batch_start + batch_size],
            device=device,
            dtype=torch.float32,
        )
        points_world_homo = torch.cat(
            (
                points_world,
                torch.ones(
                    (points_world.shape[0], 1), dtype=torch.float32, device=device
                ),
            ),
            1,
        )

        points_cam_homo = torch.matmul(
            pose_inv[None], points_world_homo[:, None, :, None]
        )
        points_cam_homo = points_cam_homo[..., 0]

        points_cam = torch.div(points_cam_homo[..., :-1], points_cam_homo[..., [-1]])

        color_points_pixel_homo = torch.matmul(color_intrinsic, points_cam[..., None])
        depth_points_pixel_homo = torch.matmul(depth_intrinsic, points_cam[..., None])

        color_points_pixel_homo = color_points_pixel_homo[..., 0]
        depth_points_pixel_homo = depth_points_pixel_homo[..., 0]

        color_points_pixel = (
            torch.div(
                color_points_pixel_homo[..., :-1],
                torch.clip(color_points_pixel_homo[..., [-1]], min=1e-8),
            )
            .round()
            .to(torch.int32)
        )
        depth_points_pixel = (
            torch.div(
                depth_points_pixel_homo[..., :-1],
                torch.clip(depth_points_pixel_homo[..., [-1]], min=1e-8),
            )
            .round()
            .to(torch.int32)
        )

        final_points_cam[batch_start : batch_start + batch_size] = (
            points_cam.cpu().numpy()
        )
        final_color_points_pixel[batch_start : batch_start + batch_size] = (
            color_points_pixel.cpu().numpy()
        )
        final_depth_points_pixel[batch_start : batch_start + batch_size] = (
            depth_points_pixel.cpu().numpy()
        )

    torch.cuda.empty_cache()
    return (final_points_cam, final_color_points_pixel, final_depth_points_pixel)


def calculate_visible_points_per_superpoint_numpy(
    color_pixes, points_seen, seg_num, seg_members, weight_points
):
    frame_count = points_seen.shape[1]

    visible_counts = np.zeros((seg_num, frame_count), dtype=float)

    visible_weight_counts = np.zeros((seg_num, frame_count), dtype=float)

    visible_coords = np.empty((seg_num, frame_count), dtype=object)
    visible_weight_coords = np.empty((seg_num, frame_count), dtype=object)
    for seg_idx in range(seg_num):
        for frame_idx in range(frame_count):
            visible_coords[seg_idx, frame_idx] = np.empty((0, 2), dtype=int)
            visible_weight_coords[seg_idx, frame_idx] = np.empty((0, 1), dtype=int)

    for seg_idx in range(seg_num):
        point_indices = np.array(seg_members[seg_idx]["point_indices"])

        visibility = points_seen[point_indices]
        weight = weight_points[point_indices]

        visible_counts[seg_idx] = visibility.sum(axis=0)
        visible_weight_counts[seg_idx] = weight.sum(axis=0)

        visible_points = color_pixes[point_indices]

        for frame_idx in range(frame_count):
            visible = visibility[:, frame_idx].astype(bool)
            coords = visible_points[visible, frame_idx]
            coords_weight = weight[visible, frame_idx]

            if coords.size > 0:
                visible_coords[seg_idx, frame_idx] = coords
                visible_weight_coords[seg_idx, frame_idx] = coords_weight

    return visible_counts, visible_coords, visible_weight_counts, visible_weight_coords


def calculate_top_k_visible_frames(
    visible_counts, visible_coords, seg_count, visible_weight_coords, iou=0.3
):
    """Select frames whose visible-point ratio exceeds ``iou`` per segment."""
    N, M = visible_counts.shape

    top_k_frames_list = []
    top_k_coords_list = []
    top_k_weight_coords_list = []

    for i in range(N):
        counts = visible_counts[i]

        ratio = counts / seg_count[i]
        top_k_indices = np.where(ratio > iou)[0]

        if len(top_k_indices) > 0:
            top_k_coords = [visible_coords[i][frame_idx] for frame_idx in top_k_indices]
            top_k_weight_coords = [
                visible_weight_coords[i][frame_idx] for frame_idx in top_k_indices
            ]
        else:
            top_k_coords = []
            top_k_weight_coords = []

        top_k_frames_list.append(top_k_indices.tolist())
        top_k_coords_list.append(top_k_coords)
        top_k_weight_coords_list.append(top_k_weight_coords)

    return top_k_frames_list, top_k_coords_list, top_k_weight_coords_list


def _mask_geometry_identity(masks):
    """Return a scene-local identity key without copying mask pixels."""
    if torch.is_tensor(masks):
        return (
            "torch",
            id(masks),
            int(masks.data_ptr()),
            tuple(masks.shape),
            tuple(masks.stride()),
            str(masks.dtype),
            str(masks.device),
            int(masks._version),
        )
    array = np.asarray(masks)
    return (
        "numpy",
        id(masks),
        int(array.__array_interface__["data"][0]),
        tuple(array.shape),
        tuple(array.strides),
        str(array.dtype),
    )


def filter_masks(
    masks,
    mask_weight,
    overlap_threshold=0.5,
    overlap_cache=None,
    return_overlap_cache=False,
):
    """Filter masks, optionally reusing geometry-only overlap pairs."""
    geometry_identity = _mask_geometry_identity(masks)
    n = masks.shape[0]

    if overlap_cache is None:
        gpu_masks = torch.tensor(masks, dtype=torch.float32, device=device)
        flat_masks = gpu_masks.reshape(n, -1)
        intersection = torch.mm(flat_masks, flat_masks.t())
        area = flat_masks.sum(dim=1)
        union = area.unsqueeze(1) + area.unsqueeze(0) - intersection
        overlap_ratio = intersection / union
        high_overlap = (overlap_ratio > overlap_threshold).triu(diagonal=1)
        high_overlap_indices = high_overlap.nonzero(as_tuple=False)
        overlap_cache = {
            "geometry_identity": geometry_identity,
            "overlap_threshold": overlap_threshold,
            "mask_count": n,
            "high_overlap_indices": high_overlap_indices,
        }
    else:
        if (
            overlap_cache["geometry_identity"] != geometry_identity
            or overlap_cache["overlap_threshold"] != overlap_threshold
            or overlap_cache["mask_count"] != n
        ):
            raise ValueError("overlap_cache does not match the input mask geometry")
        high_overlap_indices = overlap_cache["high_overlap_indices"]

    gpu_mask_weight = torch.tensor(mask_weight, dtype=torch.float32, device=device)
    keep_mask = torch.ones(n, dtype=torch.bool, device=device)
    if high_overlap_indices.shape[0] > 0:
        first = high_overlap_indices[:, 0]
        second = high_overlap_indices[:, 1]
        losers = torch.where(
            gpu_mask_weight[first] < gpu_mask_weight[second],
            first,
            second,
        )
        keep_mask[losers] = False

    keep = keep_mask.cpu().numpy()
    if return_overlap_cache:
        return keep, overlap_cache
    return keep


def compute_mask_weights(
    top_k_frames_list,
    top_k_coords_list,
    top_k_weight_coords_list,
    mask_list,
    seg_counts,
    seg_member_count_union,
    points_seen,
    cw0,
    ch0,
    weight_points,
    threshold,
):
    N = len(top_k_frames_list)
    M = len(mask_list)
    mask_w1 = [np.zeros(len(masks), dtype=np.float32) for masks in mask_list]
    mask_w2 = [np.zeros(len(masks), dtype=np.float32) for masks in mask_list]

    P = points_seen.shape[0]
    num_seg = len(seg_member_count_union)
    point2seg = np.full(P, -1, dtype=np.int32)
    for seg_id, d in seg_member_count_union.items():
        if "point_indices" in d and len(d["point_indices"]) > 0:
            point2seg[np.asarray(d["point_indices"], dtype=np.int64)] = int(seg_id)

    points_seen = points_seen.astype(bool)
    cw0 = cw0.astype(np.int64)
    ch0 = ch0.astype(np.int64)

    super_point_mask_info = {}

    for super_idx in tqdm(range(N)):
        frames = top_k_frames_list[super_idx]
        coords_list = top_k_coords_list[super_idx]

        super_point_mask_info[super_idx] = []
        for idx, frame_idx in enumerate(frames):
            coords = coords_list[idx]
            if coords.size == 0:
                continue

            mask_data = mask_list[frame_idx]
            num_masks, H, W = mask_data.shape
            x_coords, y_coords = coords[:, 0], coords[:, 1]

            num_visible_points = coords.shape[0]

            intersection = (
                mask_data[:, y_coords, x_coords].sum(axis=1).astype(np.float32)
            )

            overlap_ratio = intersection / max(1, num_visible_points)
            valid_mask_idx = np.where(overlap_ratio > threshold)[0]
            if valid_mask_idx.size > 0:
                super_point_mask_info[super_idx].append((frame_idx, valid_mask_idx))

                mask_w1[frame_idx][valid_mask_idx] += intersection[valid_mask_idx]
                mask_w2[frame_idx][valid_mask_idx] += num_visible_points

    pairs_by_frame = defaultdict(list)
    for sp, items in super_point_mask_info.items():
        for f, mask_ids in items:
            for mid in np.atleast_1d(mask_ids):
                pairs_by_frame[f].append(int(mid))

    for f in list(pairs_by_frame.keys()):
        if len(pairs_by_frame[f]) == 0:
            pairs_by_frame.pop(f)
        else:
            pairs_by_frame[f] = sorted(set(pairs_by_frame[f]))

    dist_cache = {}

    for f, mask_ids in pairs_by_frame.items():
        mask_data = mask_list[f].astype(bool)
        Kf, H, W = mask_data.shape

        vis = points_seen[:, f]
        p_idx = np.flatnonzero(vis & (point2seg >= 0))
        if p_idx.size == 0:
            for mid in mask_ids:
                dist_cache[(f, mid)] = np.zeros((num_seg,), dtype=np.float32)
            continue

        x = cw0[p_idx, f]
        y = ch0[p_idx, f]

        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not np.any(inb):
            for mid in mask_ids:
                dist_cache[(f, mid)] = np.zeros((num_seg,), dtype=np.float32)
            continue

        p_idx = p_idx[inb]
        weights = weight_points[p_idx, f]
        x = x[inb]
        y = y[inb]
        flat = (y * W + x).astype(np.int64)

        seg_ids = point2seg[p_idx].astype(np.int64)

        order = np.argsort(seg_ids, kind="mergesort")
        seg_sorted = seg_ids[order]
        uniq_seg, start_idx = np.unique(seg_sorted, return_index=True)

        boundaries = np.concatenate([start_idx, [seg_sorted.size]])
        group_sizes = np.diff(boundaries).astype(np.float32)

        sel = np.asarray(mask_ids, dtype=np.int64)
        mflat = mask_data[sel].reshape(sel.size, -1)

        inside_bool = mflat[:, flat]

        inside_bool_sorted = inside_bool[:, order]
        weights_sorted = weights[order]

        counts_inside = np.add.reduceat(
            inside_bool_sorted.astype(np.int32), boundaries[:-1], axis=1
        ).astype(np.float32)

        weighted_inside_sorted = (
            inside_bool_sorted.astype(np.float32) * weights_sorted[None, :]
        )
        sum_w_inside = np.add.reduceat(
            weighted_inside_sorted, boundaries[:-1], axis=1
        ).astype(np.float32)

        mean_w_inside = np.divide(
            sum_w_inside,
            np.maximum(counts_inside, 1.0),
            out=np.zeros_like(sum_w_inside, dtype=np.float32),
            where=counts_inside > 0,
        )

        ratio_inside = counts_inside / group_sizes[None, :]

        dist_local = (ratio_inside * mean_w_inside).astype(np.float32)

        dist_batch = np.zeros((sel.size, num_seg), dtype=np.float32)
        dist_batch[:, uniq_seg] = dist_local

        for i, mid in enumerate(sel):
            dist_cache[(f, int(mid))] = dist_batch[i]

    A_dict = {}
    cos_sim_avg_dict = {}
    pairs_dict = {}

    for sp, items in super_point_mask_info.items():
        pairs = []
        for f, mids in items:
            for mid in np.atleast_1d(mids):
                pairs.append((f, int(mid)))

        if len(pairs) == 0:
            A = np.zeros((0, num_seg), dtype=np.float32)
            A_dict[sp] = A
            cos_sim_avg_dict[sp] = np.zeros((0,), dtype=np.float32)
            pairs_dict[sp] = []
            continue

        A = np.vstack([dist_cache[(f, m)] for (f, m) in pairs]).astype(np.float32)

        if A.shape[0] == 1:
            cos_avg = np.array([1.0], dtype=np.float32)
        else:
            norms = np.linalg.norm(A, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            A_norm = A / norms
            sim = A_norm @ A_norm.T

            np.fill_diagonal(sim, 0.0)
            cos_avg = sim.sum(axis=1) / (A.shape[0] - 1)
            cos_avg = cos_avg.astype(np.float32)

        A_dict[sp] = A
        cos_sim_avg_dict[sp] = cos_avg
        pairs_dict[sp] = pairs

    mask_cos_collect = defaultdict(list)
    for sp in cos_sim_avg_dict:
        items = super_point_mask_info[sp]
        pos = 0
        for f, mids in items:
            mids = np.atleast_1d(mids)
            for mid in mids:
                mask_cos_collect[(f, int(mid))].append(cos_sim_avg_dict[sp][pos])
                pos += 1

    mask_cos_avg = [np.zeros(len(mask_list[f]), dtype=np.float32) for f in range(M)]
    for (f, m), vals in mask_cos_collect.items():
        mask_cos_avg[f][m] = np.mean(vals)

    mask_weights = mask_cos_avg

    return mask_weights


def update_masks(mask_list, keep_idx2, mask_weight):
    """Build frame labels without allocating a K x H x W id layer."""
    frame_count = len(mask_list)
    height, width = mask_list[0].shape[1:]
    masks_2D = np.zeros((frame_count, height, width), dtype=np.uint8)

    for frame_index in range(frame_count):
        masks = mask_list[frame_index]
        keep = keep_idx2[frame_index]
        weights = mask_weight[frame_index]
        indices = np.where(keep)[0]
        if len(indices) == 0:
            continue

        selected_weights = weights[indices]
        order = np.lexsort((indices, -selected_weights))
        ordered_indices = indices[order]
        class_ids = np.arange(len(indices), 0, -1)
        layer_dtype = np.result_type(masks.dtype, class_ids.dtype)
        frame_max = np.zeros((height, width), dtype=layer_dtype)
        candidate = np.empty((height, width), dtype=layer_dtype)

        for mask_index, class_id in zip(ordered_indices, class_ids):
            np.multiply(
                masks[mask_index],
                class_id,
                out=candidate,
                casting="unsafe",
            )
            np.maximum(frame_max, candidate, out=frame_max)

        masks_2D[frame_index] = frame_max
    return masks_2D
