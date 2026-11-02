import numpy as np

from scenetiq.pipeline.masking import select_masks
from scenetiq.pipeline.merging import labels_to_instance_ids
from scenetiq.pipeline.primitives import voxel_primitives


def test_points_in_the_same_voxel_share_a_primitive():
    points = np.array(
        [[0.01, 0.01, 0.01], [0.09, 0.05, 0.02], [0.11, 0.0, 0.0], [-0.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    ids = voxel_primitives(points, voxel_size=0.1)
    assert ids[0] == ids[1]
    assert len({ids[0], ids[2], ids[3]}) == 3
    # Consecutive ids starting at 0.
    assert sorted(set(ids.tolist())) == [0, 1, 2]


def test_mv3dis_labels_become_instance_ids_largest_first_with_minus_one_for_none():
    # 0 = no instance; 907 and 512 are MV3DIS region ids.
    labels = np.array([0, 907, 512, 512, 907, 512, 0])
    ids = labels_to_instance_ids(labels, min_points=1)
    assert ids.tolist() == [-1, 1, 0, 0, 1, 0, -1]
    assert ids.dtype == np.int32


def test_instances_smaller_than_min_points_are_dropped():
    labels = np.array([5, 5, 5, 9])
    assert labels_to_instance_ids(labels, min_points=2).tolist() == [0, 0, 0, -1]


def test_select_masks_keeps_best_scored_masks_above_min_area():
    masks = np.zeros((4, 4, 4), dtype=bool)
    masks[0, :2, :2] = True  # area 4
    masks[1, :, :] = True  # area 16
    masks[2, 0, 0] = True  # area 1 (too small)
    masks[3, 2:, 2:] = True  # area 4
    scores = np.array([0.5, 0.9, 0.99, 0.7], dtype=np.float32)

    kept, kept_scores = select_masks(masks, scores, max_masks=2, min_area=2)

    assert kept.shape == (2, 4, 4)
    np.testing.assert_allclose(kept_scores, [0.9, 0.7])
    assert kept.dtype == np.uint8
    assert kept[0].sum() == 16
