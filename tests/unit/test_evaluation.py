import json
from pathlib import Path

import numpy as np
import pytest

from scenetiq.evaluation.alignment import apply_similarity, umeyama
from scenetiq.evaluation.metrics import QueryRecord, best_f1_threshold, instance_ap
from scenetiq.evaluation.sampling import sample_evenly
from scenetiq.evaluation.scannet import load_gt_instances
from scenetiq.evaluation.transfer import transfer_instances


def test_sample_evenly_spreads_over_all_valid_frames():
    assert sample_evenly(list(range(10)), 5) == [0, 2, 4, 6, 8]
    assert sample_evenly(list(range(3)), 300) == [0, 1, 2]
    # step = 10 / 3: floor(0), floor(3.33), floor(6.67)
    assert sample_evenly(list(range(10)), 3) == [0, 3, 6]


def test_umeyama_recovers_a_known_similarity_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    angle = np.pi / 5
    rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    dst = 2.5 * src @ rot.T + np.array([1.0, -2.0, 0.5])

    scale, r, t = umeyama(src, dst)

    assert scale == pytest.approx(2.5)
    np.testing.assert_allclose(r, rot, atol=1e-9)
    np.testing.assert_allclose(t, [1.0, -2.0, 0.5], atol=1e-9)
    np.testing.assert_allclose(apply_similarity(src, scale, r, t), dst, atol=1e-9)


def test_transfer_gives_each_gt_point_its_nearest_predicted_instance_within_range():
    pred_points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    pred_ids = np.array([7, 3], dtype=np.int32)
    gt_points = np.array([[0.02, 0, 0], [0.97, 0, 0], [5, 5, 5]], dtype=np.float32)

    ids = transfer_instances(pred_points, pred_ids, gt_points, max_distance=0.1)

    assert ids.tolist() == [7, 3, -1]


def test_ap_is_one_for_perfect_predictions():
    gt = np.array([0, 0, 1, 1, -1])
    pred = np.array([5, 5, 9, 9, -1])
    ap = instance_ap(pred, gt)
    assert ap == {"AP": pytest.approx(1.0), "AP50": pytest.approx(1.0), "AP25": pytest.approx(1.0)}


def test_ap_counts_a_low_overlap_prediction_only_at_loose_thresholds():
    gt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Instance 1 is predicted with IoU 3/8 (3 shared points, union 8 incl. 4 extra).
    pred = np.array([2, 2, 2, 2, -1, 4, 4, 4])
    pred_extra = np.concatenate([pred, [4, 4, 4, 4]])
    gt_extra = np.concatenate([gt, [-1, -1, -1, -1]])
    # The exact prediction (2) ranks above the loose one (4).
    ap = instance_ap(pred_extra, gt_extra, scores={2: 1.0, 4: 0.5})
    # IoU 0.25: both hit -> precision 1 at recall 0.5 and 1.0.
    assert ap["AP25"] == pytest.approx(1.0)
    # IoU 0.5: hit (P=1, R=0.5) then miss -> AP = 1.0 * 0.5.
    assert ap["AP50"] == pytest.approx(0.5)
    # Every threshold 0.5..0.95 behaves like 0.5 (IoUs are 1.0 and 0.375).
    assert ap["AP"] == pytest.approx(0.5)


def test_ap_ranks_predictions_by_size_when_no_scores_are_given():
    gt = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1])
    pred = np.array([2, 2, 2, 2, -1, 4, 4, 4, 4, 4, 4, 4])
    # The 7-point miss now ranks first: P=0 at rank 1, P=1/2 at R=0.5 -> AP50 = 0.25.
    assert instance_ap(pred, gt)["AP50"] == pytest.approx(0.25)


def test_ap_with_no_predictions_is_zero():
    ap = instance_ap(np.array([-1, -1]), np.array([0, 0]))
    assert ap["AP50"] == 0.0


def test_best_f1_threshold_separates_correct_from_wrong_matches():
    records = [
        QueryRecord(score=0.9, correct=True),
        QueryRecord(score=0.8, correct=True),
        QueryRecord(score=0.6, correct=False),
        QueryRecord(score=0.4, correct=True),
        QueryRecord(score=0.2, correct=False),
    ]
    threshold, f1 = best_f1_threshold(records)
    # t = 0.8: TP 2, FP 0, FN 1 -> F1 0.8.  t = 0.4: TP 3, FP 1, FN 0 -> F1 0.857.
    assert threshold == pytest.approx(0.4)
    assert f1 == pytest.approx(6 / 7)


def test_load_gt_instances_from_scannet_aggregation(tmp_path: Path):
    scene = tmp_path / "scene0000_00"
    scene.mkdir()
    (scene / "scene0000_00_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [10, 10, 20, 30, 30, 40]})
    )
    (scene / "scene0000_00.aggregation.json").write_text(
        json.dumps(
            {
                "segGroups": [
                    {"objectId": 0, "label": "chair", "segments": [10]},
                    {"objectId": 1, "label": "wall", "segments": [20]},
                    {"objectId": 2, "label": "Office Chair", "segments": [30]},
                ]
            }
        )
    )

    gt = load_gt_instances(scene, "scene0000_00", ignore_labels={"wall", "floor"})

    assert gt.instance_ids.tolist() == [0, 0, -1, 1, 1, -1]
    assert gt.labels == ["chair", "office chair"]


def test_regression_check_flags_only_drops_beyond_tolerance():
    from scenetiq.evaluation.harness import regressions

    baseline = {"AP": 0.30, "AP50": 0.50, "AP25": 0.60}
    current = {"AP": 0.295, "AP50": 0.45, "AP25": 0.70}
    assert regressions(current, baseline, tolerance=0.01) == ["AP50 0.450 < baseline 0.500 - 0.010"]
    assert regressions(current, baseline, tolerance=0.1) == []
