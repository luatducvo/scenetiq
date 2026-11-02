"""Class-agnostic instance AP and Query threshold selection."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

AP_IOU_THRESHOLDS = np.arange(0.5, 0.951, 0.05)


def _iou_matrix(
    pred: np.ndarray, gt: np.ndarray, pred_ids: list[int], gt_ids: list[int]
) -> np.ndarray:
    ious = np.zeros((len(pred_ids), len(gt_ids)))
    gt_masks = [gt == g for g in gt_ids]
    for i, p in enumerate(pred_ids):
        pm = pred == p
        for j, gm in enumerate(gt_masks):
            inter = np.count_nonzero(pm & gm)
            if inter:
                ious[i, j] = inter / np.count_nonzero(pm | gm)
    return ious


def _average_precision(ious: np.ndarray, threshold: float) -> float:
    """All-point interpolated AP for predictions already sorted best first."""
    n_pred, n_gt = ious.shape
    if n_gt == 0:
        return float("nan")
    if n_pred == 0:
        return 0.0
    matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(n_pred)
    for i in range(n_pred):
        candidates = np.where(~matched & (ious[i] >= threshold))[0]
        if candidates.size:
            j = candidates[np.argmax(ious[i, candidates])]
            matched[j] = True
            tp[i] = 1
    ctp = np.cumsum(tp)
    precision = ctp / np.arange(1, n_pred + 1)
    recall = ctp / n_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(steps * envelope))


def instance_ap(
    pred_ids: npt.ArrayLike,
    gt_ids: npt.ArrayLike,
    scores: dict[int, float] | None = None,
) -> dict[str, float]:
    """Class-agnostic instance AP over the same points.

    Args:
        pred_ids: Predicted Instance id per ground-truth point, -1 for none.
        gt_ids: Ground-truth instance id per point, -1 for ignored / none.
        scores: Confidence per predicted id; defaults to its point count.

    Returns:
        ``AP`` (mean over IoU 0.5:0.95), ``AP50`` and ``AP25``, using all-point
        interpolation of the precision-recall curve.
    """
    pred = np.asarray(pred_ids)
    gt = np.asarray(gt_ids)
    p_ids = [int(i) for i in np.unique(pred) if i >= 0]
    g_ids = [int(i) for i in np.unique(gt) if i >= 0]
    conf = scores or {p: float(np.count_nonzero(pred == p)) for p in p_ids}
    p_ids.sort(key=lambda p: -conf.get(p, 0.0))
    ious = _iou_matrix(pred, gt, p_ids, g_ids)
    return {
        "AP": float(np.mean([_average_precision(ious, t) for t in AP_IOU_THRESHOLDS])),
        "AP50": _average_precision(ious, 0.5),
        "AP25": _average_precision(ious, 0.25),
    }


@dataclass(frozen=True)
class QueryRecord:
    """One (Query, Instance) pair: its relevance score and whether it is a right answer."""

    score: float
    correct: bool


def best_f1_threshold(records: Sequence[QueryRecord]) -> tuple[float, float]:
    """The score threshold maximising F1 of ``score >= threshold`` against ``correct``.

    Returns:
        ``(threshold, f1)``; the highest such threshold on ties.
    """
    scores = np.array([r.score for r in records])
    correct = np.array([r.correct for r in records], dtype=bool)
    best = (1.0, 0.0)
    for t in np.unique(scores)[::-1]:
        positive = scores >= t
        tp = np.count_nonzero(positive & correct)
        fp = np.count_nonzero(positive & ~correct)
        fn = np.count_nonzero(~positive & correct)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best
