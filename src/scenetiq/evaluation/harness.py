"""Run the real Pipeline on ScanNet200 scenes, report instance AP and tune the Match threshold.

This is Seam B: the end-to-end regression test of every pipeline stage. It
runs on Colab only, needs ``requirements-gpu.txt``, and is invoked with
``python -m scenetiq.evaluation --help``.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scenetiq.domain import Capture, Reconstruction, View
from scenetiq.evaluation.alignment import apply_similarity, umeyama
from scenetiq.evaluation.metrics import QueryRecord, best_f1_threshold, instance_ap
from scenetiq.evaluation.sampling import sample_evenly
from scenetiq.evaluation.scannet import (
    DEFAULT_IGNORE_LABELS,
    load_frames,
    load_gt_instances,
    load_mesh_vertices,
)
from scenetiq.evaluation.transfer import transfer_instances

logger = logging.getLogger(__name__)

METRICS = ("AP", "AP50", "AP25")


@dataclass(frozen=True)
class HarnessConfig:
    """Where the data is and how to evaluate.

    Attributes:
        scans_dir: ScanNet ``scans/`` directory (``<scene>/<scene>_vh_clean_2.ply``,
            ``<scene>.aggregation.json``, ``<scene>_vh_clean_2.0.010000.segs.json``).
        frames_dir: Exported 2D frames (``<scene>/color/*.jpg``, ``<scene>/pose/*.txt``).
        scene_ids: Scenes to evaluate.
        out_dir: Where the report is written.
        max_views: Views sampled per scene, from its valid ScanNet frames (the Capture limit).
        transfer_distance: Max metres between a GT vertex and its nearest predicted point.
        query_iou: IoU with a GT instance of the queried class for a Match to count as right.
        ignore_labels: GT categories left out of both AP and Queries (e.g. walls, floors).
    """

    scans_dir: Path
    frames_dir: Path
    scene_ids: list[str]
    out_dir: Path
    max_views: int = 300
    transfer_distance: float = 0.1
    query_iou: float = 0.5
    ignore_labels: frozenset[str] = field(default=DEFAULT_IGNORE_LABELS)


def regressions(
    current: dict[str, float], baseline: dict[str, float], tolerance: float
) -> list[str]:
    """Describe every metric that dropped more than ``tolerance`` below ``baseline``."""
    return [
        f"{k} {current[k]:.3f} < baseline {baseline[k]:.3f} - {tolerance:.3f}"
        for k in METRICS
        if k in baseline and current.get(k, 0.0) < baseline[k] - tolerance
    ]


def _align_to_scannet(rec: Reconstruction, poses: list[np.ndarray]) -> np.ndarray:
    src = np.stack([c.cam_to_world[:3, 3] for c in rec.cameras])
    dst = np.stack([p[:3, 3] for p in poses])
    scale, rot, t = umeyama(src, dst)
    logger.info("alignment scale %.3f (1.0 = DA3 metric scale matches ScanNet)", scale)
    return apply_similarity(rec.points, scale, rot, t)


def evaluate_scene(cfg: HarnessConfig, scene_id: str, pipeline, encoder) -> dict:
    """Run one scene and return its metrics plus the per-Query records."""
    frames = sample_evenly(load_frames(cfg.frames_dir / scene_id), cfg.max_views)
    capture = Capture(views=[View(path=f.color_path) for f in frames])
    rec: Reconstruction = pipeline.run(capture, lambda s: logger.info("%s: %s", scene_id, s))

    aligned = _align_to_scannet(rec, [f.cam_to_world for f in frames])
    scene_dir = cfg.scans_dir / scene_id
    gt = load_gt_instances(scene_dir, scene_id, cfg.ignore_labels)
    gt_points = load_mesh_vertices(scene_dir, scene_id)
    pred = transfer_instances(aligned, rec.instance_ids, gt_points, cfg.transfer_distance)
    metrics = instance_ap(pred, gt.instance_ids)

    records: list[QueryRecord] = []
    embedded = [i for i in rec.instances if i.embedding is not None]
    if embedded:
        matrix = np.stack([i.embedding for i in embedded if i.embedding is not None])
        for label in sorted(set(gt.labels)):
            right = _instances_matching_label(pred, gt.instance_ids, gt.labels, label, cfg)
            scores = encoder.to_score(matrix @ encoder.encode(label))
            records += [
                QueryRecord(score=float(s), correct=inst.instance_id in right)
                for inst, s in zip(embedded, scores, strict=True)
            ]
    return {"scene_id": scene_id, "views": len(frames), **metrics, "records": records}


def _instances_matching_label(
    pred: np.ndarray, gt_ids: np.ndarray, labels: list[str], label: str, cfg: HarnessConfig
) -> set[int]:
    targets = [g for g, lab in enumerate(labels) if lab == label]
    right = set()
    for p in np.unique(pred[pred >= 0]):
        pm = pred == p
        for g in targets:
            gm = gt_ids == g
            union = np.count_nonzero(pm | gm)
            if union and np.count_nonzero(pm & gm) / union >= cfg.query_iou:
                right.add(int(p))
    return right


def run(cfg: HarnessConfig, pipeline, encoder) -> dict:
    """Evaluate every scene, write ``report.json`` and ``report.md``, and return the report."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [evaluate_scene(cfg, s, pipeline, encoder) for s in cfg.scene_ids]
    records = [r for s in scenes for r in s["records"]]
    threshold, f1 = best_f1_threshold(records) if records else (float("nan"), 0.0)
    report = {
        "aggregate": {k: float(np.nanmean([s[k] for s in scenes])) for k in METRICS},
        "recommended_threshold": threshold,
        "threshold_f1": f1,
        "scenes": [{k: v for k, v in s.items() if k != "records"} for s in scenes],
    }
    (cfg.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (cfg.out_dir / "report.md").write_text(_markdown(report))
    return report


def _markdown(report: dict) -> str:
    agg = report["aggregate"]
    lines = [
        "# SceneTIQ evaluation (ScanNet200, class-agnostic)",
        "",
        f"- AP {agg['AP']:.3f} | AP50 {agg['AP50']:.3f} | AP25 {agg['AP25']:.3f}",
        f"- Recommended Match threshold: {report['recommended_threshold']:.3f} "
        f"(F1 {report['threshold_f1']:.3f})",
        "",
        "| Scene | Views | AP | AP50 | AP25 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {s['scene_id']} | {s['views']} | {s['AP']:.3f} | {s['AP50']:.3f} | {s['AP25']:.3f} |"
        for s in report["scenes"]
    ]
    return "\n".join(lines) + "\n"
