"""Reading ScanNet scenes: ground-truth instances, exported RGB frames and poses."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

DEFAULT_IGNORE_LABELS = frozenset({"wall", "floor", "ceiling"})


@dataclass
class GroundTruth:
    """Ground-truth instances on the ScanNet mesh vertices.

    Attributes:
        instance_ids: ``(V,)`` instance index per vertex, -1 for ignored or unlabelled.
        labels: Lower-cased raw category of each instance index.
    """

    instance_ids: npt.NDArray[np.int32]
    labels: list[str]


def load_gt_instances(
    scene_dir: Path,
    scene_id: str,
    ignore_labels: frozenset[str] | set[str] = DEFAULT_IGNORE_LABELS,
) -> GroundTruth:
    """Read ``<scene>.aggregation.json`` and ``<scene>_vh_clean_2.0.010000.segs.json``."""
    segs = json.loads((scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json").read_text())
    seg_indices = np.asarray(segs["segIndices"])
    agg = json.loads((scene_dir / f"{scene_id}.aggregation.json").read_text())
    instance_ids = np.full(len(seg_indices), -1, dtype=np.int32)
    labels: list[str] = []
    for group in agg["segGroups"]:
        label = str(group["label"]).strip().lower()
        if label in ignore_labels:
            continue
        instance_ids[np.isin(seg_indices, group["segments"])] = len(labels)
        labels.append(label)
    return GroundTruth(instance_ids=instance_ids, labels=labels)


def load_mesh_vertices(scene_dir: Path, scene_id: str) -> npt.NDArray[np.float32]:
    """Vertices of ``<scene>_vh_clean_2.ply``, the points the ground truth is defined on."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(scene_dir / f"{scene_id}_vh_clean_2.ply"))
    return np.asarray(mesh.vertices, dtype=np.float32)


@dataclass(frozen=True)
class Frame:
    """One exported ScanNet RGB frame with its camera-to-world pose."""

    color_path: Path
    cam_to_world: npt.NDArray[np.float64]


def load_frames(frames_dir: Path) -> list[Frame]:
    """Frames of an exported ScanNet scene (``color/<i>.jpg`` and ``pose/<i>.txt``).

    Frames whose pose is not finite (ScanNet marks tracking failures with -inf)
    are skipped. Frames are ordered by index.
    """
    frames = []
    for color in sorted((frames_dir / "color").glob("*.jpg"), key=lambda p: int(p.stem)):
        pose = np.loadtxt(frames_dir / "pose" / f"{color.stem}.txt")
        if pose.shape == (4, 4) and np.all(np.isfinite(pose)):
            frames.append(Frame(color_path=color, cam_to_world=pose))
    return frames
