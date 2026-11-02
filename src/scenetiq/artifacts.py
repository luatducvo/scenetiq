"""On-disk format of a Reconstruction inside a Scene directory.

Files:
    ``points.ply``: binary little-endian PLY with float x/y/z and uchar r/g/b.
    ``instance_ids.bin``: little-endian int32 per point, same order as the PLY.
    ``instances.json``: id, centroid, bbox_min, bbox_max of every Instance.
    ``embeddings.npy``: ``(K, D)`` float32, row ``i`` belongs to ``instances.json[i]``.
    ``cameras.json``: camera-to-world, intrinsics and size of every View.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scenetiq.domain import Camera, FloatArray, Instance, Reconstruction

POINTS_FILE = "points.ply"
INSTANCE_IDS_FILE = "instance_ids.bin"
INSTANCES_FILE = "instances.json"
EMBEDDINGS_FILE = "embeddings.npy"
CAMERAS_FILE = "cameras.json"

_VERTEX_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
)


def encode_ply(points: FloatArray, colors: np.ndarray) -> bytes:
    """Encode points and colours as a binary little-endian PLY."""
    vertices = np.empty(len(points), dtype=_VERTEX_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    return header.encode("ascii") + vertices.tobytes()


def decode_ply(data: bytes) -> tuple[FloatArray, np.ndarray]:
    """Decode a PLY written by :func:`encode_ply` into ``(points, colors)``."""
    marker = b"end_header\n"
    end = data.index(marker) + len(marker)
    vertices = np.frombuffer(data[end:], dtype=_VERTEX_DTYPE)
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1)
    return points.astype(np.float32), colors.astype(np.uint8)


def save_reconstruction(directory: Path, reconstruction: Reconstruction) -> None:
    """Write ``reconstruction`` into ``directory`` (created if missing)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / POINTS_FILE).write_bytes(encode_ply(reconstruction.points, reconstruction.colors))
    (directory / INSTANCE_IDS_FILE).write_bytes(reconstruction.instance_ids.astype("<i4").tobytes())
    instances = reconstruction.instances
    (directory / INSTANCES_FILE).write_text(
        json.dumps(
            [
                {
                    "instance_id": inst.instance_id,
                    "centroid": inst.centroid.tolist(),
                    "bbox_min": inst.bbox_min.tolist(),
                    "bbox_max": inst.bbox_max.tolist(),
                }
                for inst in instances
            ]
        )
    )
    with_embedding = [inst.embedding for inst in instances if inst.embedding is not None]
    if instances and len(with_embedding) == len(instances):
        np.save(directory / EMBEDDINGS_FILE, np.stack(with_embedding).astype(np.float32))
    (directory / CAMERAS_FILE).write_text(
        json.dumps(
            [
                {
                    "cam_to_world": cam.cam_to_world.tolist(),
                    "intrinsics": cam.intrinsics.tolist(),
                    "width": cam.width,
                    "height": cam.height,
                }
                for cam in reconstruction.cameras
            ]
        )
    )


def load_instances(directory: Path) -> list[Instance]:
    """Load the Instances (with embeddings, if stored) from ``directory``."""
    rows = json.loads((directory / INSTANCES_FILE).read_text())
    emb_path = directory / EMBEDDINGS_FILE
    embeddings = np.load(emb_path) if emb_path.exists() else None
    return [
        Instance(
            instance_id=int(row["instance_id"]),
            centroid=np.asarray(row["centroid"], dtype=np.float32),
            bbox_min=np.asarray(row["bbox_min"], dtype=np.float32),
            bbox_max=np.asarray(row["bbox_max"], dtype=np.float32),
            embedding=None if embeddings is None else embeddings[i],
        )
        for i, row in enumerate(rows)
    ]


def load_cameras(directory: Path) -> list[Camera]:
    """Load the camera of every View from ``directory``."""
    rows = json.loads((directory / CAMERAS_FILE).read_text())
    return [
        Camera(
            cam_to_world=np.asarray(r["cam_to_world"], dtype=np.float32),
            intrinsics=np.asarray(r["intrinsics"], dtype=np.float32),
            width=int(r["width"]),
            height=int(r["height"]),
        )
        for r in rows
    ]
