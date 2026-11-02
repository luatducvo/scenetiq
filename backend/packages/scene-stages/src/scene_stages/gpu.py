"""Lazy model adapters."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from scene_core.contracts import AlignManifest, ReconManifest, SegmentManifest
from scene_core.errors import SceneError
from scene_core.geometry import fit_plane
from scene_core.ports import StageAlgorithm


def _cache():
    root = Path(os.environ.get("MODELS_DIR", "models"))
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))


def _cuda():
    _cache()
    import torch

    if not torch.cuda.is_available():
        raise SceneError("GPU_REQUIRED", "This Algorithm requires a CUDA compute Target")
    return torch


def _numpy(value):
    return value.detach().float().cpu().numpy()


class DepthAnything3Reconstruction(StageAlgorithm):
    name = "recon.da3"
    stage = "recon"
    manifest_type = ReconManifest
    reads = ["keyframes"]
    writes = ["recon"]
    model_id = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

    def setup(self):
        _cache()
        self.model = None

    def download_weights(self):
        _cache()
        from huggingface_hub import snapshot_download

        snapshot_download(self.model_id)

    def run(self, workspace, params):
        torch = _cuda()
        from depth_anything_3.api import DepthAnything3

        model_id = params.get("model_id", self.model_id)
        if self.model is None or model_id != self.model_id:
            self.model = DepthAnything3.from_pretrained(model_id).to("cuda").eval()
            self.model_id = model_id
        files = sorted(workspace.path("keyframes").glob("*.jpg"))
        if not files:
            raise SceneError("NOT_ENOUGH_KEYFRAMES", "Reconstruction requires Keyframes")
        self.progress(0, 1, "batch")
        # Every Keyframe goes through one forward pass so all views share one metric world frame.
        with torch.inference_mode():
            prediction = self.model.inference(
                [str(path) for path in files],
                use_ray_pose=bool(params.get("use_ray_pose", True)),
                ref_view_strategy=params.get("ref_view_strategy", "saddle_balanced"),
                process_res=int(params.get("process_res", 504)),
                process_res_method=params.get("process_res_method", "upper_bound_resize"),
            )
        if params.get("require_metric", True) and not prediction.is_metric:
            raise SceneError("INVALID_MODEL_OUTPUT", f"{model_id} does not predict metric depth")
        depth = np.asarray(prediction.depth, dtype=np.float32)
        confidence = np.asarray(prediction.conf, dtype=np.float32)
        extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
        intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
        rgb = np.asarray(prediction.processed_images)
        count = len(files)
        if (
            depth.ndim != 3
            or len(depth) != count
            or confidence.shape != depth.shape
            or rgb.shape != (*depth.shape, 3)
            or intrinsics.shape != (count, 3, 3)
            or extrinsics.shape not in ((count, 3, 4), (count, 4, 4))
            or not np.isfinite(extrinsics).all()
            or not np.isfinite(intrinsics).all()
        ):
            raise SceneError(
                "INVALID_MODEL_OUTPUT", "Depth Anything 3 output violates Frame shapes"
            )
        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
        if not valid.any():
            raise SceneError("INVALID_MODEL_OUTPUT", "Depth Anything 3 returned no valid depth")
        # Drop the least confident pixels; the point cloud applies its own stricter threshold.
        percentile = float(params.get("confidence_percentile", 10))
        valid &= confidence >= np.percentile(confidence[valid], percentile)
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        for index, path in enumerate(files):
            world_to_camera = np.eye(4)
            world_to_camera[:3] = extrinsics[index, :3]
            Image.fromarray(rgb[index]).save(workspace.dir("recon/rgb") / f"{path.stem}.png")
            np.savez_compressed(
                workspace.dir("recon/frames") / f"{path.stem}.npz",
                depth=np.where(valid[index], depth[index], 0).astype(np.float32),
                conf=np.where(valid[index], confidence[index], 0).astype(np.float32),
                pose=np.linalg.inv(world_to_camera),
                K=intrinsics[index],
            )
        self.progress(1, 1, "batch")
        scale = prediction.scale_factor
        return ReconManifest(
            num_frames=count,
            frame_ids=[path.stem for path in files],
            metric_scale=float(scale) if scale is not None else 1.0,
            image_size=(depth.shape[2], depth.shape[1]),
        )


def _rotation_to_up(up):
    up = np.asarray(up, dtype=float)
    up /= np.linalg.norm(up)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(up, target)
    cosine = float(up @ target)
    if cosine < -1 + 1e-8:
        return np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def _world_points(frame, stride):
    depth = frame["depth"][::stride, ::stride]
    confidence = frame["conf"][::stride, ::stride]
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]] * stride
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
    pixels = np.column_stack((xx[valid], yy[valid], np.ones(int(valid.sum()))))
    camera = (pixels @ np.linalg.inv(frame["K"]).T) * depth[valid, None]
    return camera @ frame["pose"][:3, :3].T + frame["pose"][:3, 3]


class GeoCalibAlignment(StageAlgorithm):
    name = "align.geocalib_ransac"
    stage = "align"
    manifest_type = AlignManifest
    reads = ["recon"]
    writes = ["align"]

    def setup(self):
        _cache()
        self.model = None

    def download_weights(self):
        _cache()
        import torch

        destination = Path(torch.hub.get_dir()) / "geocalib" / "pinhole.tar"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            torch.hub.download_url_to_file(
                "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
                str(destination),
            )

    def run(self, workspace, params):
        _cuda()
        from geocalib import GeoCalib

        if self.model is None:
            self.model = GeoCalib().to("cuda").eval()
        files = sorted(workspace.path("recon/frames").glob("*.npz"))
        if not files:
            raise SceneError("ALIGNMENT_FAILED", "No reconstructed Frames")
        count = max(1, int(params.get("num_calibration_frames", 20)))
        selected = set(np.linspace(0, len(files) - 1, min(count, len(files)), dtype=int))
        vectors, clouds = [], []
        for index, path in enumerate(files):
            self.progress(index, len(files), "frame")
            with np.load(path, allow_pickle=False) as data:
                frame = dict(data)
            clouds.append(_world_points(frame, max(1, int(params.get("point_stride", 8)))))
            if index in selected:
                image = self.model.load_image(workspace.path(f"recon/rgb/{path.stem}.png"))
                result = self.model.calibrate(image.to("cuda"))
                # GeoCalib's zero-roll/pitch vector is (0,-1,0): up in OpenCV camera axes.
                camera_up = _numpy(result["gravity"].vec3d).reshape(-1, 3)[0]
                vectors.append(frame["pose"][:3, :3] @ camera_up)
        self.progress(len(files), len(files), "frame")
        vectors = np.asarray(vectors)
        if not np.isfinite(vectors).all():
            raise SceneError("ALIGNMENT_FAILED", "Invalid gravity estimate")
        up = vectors.mean(axis=0)
        if np.linalg.norm(up) < 1e-6:
            raise SceneError("ALIGNMENT_FAILED", "Gravity estimates disagree")
        up /= np.linalg.norm(up)
        for _ in range(20):
            weights = 1 / np.maximum(np.arccos(np.clip(vectors @ up, -1, 1)), 1e-4)
            estimate = np.sum(vectors * weights[:, None], axis=0)
            up = estimate / np.linalg.norm(estimate)
        cloud = np.concatenate(clouds)
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if len(cloud) < 100:
            raise SceneError("ALIGNMENT_FAILED", "Too few valid depth points for floor fitting")
        if len(cloud) > 30000:
            cloud = cloud[np.random.default_rng(42).choice(len(cloud), 30000, replace=False)]
        threshold = float(params.get("ransac_threshold_m", 0.02))
        remaining, horizontal, vertical = cloud.copy(), [], []
        minimum = max(30, int(0.02 * len(cloud)))
        for _ in range(12):
            if len(remaining) < minimum:
                break
            normal, _, inliers = fit_plane(remaining, threshold=threshold)
            if inliers.sum() < minimum:
                break
            points = remaining[inliers]
            agreement = float(normal @ up)
            if abs(agreement) >= np.cos(np.deg2rad(15)):
                if agreement < 0:
                    normal = -normal
                horizontal.append((float(np.median(points @ up)), normal, points))
            elif abs(agreement) <= np.sin(np.deg2rad(15)):
                vertical.append((normal, len(points)))
            remaining = remaining[~inliers]
        if not horizontal:
            raise SceneError("FLOOR_NOT_FOUND", "No large horizontal floor plane found")
        floor = min(horizontal, key=lambda plane: plane[0])
        rotation = _rotation_to_up(floor[1])
        if not vertical:
            raise SceneError("WALLS_NOT_FOUND", "No vertical planes for Manhattan alignment")
        normals = np.asarray([rotation @ normal for normal, _ in vertical])
        angles = np.arctan2(normals[:, 1], normals[:, 0])
        votes = np.asarray([weight for _, weight in vertical])
        yaw = np.arctan2(np.sum(votes * np.sin(4 * angles)), np.sum(votes * np.cos(4 * angles))) / 4
        c, s = np.cos(-yaw), np.sin(-yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rotation
        floor_z = float(np.median((floor[2] @ rotation.T)[:, 2]))
        transform = np.eye(4)
        transform[:3, :3], transform[2, 3] = rotation, -floor_z
        ceilings = [
            float(np.median((plane[2] @ rotation.T)[:, 2])) - floor_z
            for plane in horizontal
            if plane[0] > floor[0] + 1.5
        ]
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                frame = dict(data)
            frame["pose"] = transform @ frame["pose"]
            np.savez_compressed(workspace.dir("align/frames") / path.name, **frame)
        return AlignManifest(
            world_to_aligned=transform.tolist(),
            floor_z=0,
            ceiling_height=max(ceilings) if ceilings else None,
        )


def select_masks(candidates, containment_threshold, iou_threshold=0.8):
    """Keep one Frame's `(score, label, mask)` candidates from the highest score down.

    A candidate is dropped when its IoU with any kept mask exceeds `iou_threshold`, or when more
    than `containment_threshold` of its pixels lie inside a kept mask with the same label (a seat
    inside its whole chair). A contained mask with a different label, such as a door handle on a
    wardrobe, is kept.
    """
    kept = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        _, label, mask = candidate
        area = np.count_nonzero(mask)
        duplicate = False
        for _, kept_label, kept_mask in kept:
            shared = np.count_nonzero(mask & kept_mask)
            if shared / np.count_nonzero(mask | kept_mask) > iou_threshold or (
                label == kept_label and shared / area > containment_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


class Sam3Segmentation(StageAlgorithm):
    name = "segment.sam3"
    stage = "segment"
    manifest_type = SegmentManifest
    reads = ["recon"]
    writes = ["segment"]

    def setup(self):
        _cache()
        self.predictor = None
        self.half = None

    def download_weights(self):
        self._weights()

    def _weights(self):
        _cache()
        from huggingface_hub import hf_hub_download

        return hf_hub_download("facebook/sam3", "sam3.pt")

    def run(self, workspace, params):
        _cuda()
        threshold = float(params.get("score_threshold", 0.5))
        half = bool(params.get("half", True))
        batch_size = int(params.get("prompt_batch_size", 32))
        if batch_size < 1:
            raise SceneError("INVALID_PARAMETERS", "prompt_batch_size must be positive")
        containment = float(params.get("containment_threshold", 0.8))
        if not 0 < containment <= 1:
            raise SceneError("INVALID_PARAMETERS", "containment_threshold must be in (0, 1]")
        if self.predictor is None or self.half != half:
            from ultralytics.models.sam import SAM3SemanticPredictor

            self.predictor = SAM3SemanticPredictor(
                overrides={
                    "task": "segment",
                    "mode": "predict",
                    "model": self._weights(),
                    "quantize": 16 if half else 32,
                    "save": False,
                    "verbose": False,
                }
            )
            self.half = half
        self.predictor.args.conf = threshold
        vocabulary_root = Path(os.environ.get("VOCABULARY_DIR", "backend/config/vocabularies"))
        vocabulary_path = vocabulary_root / params.get("vocabulary_file", "indoor-en.txt")
        if not vocabulary_path.resolve().is_relative_to(vocabulary_root.resolve()):
            raise SceneError("INVALID_VOCABULARY", "Vocabulary must be inside VOCABULARY_DIR")
        vocabulary = list(
            dict.fromkeys(
                line.strip()
                for line in vocabulary_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            )
        )
        if not vocabulary:
            raise SceneError("INVALID_VOCABULARY", "Vocabulary is empty")
        files = sorted(workspace.path("recon/rgb").glob("*.png"))
        stride = int(params.get("frame_stride", 2))
        if stride < 1:
            raise SceneError("INVALID_PARAMETERS", "frame_stride must be positive")
        total = 0
        selected = files[::stride]
        for done, path in enumerate(selected):
            self.progress(done, len(selected), "frame")
            candidates = []
            with Image.open(path) as source:
                width, height = source.size
            # One image encoding per Frame; the Vocabulary is prompted in batches to bound VRAM.
            self.predictor.set_image(str(path))
            for offset in range(0, len(vocabulary), batch_size):
                labels = vocabulary[offset : offset + batch_size]
                masks, boxes = self.predictor.inference_features(
                    self.predictor.features, (height, width), text=labels
                )
                if masks is None:
                    continue
                # Thresholded on the GPU: a float copy of every full-resolution mask in host
                # memory is four times the size and can exhaust it within one Frame.
                masks, boxes = (masks > 0.5).cpu().numpy(), _numpy(boxes)
                if masks.shape != (len(boxes), height, width):
                    raise SceneError(
                        "INVALID_MODEL_OUTPUT", "SAM 3 mask dimensions differ from RGB"
                    )
                for binary, score, label in zip(
                    masks, boxes[:, 4], boxes[:, 5].astype(int), strict=True
                ):
                    if np.isfinite(score) and score >= threshold and binary.any():
                        candidates.append((float(score), labels[label], binary))
            kept = select_masks(candidates, containment)
            masks = (
                np.stack([item[2] for item in kept])
                if kept
                else np.empty((0, height, width), dtype=bool)
            )
            np.savez_compressed(
                workspace.dir("segment/masks") / f"{path.stem}.npz",
                masks=masks,
                labels=np.asarray([item[1] for item in kept], dtype=str),
                scores=np.asarray([item[0] for item in kept], dtype=np.float32),
            )
            total += len(kept)
        self.progress(len(selected), len(selected), "frame")
        return SegmentManifest(num_masks=total, vocabulary_size=len(vocabulary))
