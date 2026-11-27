"""Aligning stage: Gravity alignment of the fused geometry and the cameras (ADR 0007).

Physical up becomes -Y. Weak evidence leaves the Scene as it is and the job goes on; a
technical failure propagates and fails the job like any other stage's.
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from scenetiq.config import AlignmentOptions
from scenetiq.domain import FloatArray
from scenetiq.pipeline.gpu_memory import free_gpu_memory
from scenetiq.pipeline.view_data import FusedGeometry, ViewData

Float64Array = npt.NDArray[np.float64]
"""Rotations and directions, which are computed in double precision."""

CONVENTIONAL_UP = (0.0, -1.0, 0.0)
"""The direction physical up has in a Reconstruction once it is Gravity-aligned."""

ALIGNMENT_LOGGER = "scenetiq.alignment"
"""Logger of the per-job line; ``colab_logging.json`` sends it to stdout unformatted."""

_logger = logging.getLogger(ALIGNMENT_LOGGER)


class UpEstimator(Protocol):
    """Estimates which way is up in single photos."""

    def estimate(
        self, images: npt.NDArray[np.uint8], focals: npt.NDArray[np.float32] | None
    ) -> npt.ArrayLike:
        """Return one up vector per View, each in its camera's frame (OpenCV convention).

        Args:
            images: ``(K, H, W, 3)`` uint8 RGB.
            focals: ``(K,)`` focal lengths in pixels of ``images``, if known.
        """
        ...


class GeoCalibUpEstimator:
    """GeoCalib behind the :class:`UpEstimator` interface.

    The model loads for each :meth:`estimate` call and leaves the GPU when it ends.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Configure the estimator; nothing is loaded until :meth:`estimate`."""
        self.device = device

    def estimate(
        self, images: npt.NDArray[np.uint8], focals: npt.NDArray[np.float32] | None
    ) -> npt.NDArray[np.float32]:
        """Run GeoCalib on each image independently; see :class:`UpEstimator`."""
        try:
            return self._estimate(images, focals)
        finally:
            free_gpu_memory()

    def _estimate(self, images: Any, focals: Any) -> npt.NDArray[np.float32]:
        import torch
        from geocalib import GeoCalib

        model = GeoCalib().to(self.device)
        ups: list[np.ndarray] = []
        for i, image in enumerate(images):
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            priors = None
            if focals is not None:
                priors = {"focal": torch.tensor(float(focals[i]), device=self.device)}
            result = model.calibrate(tensor.to(self.device), priors=priors)
            # Despite the name, this vector is (0, -1, 0) for a level camera: it points up.
            ups.append(result["gravity"].vec3d.reshape(3).cpu().numpy())
        return np.stack(ups).astype(np.float32)


@dataclass(frozen=True)
class PlaneCandidate:
    """A plane found in the fused geometry.

    Attributes:
        normal: Unit normal of the plane.
        fraction: Share of the searched points that lie on the plane.
    """

    normal: Float64Array
    fraction: float


PlaneSearch = Callable[[FloatArray, AlignmentOptions], list[PlaneCandidate]]
"""Finds candidate planes in ``(N, 3)`` points."""


@dataclass(frozen=True)
class UpConsensus:
    """What the per-View up estimates agree on.

    Attributes:
        direction: Unit up direction in the world frame, or ``None`` without a consensus.
        fraction: Share of the estimates in the largest group that agrees.
    """

    direction: Float64Array | None
    fraction: float


def find_up_consensus(
    world_ups: npt.ArrayLike, max_angle_deg: float, min_fraction: float
) -> UpConsensus:
    """Find the direction most per-View up estimates agree on.

    The group around each estimate holds every estimate within ``max_angle_deg`` of it,
    itself included. The largest group wins, with ties going to the earlier View.

    Args:
        world_ups: ``(K, 3)`` up estimates in the world frame, in View order.
        max_angle_deg: Largest angle between an estimate and its group's centre.
        min_fraction: The group must hold more than this share of the estimates.

    Returns:
        The group's normalised mean as the direction, if the group is large enough.
    """
    ups = np.asarray(world_ups, dtype=np.float64)
    if len(ups) == 0:
        return UpConsensus(direction=None, fraction=0.0)
    units = ups / np.linalg.norm(ups, axis=1, keepdims=True)
    within = units @ units.T >= np.cos(np.deg2rad(max_angle_deg))
    counts = within.sum(axis=1)
    centre = int(np.argmax(counts))
    fraction = float(counts[centre]) / len(units)
    if fraction <= min_fraction:
        return UpConsensus(direction=None, fraction=fraction)
    return UpConsensus(direction=_unit(units[within[centre]].mean(axis=0)), fraction=fraction)


def apply_rotation(
    geometry: FusedGeometry, view_data: ViewData, rotation: npt.ArrayLike
) -> tuple[FusedGeometry, ViewData]:
    """Rotate the points and every camera about the origin, keeping the point order.

    Depth is measured in each camera's frame, so it stays valid and is not touched.

    Args:
        geometry: The fused geometry.
        view_data: The Views whose cameras move with the geometry.
        rotation: 3x3 rotation.

    Returns:
        New geometry and View data; the inputs are unchanged.
    """
    rot = np.asarray(rotation, dtype=np.float64)
    vertices = (geometry.vertices.astype(np.float64) @ rot.T).astype(np.float32)
    transform = np.eye(4)
    transform[:3, :3] = rot
    cameras = [
        replace(cam, cam_to_world=(transform @ cam.cam_to_world).astype(np.float32))
        for cam in view_data.cameras
    ]
    return replace(geometry, vertices=vertices), replace(view_data, cameras=cameras)


def evenly_spaced_views(view_count: int, max_views: int) -> list[int]:
    """Return at most ``max_views`` View indices spread evenly from first to last."""
    if view_count <= max_views:
        return list(range(view_count))
    return [int(i) for i in np.linspace(0, view_count - 1, max_views).round()]


def find_planes(points: FloatArray, options: AlignmentOptions) -> list[PlaneCandidate]:
    """Find the largest planes in ``points`` with RANSAC, removing each one's points in turn.

    Works on a copy, subsampled to ``options.plane_max_points``, so ``points`` is not changed.

    Args:
        points: ``(N, 3)`` positions in metres.
        options: Stage settings.

    Returns:
        The planes found, in the order they were found. Normals have an arbitrary sign.
    """
    import open3d as o3d

    remaining = np.asarray(points, dtype=np.float64)
    if len(remaining) > options.plane_max_points:
        rng = np.random.default_rng(0)
        remaining = rng.choice(remaining, options.plane_max_points, replace=False, axis=0)
    total = len(remaining)
    # A smaller remainder could not hold a plane that qualifies.
    smallest = max(3, math.ceil(options.min_plane_fraction * total))
    candidates: list[PlaneCandidate] = []
    for _ in range(options.max_planes):
        if len(remaining) < smallest:
            break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining))
        plane, inliers = cloud.segment_plane(
            distance_threshold=options.plane_distance,
            ransac_n=3,
            num_iterations=options.plane_iterations,
        )
        candidates.append(
            PlaneCandidate(normal=_unit(np.asarray(plane)[:3]), fraction=len(inliers) / total)
        )
        remaining = np.delete(remaining, inliers, axis=0)
    return candidates


def select_plane(
    candidates: Sequence[PlaneCandidate],
    up: npt.ArrayLike,
    max_angle_deg: float,
    min_fraction: float,
) -> PlaneCandidate | None:
    """Pick the largest plane level enough to refine ``up``.

    Floor, tabletop and ceiling all qualify: only the plane's direction matters.

    Args:
        candidates: The planes to choose from.
        up: Current estimate of up.
        max_angle_deg: Largest angle between a normal and ``up``, ignoring the normal's sign.
        min_fraction: Smallest share of the points a plane may hold.

    Returns:
        The chosen plane with its normal flipped to agree with ``up``, or ``None``.
    """
    up_unit = _unit(up)
    cos_limit = np.cos(np.deg2rad(max_angle_deg))
    best: PlaneCandidate | None = None
    for candidate in candidates:
        normal = _unit(candidate.normal)
        cos_angle = float(normal @ up_unit)
        if candidate.fraction < min_fraction or abs(cos_angle) < cos_limit:
            continue
        if best is None or candidate.fraction > best.fraction:
            best = PlaneCandidate(
                normal=normal if cos_angle > 0 else -normal, fraction=candidate.fraction
            )
    return best


def format_alignment_line(
    method: str,
    angle_deg: float,
    views: int,
    consensus: float,
    plane_fraction: float | None = None,
) -> str:
    """Return the log line of one job's Aligning stage."""
    line = (
        f"gravity-alignment method={method} angle={angle_deg:.1f}deg "
        f"views={views} consensus={consensus:.2f}"
    )
    return line if plane_fraction is None else f"{line} plane={plane_fraction:.2f}"


def gravity_align(
    geometry: FusedGeometry,
    view_data: ViewData,
    estimator: UpEstimator,
    options: AlignmentOptions | None = None,
    plane_search: PlaneSearch = find_planes,
) -> tuple[FusedGeometry, ViewData, bool]:
    """Rotate the geometry and cameras so that physical up is -Y, if the Views agree on up.

    A qualifying plane refines the direction the Views agree on, but never replaces a
    missing consensus. Logs one line saying what was done. The point order never changes.

    Args:
        geometry: The fused geometry.
        view_data: Every View, with its camera.
        estimator: The source of per-View up vectors.
        options: Stage settings.
        plane_search: The source of candidate planes.

    Returns:
        The aligned geometry and View data, or the inputs when the Views do not agree, and
        whether the Scene was aligned.
    """
    opts = options or AlignmentOptions()
    chosen = evenly_spaced_views(len(view_data), opts.max_views)
    cameras = [view_data.cameras[i] for i in chosen]
    focals = None
    if opts.use_focal_prior:
        focals = np.array(
            [(cam.intrinsics[0, 0] + cam.intrinsics[1, 1]) / 2 for cam in cameras],
            dtype=np.float32,
        )
    ups = np.asarray(estimator.estimate(view_data.images[chosen], focals), dtype=np.float64)
    world_ups = np.stack(
        [
            cam.cam_to_world[:3, :3].astype(np.float64) @ up
            for cam, up in zip(cameras, ups, strict=True)
        ]
    )
    consensus = find_up_consensus(world_ups, opts.consensus_angle_deg, opts.consensus_fraction)
    if consensus.direction is None:
        _logger.info("%s", format_alignment_line("none", 0.0, len(chosen), consensus.fraction))
        return geometry, view_data, False
    plane = select_plane(
        plane_search(geometry.vertices, opts),
        consensus.direction,
        opts.max_plane_angle_deg,
        opts.min_plane_fraction,
    )
    up = consensus.direction if plane is None else plane.normal
    cos_angle = float(np.clip(up @ np.array(CONVENTIONAL_UP), -1.0, 1.0))
    line = format_alignment_line(
        "geocalib" if plane is None else "plane",
        float(np.rad2deg(np.arccos(cos_angle))),
        len(chosen),
        consensus.fraction,
        None if plane is None else plane.fraction,
    )
    _logger.info("%s", line)
    return (*apply_rotation(geometry, view_data, rotation_between(up, CONVENTIONAL_UP)), True)


def rotation_between(source: npt.ArrayLike, target: npt.ArrayLike) -> Float64Array:
    """Return the smallest 3x3 rotation taking direction ``source`` to direction ``target``.

    Opposite directions have no unique smallest rotation, so any half turn about an
    axis perpendicular to them is returned.
    """
    src = _unit(source)
    dst = _unit(target)
    cos_angle = float(src @ dst)
    if cos_angle < -1.0 + 1e-9:
        helper = np.eye(3)[int(np.argmin(np.abs(src)))]
        axis = _unit(np.cross(src, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    x, y, z = np.cross(src, dst)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + skew + skew @ skew / (1.0 + cos_angle)


def _unit(vector: npt.ArrayLike) -> Float64Array:
    arr = np.asarray(vector, dtype=np.float64)
    return arr / np.linalg.norm(arr)
