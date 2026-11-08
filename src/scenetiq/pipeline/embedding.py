"""Embedding stage: one semantic embedding per Instance from multi-view crops.

For every Instance, pick the Views that see most of its points, crop it from
the original full-resolution photos at two context scales, encode the crops
with SigLIP2, and average them into one L2-normalised embedding.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

from scenetiq.config import EmbeddingOptions
from scenetiq.domain import Camera, FloatArray
from scenetiq.geometry import l2_normalize
from scenetiq.pipeline.adapters import c2w_to_w2c
from scenetiq.pipeline.view_data import ViewData


class ImageEncoder(Protocol):
    """Encodes image crops into the space Query text is encoded into."""

    def encode_images(self, images: list) -> FloatArray:
        """Return ``(len(images), D)`` L2-normalised embeddings."""
        ...


def visible_pixels(
    points: FloatArray, camera: Camera, depth: FloatArray, rtol: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Project ``points`` into ``camera`` and keep the unoccluded ones.

    A point counts as visible when it lies in front of the camera, inside the
    image, on a pixel with known depth, and within ``rtol`` of that depth.

    Returns:
        ``(u, v)`` integer pixel coordinates (at the depth-map resolution).
    """
    w2c = c2w_to_w2c(camera.cam_to_world)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    front = z > 1e-6
    k = camera.intrinsics
    safe_z = np.where(front, z, 1.0)
    u = np.round(k[0, 0] * cam[:, 0] / safe_z + k[0, 2]).astype(np.int64)
    v = np.round(k[1, 1] * cam[:, 1] / safe_z + k[1, 2]).astype(np.int64)
    inside = front & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    u, v, z = u[inside], v[inside], z[inside]
    seen = depth[v, u]
    ok = (seen > 0) & (np.abs(z - seen) <= rtol * seen)
    return u[ok], v[ok]


def select_views(visible_counts: Sequence[int], top_k: int, min_pixels: int) -> list[int]:
    """Indices of the ``top_k`` Views seeing the most points (at least ``min_pixels``)."""
    counts = np.asarray(visible_counts)
    order = np.argsort(-counts, kind="stable")
    return [int(i) for i in order[:top_k] if counts[i] >= min_pixels]


def crop_box(
    u: npt.NDArray[np.int64],
    v: npt.NDArray[np.int64],
    expand: float,
    width: int,
    height: int,
    min_size: int = 1,
) -> tuple[int, int, int, int]:
    """Bounding box ``(left, top, right, bottom)`` of the pixels, grown and clipped.

    Each side grows by ``expand`` times the box size; ``right``/``bottom`` are
    exclusive. The result is at least ``min_size`` pixels per side when the
    image allows it.
    """
    lo_u, hi_u = int(u.min()), int(u.max()) + 1
    lo_v, hi_v = int(v.min()), int(v.max()) + 1
    grow_u = max(expand * (hi_u - lo_u), (min_size - (hi_u - lo_u)) / 2)
    grow_v = max(expand * (hi_v - lo_v), (min_size - (hi_v - lo_v)) / 2)
    left = max(0, math.floor(lo_u - grow_u))
    right = min(width, math.ceil(hi_u + grow_u))
    top = max(0, math.floor(lo_v - grow_v))
    bottom = min(height, math.ceil(hi_v + grow_v))
    return left, top, right, bottom


class PlannedView(NamedTuple):
    """One View chosen for an Instance, with the Instance's visible pixels in it.

    Attributes:
        view_index: Index of the View in the Capture.
        u: Column of every visible point, at the depth-map resolution.
        v: Row of every visible point, at the depth-map resolution.
    """

    view_index: int
    u: npt.NDArray[np.int64]
    v: npt.NDArray[np.int64]


ViewPlans = dict[int, list[PlannedView]]
"""Instance id -> the Views that see it best, best first (empty if none sees it enough)."""


def plan_views(
    points: FloatArray,
    instance_ids: npt.NDArray[np.int32],
    view_data: ViewData,
    options: EmbeddingOptions | None = None,
) -> ViewPlans:
    """Choose, for every Instance, the Views that see most of its points.

    Args:
        points: ``(N, 3)`` Reconstruction points.
        instance_ids: ``(N,)`` Instance id per point, -1 for none.
        view_data: Cameras and depth at the depth-map resolution.
        options: ``views_per_instance``, ``min_visible_pixels`` and ``visibility_rtol``.

    Returns:
        A plan for every Instance id, empty when no View sees at least
        ``min_visible_pixels`` of its points.
    """
    opts = options or EmbeddingOptions()
    plans: ViewPlans = {}
    for iid in (int(i) for i in np.unique(instance_ids) if i >= 0):
        member = points[instance_ids == iid]
        per_view = [
            visible_pixels(member, cam, view_data.depth[i], opts.visibility_rtol)
            for i, cam in enumerate(view_data.cameras)
        ]
        chosen = select_views(
            [len(u) for u, _ in per_view], opts.views_per_instance, opts.min_visible_pixels
        )
        plans[iid] = [PlannedView(vi, *per_view[vi]) for vi in chosen]
    return plans


class PhotoPixels(NamedTuple):
    """One planned View of an Instance, with its pixels in the original photo.

    Attributes:
        instance_id: The Instance.
        rank: Position of the View in the Instance's plan, 0 for the best.
        photo: The View's original photo as a PIL RGB image.
        u: Column of every visible point, at the photo's resolution.
        v: Row of every visible point, at the photo's resolution.
    """

    instance_id: int
    rank: int
    photo: Any
    u: npt.NDArray[np.int64]
    v: npt.NDArray[np.int64]


def photo_pixels(
    plans: ViewPlans, image_paths: Sequence[Path], cameras: Sequence[Camera]
) -> Iterator[PhotoPixels]:
    """Yield every planned View of every Instance with its pixels scaled to the photo.

    Each photo is opened once, in View order, so the stages that crop Instances from
    full-resolution photos never hold more than one in memory.

    Args:
        plans: The Views chosen for every Instance, with pixels at the depth-map resolution.
        image_paths: Original photo of every View, in View order.
        cameras: Camera of every View; its size is the depth-map resolution.
    """
    from PIL import Image

    for vi in sorted({p.view_index for plan in plans.values() for p in plan}):
        cam = cameras[vi]
        with Image.open(image_paths[vi]) as img:
            photo = img.convert("RGB")
        sx, sy = photo.width / cam.width, photo.height / cam.height
        for iid, plan in plans.items():
            for rank, p in enumerate(plan):
                if p.view_index == vi:
                    pu = (p.u * sx).astype(np.int64)
                    pv = (p.v * sy).astype(np.int64)
                    yield PhotoPixels(iid, rank, photo, pu, pv)


def embed_instances(
    plans: ViewPlans,
    view_data: ViewData,
    image_paths: Sequence[Path],
    encoder: ImageEncoder,
    options: EmbeddingOptions | None = None,
) -> dict[int, FloatArray]:
    """Compute an embedding for every Instance that some View sees well enough.

    Instances with an empty plan get no embedding; the caller drops them, so
    every remaining Instance can be queried.

    Args:
        plans: The Views chosen for every Instance (:func:`plan_views`).
        view_data: Cameras at the depth-map resolution.
        image_paths: Original photo of every View, in View order.
        encoder: Image tower of the embedding model.
        options: Stage settings.

    Returns:
        Instance id -> L2-normalised embedding.
    """
    opts = options or EmbeddingOptions()

    crops: list = []
    owners: list[int] = []
    for seen in photo_pixels(plans, image_paths, view_data.cameras):
        photo = seen.photo
        for expand in opts.expansions:
            box = crop_box(seen.u, seen.v, expand, photo.width, photo.height, min_size=16)
            crops.append(photo.crop(box))
            owners.append(seen.instance_id)

    if not crops:
        return {}
    feats = encoder.encode_images(crops)
    owner_arr = np.asarray(owners)
    return {
        iid: l2_normalize(feats[owner_arr == iid].mean(axis=0))
        for iid in plans
        if np.any(owner_arr == iid)
    }
