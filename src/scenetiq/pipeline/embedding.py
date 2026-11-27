import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

from scenetiq.config import EmbeddingOptions
from scenetiq.domain import Camera, FloatArray
from scenetiq.geometry import l2_normalize
from scenetiq.pipeline.da3_adapters import c2w_to_w2c
from scenetiq.pipeline.view_data import ViewData


class ImageEncoder(Protocol):
    def encode_images(self, images: list) -> FloatArray: ...


MASK_FILL = (127, 127, 127)


def visible_pixels(
    points: FloatArray, camera: Camera, depth: FloatArray, rtol: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    u, v, _ = _visible(points, camera, depth, rtol)
    return u, v


def _visible(
    points: FloatArray, camera: Camera, depth: FloatArray, rtol: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], FloatArray]:
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
    return u[ok], v[ok], z[ok]


def splat_radius(
    z: FloatArray, focal: float, spacing: float, max_radius: int
) -> npt.NDArray[np.int64]:
    radius = np.ceil(spacing * focal / (2.0 * np.asarray(z, dtype=np.float64)))
    return np.clip(radius, 1, max_radius).astype(np.int64)


def instance_mask(
    u: npt.NDArray[np.int64],
    v: npt.NDArray[np.int64],
    radius: npt.NDArray[np.int64],
    width: int,
    height: int,
) -> npt.NDArray[np.bool_]:
    left, right = np.clip(u - radius, 0, width), np.clip(u + radius + 1, 0, width)
    top, bottom = np.clip(v - radius, 0, height), np.clip(v + radius + 1, 0, height)
    # Mark each square's corners, then sum along both axes: the count of squares per pixel.
    corners = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.add.at(corners, (top, left), 1)
    np.add.at(corners, (top, right), -1)
    np.add.at(corners, (bottom, left), -1)
    np.add.at(corners, (bottom, right), 1)
    return corners.cumsum(axis=0).cumsum(axis=1)[:height, :width] > 0


def select_views(visible_counts: Sequence[int], top_k: int, min_pixels: int) -> list[int]:
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
    view_index: int
    u: npt.NDArray[np.int64]
    v: npt.NDArray[np.int64]
    radius: npt.NDArray[np.int64] | None = None


ViewPlans = dict[int, list[PlannedView]]


def plan_views(
    points: FloatArray,
    instance_ids: npt.NDArray[np.int32],
    view_data: ViewData,
    options: EmbeddingOptions | None = None,
    point_spacing: float | None = None,
) -> ViewPlans:
    opts = options or EmbeddingOptions()
    plans: ViewPlans = {}
    for iid in (int(i) for i in np.unique(instance_ids) if i >= 0):
        member = points[instance_ids == iid]
        per_view = [
            _visible(member, cam, view_data.depth[i], opts.visibility_rtol)
            for i, cam in enumerate(view_data.cameras)
        ]
        chosen = select_views(
            [len(u) for u, _, _ in per_view], opts.views_per_instance, opts.min_visible_pixels
        )
        plan = []
        for vi in chosen:
            u, v, z = per_view[vi]
            focal = float(view_data.cameras[vi].intrinsics[0, 0])
            radius = (
                None
                if point_spacing is None
                else splat_radius(z, focal, point_spacing, opts.max_splat_radius)
            )
            plan.append(PlannedView(vi, u, v, radius))
        plans[iid] = plan
    return plans


class PhotoPixels(NamedTuple):
    instance_id: int
    rank: int
    photo: Any
    u: npt.NDArray[np.int64]
    v: npt.NDArray[np.int64]
    mask: Any = None


def photo_pixels(
    plans: ViewPlans, image_paths: Sequence[Path], cameras: Sequence[Camera]
) -> Iterator[PhotoPixels]:
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
                    mask = None
                    if p.radius is not None:
                        drawn = instance_mask(p.u, p.v, p.radius, cam.width, cam.height)
                        mask = Image.fromarray(drawn.astype(np.uint8) * 255)
                        if mask.size != photo.size:
                            mask = mask.resize(photo.size, Image.Resampling.BILINEAR)
                    yield PhotoPixels(iid, rank, photo, pu, pv, mask)


def embed_instances(
    plans: ViewPlans,
    view_data: ViewData,
    image_paths: Sequence[Path],
    encoder: ImageEncoder,
    options: EmbeddingOptions | None = None,
) -> dict[int, FloatArray]:
    opts = options or EmbeddingOptions()

    from PIL import Image

    crops: list = []
    owners: list[int] = []
    masked: list[bool] = []
    for seen in photo_pixels(plans, image_paths, view_data.cameras):
        photo = seen.photo
        for expand in opts.expansions:
            box = crop_box(seen.u, seen.v, expand, photo.width, photo.height, min_size=16)
            crops.append(photo.crop(box))
            owners.append(seen.instance_id)
            masked.append(False)
        if seen.mask is not None and opts.masked_weight > 0 and opts.expansions:
            tight = min(opts.expansions)
            box = crop_box(seen.u, seen.v, tight, photo.width, photo.height, min_size=16)
            crop = photo.crop(box)
            fill = Image.new("RGB", crop.size, MASK_FILL)
            crops.append(Image.composite(crop, fill, seen.mask.crop(box)))
            owners.append(seen.instance_id)
            masked.append(True)

    if not crops:
        return {}
    feats = encoder.encode_images(crops)
    owner_arr = np.asarray(owners)
    masked_arr = np.asarray(masked)
    embeddings: dict[int, FloatArray] = {}
    for iid in plans:
        own = owner_arr == iid
        if not np.any(own):
            continue
        embedding = l2_normalize(feats[own & ~masked_arr].mean(axis=0))
        if np.any(own & masked_arr):
            inside = l2_normalize(feats[own & masked_arr].mean(axis=0))
            weight = opts.masked_weight
            embedding = l2_normalize(weight * inside + (1 - weight) * embedding)
        embeddings[iid] = embedding
    return embeddings
