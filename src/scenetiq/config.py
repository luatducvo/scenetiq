"""Every setting of the backend and its defaults, in one place.

Overrides come from one JSON object in ``SCENETIQ_CONFIG``, which the notebook fills from its
``CONFIG`` cell, so a value there wins over the default here. Only the bearer token travels
separately, in ``SCENETIQ_TOKEN``. This module uses only the standard library, because the
notebook imports it before anything is installed.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

DEFAULT_STORAGE_ROOT = Path("./data")
"""Where Scenes are stored when the config names no ``storage_root``."""


@dataclass(frozen=True)
class DA3Options:
    """Settings of the DA3 Reconstructing stage.

    Attributes:
        model_id: Hugging Face id of the DA3 weights.
        process_res: Long-side resolution DA3 processes at.
        process_res_method: How DA3 resizes photos to ``process_res``.
        ref_view_strategy: How DA3 picks its reference View.
        use_ray_pose: Derive cameras from DA3's ray head instead of its camera head.
        conf_percentile: Pixels below this confidence percentile (over the whole
            Capture) get no depth.
        max_depth: Depth beyond this many metres is dropped, so no later stage sees it.
    """

    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    ref_view_strategy: str = "saddle_balanced"
    use_ray_pose: bool = False
    conf_percentile: float = 30.0
    max_depth: float = 12.0


@dataclass(frozen=True)
class FusionOptions:
    """Settings of the TSDF Fusing stage.

    Attributes:
        voxel_size: TSDF voxel edge in metres; sets the point spacing (1-2 cm).
        sdf_trunc_voxels: Truncation distance, in voxels.
        block_resolution: Voxels along each edge of a TSDF block.
        initial_block_count: Blocks allocated up front; the grid grows beyond this as needed.
    """

    voxel_size: float = 0.02
    sdf_trunc_voxels: float = 4.0
    block_resolution: int = 16
    initial_block_count: int = 1000


@dataclass(frozen=True)
class AlignmentOptions:
    """Settings of the Aligning stage (ADR 0007).

    Attributes:
        max_views: Most Views to estimate up from, spaced evenly through the Capture.
        use_focal_prior: Give the up estimator the focal length from the Reconstructing stage.
        consensus_angle_deg: Largest angle, in degrees, between estimates that agree.
        consensus_fraction: Agreeing estimates must be more than this share of them all.
        plane_distance: RANSAC inlier distance in metres.
        max_planes: Most planes to look for.
        min_plane_fraction: A plane must hold at least this share of the searched points.
        max_plane_angle_deg: Largest angle, in degrees, between a plane's normal and the
            direction the Views agree on.
        plane_max_points: The plane search subsamples to this many points.
        plane_iterations: RANSAC iterations per plane.
    """

    max_views: int = 12
    use_focal_prior: bool = True
    consensus_angle_deg: float = 10.0
    consensus_fraction: float = 0.5
    plane_distance: float = 0.03
    max_planes: int = 5
    min_plane_fraction: float = 0.05
    max_plane_angle_deg: float = 10.0
    plane_max_points: int = 200_000
    plane_iterations: int = 1000


@dataclass(frozen=True)
class SuperpointOptions:
    """Settings of the Superpoints stage.

    Attributes:
        primitives: ``"superpoints"`` (ScanNet Segmentator on the fused mesh) or
            ``"voxel"`` (voxel-grid fallback, for debugging).
        k_thresh: Segmentator threshold; larger gives larger superpoints.
        min_verts: Segmentator minimum superpoint size.
        primitive_voxel_size: Edge, in metres, of the voxel-grid fallback primitives.
    """

    primitives: Literal["superpoints", "voxel"] = "superpoints"
    k_thresh: float = 0.01
    min_verts: int = 20
    primitive_voxel_size: float = 0.15


@dataclass(frozen=True)
class SAM2Options:
    """Settings of the SAM2 Masking stage.

    Attributes:
        model_id: Hugging Face id of the SAM2 weights.
        points_per_side: Prompt grid density of the automatic generator.
        pred_iou_thresh: Drop Masks SAM2 itself rates below this.
        stability_score_thresh: Drop unstable Masks.
        min_area_fraction: Drop Masks smaller than this fraction of the image.
    """

    model_id: str = "facebook/sam2.1-hiera-large"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.9
    min_area_fraction: float = 0.001


@dataclass(frozen=True)
class MergingOptions:
    """Settings of the MV3DIS Merging stage.

    Attributes:
        visibility_rtol: Relative depth tolerance for a point to count as seen.
        merge_small_points: MV3DIS merges or drops regions smaller than this.
        min_instance_points: Instances with fewer points are dropped.
    """

    visibility_rtol: float = 0.05
    merge_small_points: int = 200
    min_instance_points: int = 50


@dataclass(frozen=True)
class EmbeddingOptions:
    """Settings of the Embedding stage.

    Attributes:
        views_per_instance: How many of the best Views to crop each Instance from.
        min_visible_pixels: A View must see at least this many of the Instance's points.
        expansions: Context added around the tight crop, as fractions of its size; one
            crop per value.
        visibility_rtol: Relative depth tolerance of the occlusion test.
        masked_weight: Share, in [0, 1], that the crops with everything outside the Instance
            greyed out take in its embedding; the context crops take the rest (ADR 0011).
            0 makes no masked crop. Not validated.
        max_splat_radius: Largest square, in depth-map pixels on each side of a point,
            that a point draws in its Instance's mask; reached by points close to the camera.
    """

    views_per_instance: int = 5
    min_visible_pixels: int = 30
    expansions: tuple[float, ...] = (0.1, 0.4)
    visibility_rtol: float = 0.05
    masked_weight: float = 0.5
    max_splat_radius: int = 12


@dataclass(frozen=True)
class SigLIP2Options:
    """Settings of the SigLIP2 encoder.

    Attributes:
        model_id: Hugging Face id of the SigLIP2 checkpoint (Apache-2.0).
        prompt: Template wrapping a short Query; ``{}`` is replaced by the text.
        prompt_max_words: Longest Query, in words, that is wrapped in ``prompt``; longer
            Queries reach SigLIP2 as written. 0 never wraps.
        batch_size: Crops encoded per forward pass.
    """

    model_id: str = "google/siglip2-giant-opt-patch16-384"
    prompt: str = "This is a photo of {}."
    prompt_max_words: int = 4
    batch_size: int = 64


@dataclass(frozen=True)
class CaptioningOptions:
    """Settings of the Captioning stage (ADR 0008).

    Attributes:
        model_id: Hugging Face id of the vision-language model.
        views_per_instance: How many of an Instance's best Views (from the Embedding
            stage's plan) go into its prompt.
        context_expand: Context added around the Instance's tight box, as a fraction of
            its size; the tight box is drawn on the crop.
        max_pixels: Pixel budget per image in the model's processor.
        max_new_tokens: Longest reply generated per Instance; a reply cut off here gets
            no Caption.
        background_dim: Brightness, in (0, 1], of everything outside the Instance's mask
            in its crops, so the model can tell a floor from what stands on it (ADR 0011).
            1 leaves the crops as they are. Not validated.
    """

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    views_per_instance: int = 3
    context_expand: float = 0.6
    background_dim: float = 0.35
    max_pixels: int = 448 * 448
    max_new_tokens: int = 320


@dataclass(frozen=True)
class QueryParserOptions:
    """Settings of the Query parser on the Captioning stage's vision-language model (ADR 0009).

    Attributes:
        max_new_tokens: Longest reply generated per Query; a reply cut off here is a parse
            failure. The reply ends when its JSON closes, so a generous limit costs nothing
            on a normal reply. Separate from ``CaptioningOptions.max_new_tokens``.
    """

    max_new_tokens: int = 1024


@dataclass(frozen=True)
class CaptionEmbeddingOptions:
    """Settings of the text model that embeds Descriptions and Query text (ADR 0008).

    Attributes:
        model_id: Hugging Face id of the text embedding model.
        query_instruction: Task description prefixed to Query text (not to Descriptions).
        batch_size: Descriptions encoded per forward pass.
        max_length: Longest input, in tokens.
    """

    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    query_instruction: str = (
        "Given a description of an object in a scene, retrieve the caption of that object"
    )
    batch_size: int = 32
    max_length: int = 256


@dataclass(frozen=True)
class SpatialOptions:
    """Settings of Queries with a Relation (ADR 0009). No default is validated.

    Attributes:
        candidates_per_object: M, how many Instances of each object's phrase ranking are
            shortlisted. The wanted object takes ``query_top_k`` when that is larger.
        falloff: The fraction of the Scene extent at which a distance-based Relation's
            score drops to ``1/e``; for between, the fraction of the distance between its
            two Anchors.
        caption_margin: How far a shortlisted Instance's best Description similarity may
            be below the best among the object's shortlist and still pass.
        image_margin: How far its image similarity may be below the best among the
            object's shortlist and still pass. A shortlisted Instance stays a candidate
            when it passes on either channel. Neither margin is a threshold on Matches.
    """

    candidates_per_object: int = 5
    falloff: float = 0.1
    caption_margin: float = 0.1
    image_margin: float = 0.05


@dataclass(frozen=True)
class Settings:
    """Backend settings.

    Attributes:
        storage_root: Directory holding the SQLite database and all Scene data.
        token: Shared secret every request must present as a bearer token.
        query_top_k: The most Matches a Query returns when it does not ask for a number
            (ADR 0011).
        query_max_top_k: The most Matches a Query may ask for.
        query_score_ratio: A Query without a Relation keeps an Instance as a Match only
            when its score is at least this fraction of the best Match's (ADR 0011). 0
            keeps every Instance that scores above the Scene's average. Not validated.
        negative_query_threshold: Minimum semantic label or caption similarity across the
            Scene required for a Query to be considered positive; below this, it is treated
            as a negative Query and returns 0 Matches (ADR 0012).
        pipeline: Which Pipeline to run: ``"fake"`` or ``"real"``.
        min_views: Fewest photos a Capture may upload.
        max_views: Most photos a Capture may upload.
        upload_max_side: The frontend shrinks every photo so that its long side is at most
            this many pixels before uploading it (ADR 0010).
        upload_jpeg_quality: The JPEG quality, in (0, 1], the frontend re-encodes photos at.
        view_stride: The Pipeline gets every ``view_stride``-th uploaded View, from the first.
        cors_origins: Browser origins allowed to call the API.
        da3: Reconstructing stage.
        fusion: Fusing stage.
        alignment: Aligning stage.
        superpoints: Superpoints stage.
        sam2: Masking stage.
        merging: Merging stage.
        embedding: Embedding stage.
        siglip: The encoder of Instance crops and Query text.
        captioning: Captioning stage.
        query_parser: The Query parser, which runs on the Captioning stage's model.
        caption_embedding: The encoder of Descriptions and Query text.
        spatial: Queries with a Relation.

    Raises:
        ValueError: If ``min_views`` is below 1 or above ``max_views``, ``query_max_top_k`` is
            below ``query_top_k``, ``query_score_ratio`` or
            ``embedding.masked_weight`` is outside [0, 1], ``embedding.max_splat_radius`` is
            below 1, ``captioning.background_dim`` is outside (0, 1], ``upload_max_side``
            is below 1, ``upload_jpeg_quality`` is outside (0, 1], ``view_stride``,
            ``spatial.candidates_per_object`` or ``query_parser.max_new_tokens`` is below
            1, ``spatial.falloff`` is not positive, or ``spatial.caption_margin`` or
            ``spatial.image_margin`` is negative.
    """

    storage_root: Path
    token: str
    query_top_k: int = 5
    query_max_top_k: int = 20
    query_score_ratio: float = 0.5
    negative_query_threshold: float = 0.35
    pipeline: Literal["fake", "real"] = "fake"
    min_views: int = 1
    max_views: int = 2000
    upload_max_side: int = 2048
    upload_jpeg_quality: float = 0.9
    view_stride: int = 1
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    da3: DA3Options = field(default_factory=DA3Options)
    fusion: FusionOptions = field(default_factory=FusionOptions)
    alignment: AlignmentOptions = field(default_factory=AlignmentOptions)
    superpoints: SuperpointOptions = field(default_factory=SuperpointOptions)
    sam2: SAM2Options = field(default_factory=SAM2Options)
    merging: MergingOptions = field(default_factory=MergingOptions)
    embedding: EmbeddingOptions = field(default_factory=EmbeddingOptions)
    siglip: SigLIP2Options = field(default_factory=SigLIP2Options)
    captioning: CaptioningOptions = field(default_factory=CaptioningOptions)
    query_parser: QueryParserOptions = field(default_factory=QueryParserOptions)
    caption_embedding: CaptionEmbeddingOptions = field(default_factory=CaptionEmbeddingOptions)
    spatial: SpatialOptions = field(default_factory=SpatialOptions)

    def __post_init__(self) -> None:
        """Refuse View limits that no Capture could satisfy."""
        if self.min_views < 1:
            raise ValueError(f"min_views must be at least 1, got {self.min_views}")
        if self.min_views > self.max_views:
            raise ValueError(
                f"min_views ({self.min_views}) must not exceed max_views ({self.max_views})"
            )
        if self.upload_max_side < 1:
            raise ValueError(f"upload_max_side must be at least 1, got {self.upload_max_side}")
        if not 0 < self.upload_jpeg_quality <= 1:
            raise ValueError(
                f"upload_jpeg_quality must be in (0, 1], got {self.upload_jpeg_quality}"
            )
        if self.view_stride < 1:
            raise ValueError(f"view_stride must be at least 1, got {self.view_stride}")
        if self.query_top_k < 1:
            raise ValueError(f"query_top_k must be at least 1, got {self.query_top_k}")
        if self.query_max_top_k < self.query_top_k:
            raise ValueError(
                f"query_max_top_k ({self.query_max_top_k}) must not be below "
                f"query_top_k ({self.query_top_k})"
            )
        if not 0 <= self.query_score_ratio <= 1:
            raise ValueError(f"query_score_ratio must be in [0, 1], got {self.query_score_ratio}")
        if not 0 <= self.negative_query_threshold <= 1:
            raise ValueError(
                f"negative_query_threshold must be in [0, 1], got {self.negative_query_threshold}"
            )
        if not 0 <= self.embedding.masked_weight <= 1:
            raise ValueError(
                f"embedding.masked_weight must be in [0, 1], got {self.embedding.masked_weight}"
            )
        if self.embedding.max_splat_radius < 1:
            raise ValueError(
                "embedding.max_splat_radius must be at least 1, "
                f"got {self.embedding.max_splat_radius}"
            )
        if not 0 < self.captioning.background_dim <= 1:
            raise ValueError(
                f"captioning.background_dim must be in (0, 1], got {self.captioning.background_dim}"
            )
        if self.siglip.prompt_max_words < 0:
            raise ValueError(
                f"siglip.prompt_max_words must not be negative, got {self.siglip.prompt_max_words}"
            )
        if self.query_parser.max_new_tokens < 1:
            raise ValueError(
                "query_parser.max_new_tokens must be at least 1, "
                f"got {self.query_parser.max_new_tokens}"
            )
        if self.spatial.candidates_per_object < 1:
            raise ValueError(
                "spatial.candidates_per_object must be at least 1, "
                f"got {self.spatial.candidates_per_object}"
            )
        if not self.spatial.falloff > 0:
            raise ValueError(f"spatial.falloff must be positive, got {self.spatial.falloff}")
        for name in ("caption_margin", "image_margin"):
            margin = getattr(self.spatial, name)
            if not margin >= 0:
                raise ValueError(f"spatial.{name} must not be negative, got {margin}")

    @classmethod
    def from_overrides(cls, token: str, overrides: Mapping[str, Any]) -> Settings:
        """Build settings from the defaults here and a JSON-like object of overrides.

        Args:
            token: The bearer token, which never travels with the overrides.
            overrides: Values to replace, nested by stage, for example
                ``{"view_stride": 7, "da3": {"process_res": 392}}``.

        Raises:
            ValueError: If an override has an unknown key, a value of the wrong type,
                or leaves the settings unusable.
        """
        if "token" in overrides:
            raise ValueError("token must not be in the config; it is a secret")
        values = _field_values(cls, overrides, prefix="")
        values.setdefault("storage_root", DEFAULT_STORAGE_ROOT)
        return cls(token=token, **values)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``SCENETIQ_TOKEN`` and ``SCENETIQ_CONFIG``.

        Raises:
            RuntimeError: If ``SCENETIQ_TOKEN`` is not set.
            ValueError: If ``SCENETIQ_CONFIG`` is not a JSON object of valid overrides.
        """
        token = os.environ.get("SCENETIQ_TOKEN")
        if not token:
            raise RuntimeError("SCENETIQ_TOKEN must be set")
        return cls.from_overrides(token, overrides_from_json(os.environ.get("SCENETIQ_CONFIG")))


def overrides_from_json(raw: str | None) -> dict[str, Any]:
    """Parse the value of ``SCENETIQ_CONFIG``; an unset variable means no overrides.

    Raises:
        ValueError: If ``raw`` is not a JSON object.
    """
    if not raw:
        return {}
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SCENETIQ_CONFIG is not valid JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise ValueError(f"SCENETIQ_CONFIG must be a JSON object, got {raw!r}")
    return overrides


def _field_values(cls: type, overrides: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    hints = get_type_hints(cls)
    unknown = [key for key in overrides if key not in hints]
    if unknown:
        names = ", ".join(f"{prefix}{key}" for key in unknown)
        raise ValueError(f"unknown config key: {names}")
    return {
        key: _typed_value(hints[key], value, f"{prefix}{key}") for key, value in overrides.items()
    }


def _typed_value(hint: Any, value: Any, name: str) -> Any:
    origin = get_origin(hint)
    if dataclasses.is_dataclass(hint):
        if isinstance(value, Mapping):
            return hint(**_field_values(hint, value, prefix=f"{name}."))
    elif hint is Path:
        if isinstance(value, str):
            return Path(value)
    elif hint is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif origin is Literal:
        if value in get_args(hint):
            return value
    elif origin in (list, tuple):
        if isinstance(value, list | tuple):
            element = get_args(hint)[0]
            return origin(_typed_value(element, item, name) for item in value)
    elif type(value) is hint:
        return value
    raise ValueError(f"config value {name} has the wrong type: {value!r}")
