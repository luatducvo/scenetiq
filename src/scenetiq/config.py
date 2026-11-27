from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

DEFAULT_STORAGE_ROOT = Path("./data")


@dataclass(frozen=True)
class DA3Options:
    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    ref_view_strategy: str = "saddle_balanced"
    use_ray_pose: bool = False
    conf_percentile: float = 30.0
    max_depth: float = 12.0


@dataclass(frozen=True)
class FusionOptions:
    voxel_size: float = 0.02
    sdf_trunc_voxels: float = 4.0
    block_resolution: int = 16
    initial_block_count: int = 1000


@dataclass(frozen=True)
class AlignmentOptions:
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
    primitives: Literal["superpoints", "voxel"] = "superpoints"
    k_thresh: float = 0.01
    min_verts: int = 20
    primitive_voxel_size: float = 0.15


@dataclass(frozen=True)
class SAM2Options:
    model_id: str = "facebook/sam2.1-hiera-large"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.9
    min_area_fraction: float = 0.001


@dataclass(frozen=True)
class MergingOptions:
    visibility_rtol: float = 0.05
    merge_small_points: int = 200
    min_instance_points: int = 50


@dataclass(frozen=True)
class EmbeddingOptions:
    views_per_instance: int = 5
    min_visible_pixels: int = 30
    expansions: tuple[float, ...] = (0.1, 0.4)
    visibility_rtol: float = 0.05
    masked_weight: float = 0.5
    max_splat_radius: int = 12


@dataclass(frozen=True)
class SigLIP2Options:
    model_id: str = "google/siglip2-giant-opt-patch16-384"
    prompt: str = "This is a photo of {}."
    prompt_max_words: int = 4
    batch_size: int = 64


@dataclass(frozen=True)
class CaptioningOptions:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    views_per_instance: int = 3
    context_expand: float = 0.6
    background_dim: float = 0.35
    max_pixels: int = 448 * 448
    max_new_tokens: int = 320


@dataclass(frozen=True)
class QueryParserOptions:
    max_new_tokens: int = 1024


@dataclass(frozen=True)
class CaptionEmbeddingOptions:
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    query_instruction: str = (
        "Given a description of an object in a scene, retrieve the caption of that object"
    )
    batch_size: int = 32
    max_length: int = 256


@dataclass(frozen=True)
class SpatialOptions:
    candidates_per_object: int = 5
    falloff: float = 0.1
    caption_margin: float = 0.1
    image_margin: float = 0.05


@dataclass(frozen=True)
class Settings:
    storage_root: Path
    token: str
    query_top_k: int = 5
    query_max_top_k: int = 20
    query_score_ratio: float = 0.5
    negative_query_threshold: float = 0.35
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
        if "token" in overrides:
            raise ValueError("token must not be in the config; it is a secret")
        values = _field_values(cls, overrides, prefix="")
        values.setdefault("storage_root", DEFAULT_STORAGE_ROOT)
        return cls(token=token, **values)

    @classmethod
    def from_env(cls) -> Settings:
        token = os.environ.get("SCENETIQ_TOKEN")
        if not token:
            raise RuntimeError("SCENETIQ_TOKEN must be set")
        return cls.from_overrides(token, overrides_from_json(os.environ.get("SCENETIQ_CONFIG")))


def overrides_from_json(raw: str | None) -> dict[str, Any]:
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
