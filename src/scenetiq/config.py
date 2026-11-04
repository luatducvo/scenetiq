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

DEFAULT_MATCH_THRESHOLD = 0.5
"""Default relevance threshold for a Match.

With the real Pipeline this is a SigLIP2 match probability. The value has not been validated
against ground truth; override it per deployment with ``match_threshold`` in the config.
"""

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
    """

    max_views: int = 12
    use_focal_prior: bool = True
    consensus_angle_deg: float = 10.0
    consensus_fraction: float = 0.5


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
        expansions: Context added around the tight crop, as fractions of its size.
        visibility_rtol: Relative depth tolerance of the occlusion test.
    """

    views_per_instance: int = 5
    min_visible_pixels: int = 30
    expansions: tuple[float, ...] = (0.1, 0.4)
    visibility_rtol: float = 0.05


@dataclass(frozen=True)
class SigLIP2Options:
    """Settings of the SigLIP2 encoder.

    Attributes:
        model_id: Hugging Face id of the SigLIP2 checkpoint (Apache-2.0).
        prompt: Template wrapping Query text; ``{}`` is replaced by the text.
        batch_size: Crops encoded per forward pass.
    """

    model_id: str = "google/siglip2-giant-opt-patch16-384"
    prompt: str = "This is a photo of {}."
    batch_size: int = 64


@dataclass(frozen=True)
class Settings:
    """Backend settings.

    Attributes:
        storage_root: Directory holding the SQLite database and all Scene data.
        token: Shared secret every request must present as a bearer token.
        match_threshold: Minimum relevance score for an Instance to be a Match.
        pipeline: Which Pipeline to run: ``"fake"`` or ``"real"``.
        min_views: Fewest photos a Capture may upload.
        max_views: Most photos a Capture may upload.
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

    Raises:
        ValueError: If ``min_views`` is below 1 or above ``max_views``, or ``view_stride``
            is below 1.
    """

    storage_root: Path
    token: str
    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    pipeline: Literal["fake", "real"] = "fake"
    min_views: int = 1
    max_views: int = 2000
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

    def __post_init__(self) -> None:
        """Refuse View limits that no Capture could satisfy."""
        if self.min_views < 1:
            raise ValueError(f"min_views must be at least 1, got {self.min_views}")
        if self.min_views > self.max_views:
            raise ValueError(
                f"min_views ({self.min_views}) must not exceed max_views ({self.max_views})"
            )
        if self.view_stride < 1:
            raise ValueError(f"view_stride must be at least 1, got {self.view_stride}")

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
