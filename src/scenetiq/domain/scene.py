from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


class Stage(StrEnum):
    RECONSTRUCTING = "reconstructing"
    FUSING = "fusing"
    ALIGNING = "aligning"
    SUPERPOINTS = "superpoints"
    MASKING = "masking"
    MERGING = "merging"
    EMBEDDING = "embedding"
    CAPTIONING = "captioning"


@dataclass(frozen=True)
class View:
    path: Path


@dataclass(frozen=True)
class Capture:
    views: list[View]


@dataclass(frozen=True)
class Camera:
    cam_to_world: FloatArray
    intrinsics: FloatArray
    width: int
    height: int


class Aspect(StrEnum):
    IDENTITY = "identity"
    APPEARANCE = "appearance"
    MATERIAL = "material"
    USE = "use"


@dataclass(frozen=True)
class Caption:
    name: str
    synonyms: tuple[str, ...] = ()
    identity: str = ""
    appearance: str = ""
    material: str = ""
    use: str = ""

    def description(self, aspect: Aspect) -> str:
        return getattr(self, aspect.value)

    def descriptions(self) -> dict[Aspect, str]:
        return {a: self.description(a) for a in Aspect if self.description(a)}


@dataclass(frozen=True, eq=False)
class EmbeddedCaption:
    caption: Caption
    embeddings: dict[Aspect, FloatArray]


@dataclass
class Instance:
    instance_id: int
    centroid: FloatArray
    bbox_min: FloatArray
    bbox_max: FloatArray
    embedding: FloatArray | None = None
    caption: Caption | None = None
    caption_embeddings: dict[Aspect, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.caption_embeddings and self.caption is None:
            raise ValueError(f"Instance {self.instance_id} has Caption embeddings but no Caption")


@dataclass
class Reconstruction:
    points: FloatArray
    colors: npt.NDArray[np.uint8]
    instance_ids: npt.NDArray[np.int32]
    cameras: list[Camera] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
