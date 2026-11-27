from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

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


class Relation(StrEnum):
    ABOVE = "above"
    BELOW = "below"
    ON = "on"
    UNDER = "under"
    NEAR = "near"
    FAR_FROM = "far_from"
    BETWEEN = "between"
    NEAREST_TO = "nearest_to"
    FARTHEST_FROM = "farthest_from"
    LARGEST = "largest"
    SMALLEST = "smallest"
    TALLEST = "tallest"
    LOWEST = "lowest"

    @property
    def anchor_count(self) -> int:
        if self in (Relation.LARGEST, Relation.SMALLEST, Relation.TALLEST, Relation.LOWEST):
            return 0
        return 2 if self is Relation.BETWEEN else 1


MAX_QUERY_OBJECTS = 4
MAX_CONSTRAINTS = 4


@dataclass(frozen=True)
class QueryObject:
    object_id: str
    phrase: str
    aspects: tuple[Aspect, ...] = ()


@dataclass(frozen=True)
class Constraint:
    relation: Relation
    subject: str
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedQuery:
    objects: tuple[QueryObject, ...]
    target: str
    constraints: tuple[Constraint, ...] = ()
    unsupported: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.objects:
            raise ValueError("a parsed Query needs at least one object")
        if len(self.objects) > MAX_QUERY_OBJECTS:
            raise ValueError(f"a parsed Query has at most {MAX_QUERY_OBJECTS} objects")
        if len(self.constraints) > MAX_CONSTRAINTS:
            raise ValueError(f"a parsed Query has at most {MAX_CONSTRAINTS} constraints")
        ids = [o.object_id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"a parsed Query repeats an object id: {ids}")
        if self.target not in ids:
            raise ValueError(f"the target {self.target!r} is not one of the objects {ids}")
        for c in self.constraints:
            for named in (c.subject, *c.anchors):
                if named not in ids:
                    raise ValueError(f"{c.relation} names object {named!r}, which is not listed")
            if len(c.anchors) != c.relation.anchor_count:
                raise ValueError(
                    f"{c.relation} takes {c.relation.anchor_count} Anchor(s), got {len(c.anchors)}"
                )
            if c.subject in c.anchors:
                raise ValueError(f"{c.relation} uses object {c.subject!r} as its own Anchor")
            if len(set(c.anchors)) != len(c.anchors):
                raise ValueError(f"{c.relation} names one Anchor twice: {list(c.anchors)}")

    @classmethod
    def without_relations(cls, text: str) -> ParsedQuery:
        return cls(objects=(QueryObject("a", text),), target="a")


class JobCancelledError(Exception): ...


class JobReporter(Protocol):
    def stage(self, stage: Stage) -> None: ...

    def checkpoint(self) -> None: ...

    def note(self, message: str) -> None: ...


class Pipeline(Protocol):
    def run(self, capture: Capture, reporter: JobReporter) -> Reconstruction: ...


class TextEncoder(Protocol):
    def encode(self, text: str) -> FloatArray: ...


class CaptionEncoder(Protocol):
    def encode_documents(self, texts: list[str]) -> FloatArray: ...

    def encode_query(self, text: str) -> FloatArray: ...


class QueryParseError(Exception): ...


class QueryParser(Protocol):
    def parse(self, text: str) -> ParsedQuery: ...
