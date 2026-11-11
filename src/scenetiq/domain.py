"""Core domain types: Capture, View, Reconstruction, Instance, Caption, Relation and Stages.

Also the parsed Query, and the protocols of the Pipeline, the encoders and the parser.

Terms follow ``GLOSSARY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


class Stage(StrEnum):
    """A step of the pipeline, reported to the user as job progress."""

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
    """One photo of a Capture, stored on disk."""

    path: Path


@dataclass(frozen=True)
class Capture:
    """The Views of one Capture that a Pipeline reconstructs, in upload order.

    Holds only the Views the View stride selected.
    """

    views: list[View]


@dataclass(frozen=True)
class Camera:
    """The reconstructed camera of one View.

    Attributes:
        cam_to_world: 4x4 camera-to-world transform (OpenCV convention), metres.
        intrinsics: 3x3 pinhole intrinsics for an image of ``width`` x ``height``.
        width: Image width in pixels the intrinsics refer to.
        height: Image height in pixels the intrinsics refer to.
    """

    cam_to_world: FloatArray
    intrinsics: FloatArray
    width: int
    height: int


class Aspect(StrEnum):
    """A fixed side of an Instance that a Caption describes in one Description.

    The order is fixed: prompt, storage and scoring all follow it.
    """

    IDENTITY = "identity"
    APPEARANCE = "appearance"
    MATERIAL = "material"
    USE = "use"


@dataclass(frozen=True)
class Caption:
    """What a vision-language model reads from an Instance's Views.

    Attributes:
        name: Short free-text name of the Instance; never empty.
        synonyms: Up to three other names for it.
        identity: Description of what it is and what else it is called.
        appearance: Description of its colour, shape and relative size.
        material: Description of what it is made of and its parts.
        use: Description of what it is used for.
    """

    name: str
    synonyms: tuple[str, ...] = ()
    identity: str = ""
    appearance: str = ""
    material: str = ""
    use: str = ""

    def description(self, aspect: Aspect) -> str:
        """The Description of ``aspect``, empty when the model gave none."""
        return getattr(self, aspect.value)

    def descriptions(self) -> dict[Aspect, str]:
        """The non-empty Descriptions, in Aspect order."""
        return {a: self.description(a) for a in Aspect if self.description(a)}


@dataclass(frozen=True, eq=False)
class EmbeddedCaption:
    """A Caption with the embeddings of its Descriptions, as the Captioning stage outputs it.

    Attributes:
        caption: The Caption.
        embeddings: L2-normalised embedding of every non-empty Description, by Aspect,
            in the Caption encoder's space.
    """

    caption: Caption
    embeddings: dict[Aspect, FloatArray]


@dataclass
class Instance:
    """One distinct physical object in a Reconstruction.

    Attributes:
        instance_id: Non-negative id, unique within its Reconstruction.
        centroid: Mean position of the Instance's points, metres.
        bbox_min: Minimum corner of the axis-aligned bounding box, metres.
        bbox_max: Maximum corner of the axis-aligned bounding box, metres.
        embedding: L2-normalised image embedding, or ``None`` before the
            Embedding stage has run.
        caption: The Instance's Caption, or ``None`` when it has none.
        caption_embeddings: L2-normalised embedding of each non-empty Description of the
            Caption, by Aspect; empty without a Caption.
    """

    instance_id: int
    centroid: FloatArray
    bbox_min: FloatArray
    bbox_max: FloatArray
    embedding: FloatArray | None = None
    caption: Caption | None = None
    caption_embeddings: dict[Aspect, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse Caption embeddings that have no Caption."""
        if self.caption_embeddings and self.caption is None:
            raise ValueError(f"Instance {self.instance_id} has Caption embeddings but no Caption")


@dataclass
class Reconstruction:
    """The 3D result of one Capture.

    Attributes:
        points: ``(N, 3)`` float32 positions in metres.
        colors: ``(N, 3)`` uint8 RGB colours.
        instance_ids: ``(N,)`` int32 Instance id per point, ``-1`` for none.
        cameras: The camera of every View, in Capture order.
        instances: The Instances found in the Scene.
    """

    points: FloatArray
    colors: npt.NDArray[np.uint8]
    instance_ids: npt.NDArray[np.int32]
    cameras: list[Camera] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)


class Relation(StrEnum):
    """A spatial relationship between Instances that a Query states (ADR 0009).

    The set is closed: the parser's schema, its prompt and the solver all follow it. The
    values are the names the parser writes.
    """

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
        """How many Anchors the Relation takes: 0, 1 or 2."""
        if self in (Relation.LARGEST, Relation.SMALLEST, Relation.TALLEST, Relation.LOWEST):
            return 0
        return 2 if self is Relation.BETWEEN else 1


MAX_QUERY_OBJECTS = 4
"""Most objects a parsed Query may mention."""
MAX_CONSTRAINTS = 4
"""Most constraints a parsed Query may state."""


@dataclass(frozen=True)
class QueryObject:
    """One object a Query mentions.

    Attributes:
        object_id: Id of the object within its parsed Query, such as ``"a"``.
        phrase: Short English phrase naming the object, ranked like a Query of its own.
    """

    object_id: str
    phrase: str


@dataclass(frozen=True)
class Constraint:
    """A Relation a Query states between one object and its Anchor objects.

    Attributes:
        relation: The Relation.
        subject: Id of the object the Relation is about.
        anchors: Ids of the objects it is measured against, as many as the Relation takes.
    """

    relation: Relation
    subject: str
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedQuery:
    """A Query split into the objects it mentions and the Relations between them.

    Attributes:
        objects: The objects, one to :data:`MAX_QUERY_OBJECTS`, each with a unique id.
        target: Id of the object the user wants; the other objects are Anchors.
        constraints: Up to :data:`MAX_CONSTRAINTS` Relations between the objects. A
            Query without constraints is answered as a plain Query, with its original text.
        unsupported: Parts of the Query the Relation set cannot express, quoted from it.

    Raises:
        ValueError: If the objects, the target and the constraints do not fit together:
            an object id repeats, the target or a constraint names an object that is not
            listed, a Relation has the wrong number of Anchors, a subject is its own
            Anchor, or a Relation names one Anchor twice.
    """

    objects: tuple[QueryObject, ...]
    target: str
    constraints: tuple[Constraint, ...] = ()
    unsupported: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a parsed Query whose parts do not fit together."""
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
        """A Query with no Relation: one object, the whole text, which is the target."""
        return cls(objects=(QueryObject("a", text),), target="a")


class JobReporter(Protocol):
    """What a running Pipeline tells its job: stages, notes and when it may stop."""

    def stage(self, stage: Stage) -> None:
        """Report that ``stage`` starts."""
        ...

    def checkpoint(self) -> None:
        """Mark a point where the job may stop if the user asked it to."""
        ...

    def note(self, message: str) -> None:
        """Record one line of the job summary the user sees, under the running stage."""
        ...


class Pipeline(Protocol):
    """Turns a Capture into a Reconstruction."""

    def run(self, capture: Capture, reporter: JobReporter) -> Reconstruction:
        """Reconstruct ``capture``, reporting each stage as it starts through ``reporter``."""
        ...


class TextEncoder(Protocol):
    """Encodes Query text into the same space as Instance image embeddings."""

    def encode(self, text: str) -> FloatArray:
        """Return the L2-normalised embedding of ``text``."""
        ...


class CaptionEncoder(Protocol):
    """Embeds Descriptions and Query text into one text space (ADR 0008)."""

    def encode_documents(self, texts: list[str]) -> FloatArray:
        """Return ``(len(texts), D)`` L2-normalised embeddings of Descriptions."""
        ...

    def encode_query(self, text: str) -> FloatArray:
        """Return the L2-normalised embedding of Query ``text``."""
        ...


class QueryParseError(Exception):
    """Raised by a Query parser that could not read a Query, such as an unusable reply.

    It is the only failure the Query endpoint treats as a parse failure: the Query is then
    answered as a plain Query and the response says the parse failed. Any other exception
    from a parser is a bug and fails the request. The message is the reason, and it is
    logged.
    """


class QueryParser(Protocol):
    """Splits a Query into the objects it mentions and the Relations between them."""

    def parse(self, text: str) -> ParsedQuery:
        """Return the parsed Query of ``text``, without constraints when it states no Relation.

        Raises:
            QueryParseError: If the Query could not be read; its message says why.
        """
        ...
