"""Core domain types: Capture, View, Reconstruction, Instance, Caption, Aspect and Stages.

Terms follow ``GLOSSARY.md``.
"""

from __future__ import annotations

from collections.abc import Callable
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


StageCallback = Callable[[Stage], None]


class Pipeline(Protocol):
    """Turns a Capture into a Reconstruction."""

    def run(self, capture: Capture, on_stage: StageCallback) -> Reconstruction:
        """Reconstruct ``capture``, calling ``on_stage`` as each stage starts."""
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
