"""Core domain types: Capture, View, Reconstruction, Instance and pipeline Stages.

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
    SUPERPOINTS = "superpoints"
    MASKING = "masking"
    MERGING = "merging"
    EMBEDDING = "embedding"


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


@dataclass
class Instance:
    """One distinct physical object in a Reconstruction.

    Attributes:
        instance_id: Non-negative id, unique within its Reconstruction.
        centroid: Mean position of the Instance's points, metres.
        bbox_min: Minimum corner of the axis-aligned bounding box, metres.
        bbox_max: Maximum corner of the axis-aligned bounding box, metres.
        embedding: L2-normalised semantic embedding, or ``None`` before the
            Embedding stage has run.
    """

    instance_id: int
    centroid: FloatArray
    bbox_min: FloatArray
    bbox_max: FloatArray
    embedding: FloatArray | None = None


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
    """Encodes Query text into the same space as Instance embeddings."""

    def encode(self, text: str) -> FloatArray:
        """Return the L2-normalised embedding of ``text``."""
        ...

    def to_score(self, cosine: FloatArray) -> FloatArray:
        """Map cosine similarities to the relevance scores the Match threshold applies to."""
        ...
