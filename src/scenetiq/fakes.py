from __future__ import annotations

import threading
from collections.abc import Mapping

import numpy as np

from scenetiq.domain import (
    Aspect,
    Caption,
    Capture,
    EmbeddedCaption,
    FloatArray,
    JobReporter,
    ParsedQuery,
    QueryParseError,
    Reconstruction,
    Stage,
)
from scenetiq.geometry import build_instances, l2_normalize

EMBEDDING_DIM = 8
CAPTION_EMBEDDING_DIM = 7


def _basis(i: int, dim: int = EMBEDDING_DIM) -> FloatArray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


FAKE_INSTANCE_EMBEDDINGS: dict[int, FloatArray] = {0: _basis(0), 1: _basis(1), 2: _basis(2)}

FAKE_TEXT_EMBEDDINGS: dict[str, FloatArray] = {
    "red chair": _basis(0),
    "table": _basis(1),
    "lamp": _basis(2),
    # Close to the chair (score 0.8), far from the rest.
    "chair": l2_normalize([0.8, 0.0, 0.0, 0.6, 0, 0, 0, 0]),
    # Similar to both table (0.7) and lamp (0.6).
    "furniture light": l2_normalize([0.0, 0.7, 0.6, 0.3873, 0, 0, 0, 0]),
    # Looks most like the lamp (0.5), then the table (0.3), then the chair (0.1).
    "wooden thing": l2_normalize([0.1, 0.3, 0.5, 0, 0, 0, 0, 0.806226]),
    # Looks most like the lamp (0.6), then the chair (0.2), then the table (0.1).
    "red fabric": l2_normalize([0.2, 0.1, 0.6, 0, 0, 0, 0, 0.768115]),
}

UNKNOWN_TEXT_EMBEDDING = l2_normalize([0.3, 0, 0, 0, 0, 0, 0, 0.9539])


def _caption_basis(i: int) -> FloatArray:
    return _basis(i, CAPTION_EMBEDDING_DIM)


FAKE_CAPTIONS: dict[int, EmbeddedCaption] = {
    0: EmbeddedCaption(
        Caption(
            "chair",
            ("seat",),
            identity="A chair, also called a seat.",
            appearance="It is red with a fabric cushion.",
        ),
        {Aspect.IDENTITY: _caption_basis(0), Aspect.APPEARANCE: _caption_basis(2)},
    ),
    1: EmbeddedCaption(
        Caption(
            "table",
            ("desk",),
            identity="A table, also called a desk.",
            appearance="It is low and rectangular.",
            material="It is made of wood.",
            use="It is used for putting things on.",
        ),
        {
            Aspect.IDENTITY: _caption_basis(1),
            Aspect.APPEARANCE: _caption_basis(5),
            Aspect.MATERIAL: _caption_basis(3),
            Aspect.USE: _caption_basis(4),
        },
    ),
}

FAKE_CAPTION_QUERY_EMBEDDINGS: dict[str, FloatArray] = {
    "red chair": _caption_basis(0),
    "chair": _caption_basis(0),
    "table": _caption_basis(1),
    # The table's Material Description.
    "wooden thing": _caption_basis(3),
    # The chair's Appearance (0.8) and the table's Identity (0.6).
    "red fabric": l2_normalize([0.0, 0.6, 0.8, 0, 0, 0, 0]),
    # Unlike the chair's Identity (-0.6) and Appearance (-0.8); orthogonal to the table.
    "metal legs": l2_normalize([-0.6, 0.0, -0.8, 0, 0, 0, 0]),
}

UNKNOWN_CAPTION_QUERY_EMBEDDING = _caption_basis(6)


def fake_reconstruction(with_captions: bool = True) -> Reconstruction:
    points = np.array(
        [
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 1.0],
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.8],
            [5.0, 5.0, 1.5], [5.0, 5.0, 1.9],
            [-1.0, -1.0, 0.0], [9.0, 9.0, 0.0],
        ],
        dtype=np.float32,
    )  # fmt: skip
    colors = np.tile(np.array([[200, 30, 30]], dtype=np.uint8), (len(points), 1))
    instance_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, -1, -1], dtype=np.int32)
    return Reconstruction(
        points=points,
        colors=colors,
        instance_ids=instance_ids,
        instances=build_instances(
            points,
            instance_ids,
            FAKE_INSTANCE_EMBEDDINGS,
            FAKE_CAPTIONS if with_captions else None,
        ),
    )


FAKE_NOTE = "Reconstructed the fake Scene."


class FakePipeline:
    def __init__(
        self,
        reconstruction: Reconstruction | None = None,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.reconstruction = reconstruction
        self.error = error
        self.gate = gate
        self.captures: list[Capture] = []

    def run(self, capture: Capture, reporter: JobReporter) -> Reconstruction:
        self.captures.append(capture)
        for i, stage in enumerate(Stage):
            reporter.stage(stage)
            if i == 0:
                reporter.note(FAKE_NOTE)
            if i == 0 and self.gate is not None:
                self.gate.wait(timeout=10)
            if i == 0 and self.error is not None:
                raise self.error
        return self.reconstruction or fake_reconstruction()


class FakeTextEncoder:
    def encode(self, text: str) -> FloatArray:
        return FAKE_TEXT_EMBEDDINGS.get(text.strip().lower(), UNKNOWN_TEXT_EMBEDDING)


class FakeCaptionEncoder:
    def encode_documents(self, texts: list[str]) -> FloatArray:
        return np.stack([self.encode_query(t) for t in texts]).astype(np.float32)

    def encode_query(self, text: str) -> FloatArray:
        return FAKE_CAPTION_QUERY_EMBEDDINGS.get(
            text.strip().lower(), UNKNOWN_CAPTION_QUERY_EMBEDDING
        )


class FakeQueryParser:
    def __init__(self, prepared: Mapping[str, ParsedQuery | QueryParseError] | None = None) -> None:
        self.prepared = {t.strip().lower(): p for t, p in (prepared or {}).items()}

    def parse(self, text: str) -> ParsedQuery:
        found = self.prepared.get(text.strip().lower())
        if isinstance(found, QueryParseError):
            raise found
        return found if found is not None else ParsedQuery.without_relations(text)
