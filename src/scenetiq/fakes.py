"""Deterministic stand-ins for the GPU Pipeline and the text encoders.

Used by the CPU test suite and selectable on a running backend with
``"pipeline": "fake"`` in the config, so the whole system can be exercised without models.
"""

from __future__ import annotations

import threading

import numpy as np

from scenetiq.domain import (
    Caption,
    Capture,
    EmbeddedCaption,
    FloatArray,
    Reconstruction,
    Stage,
    StageCallback,
)
from scenetiq.geometry import build_instances, l2_normalize

EMBEDDING_DIM = 8
"""Size of the fake image embeddings and Query embeddings."""
CAPTION_EMBEDDING_DIM = 4
"""Size of the fake Caption embeddings and Caption Query embeddings."""


def _basis(i: int, dim: int = EMBEDDING_DIM) -> FloatArray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


FAKE_INSTANCE_EMBEDDINGS: dict[int, FloatArray] = {0: _basis(0), 1: _basis(1), 2: _basis(2)}
"""Fake Instances: 0 is a red chair, 1 a table, 2 a lamp."""

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
}
"""Known Query texts. Any other text weakly resembles the chair (score 0.3)."""

UNKNOWN_TEXT_EMBEDDING = l2_normalize([0.3, 0, 0, 0, 0, 0, 0, 0.9539])
"""Image-space embedding of any other Query text: weakly resembles the chair (0.3)."""

FAKE_CAPTIONS: dict[int, EmbeddedCaption] = {
    0: EmbeddedCaption(
        Caption("chair", ("seat",), ("red", "fabric"), "sitting"),
        _basis(0, CAPTION_EMBEDDING_DIM),
    ),
    1: EmbeddedCaption(
        Caption("table", ("desk",), ("wooden",), "putting things on"),
        _basis(1, CAPTION_EMBEDDING_DIM),
    ),
}
"""Captions of the fake Instances; the lamp (2) has none."""

FAKE_CAPTION_QUERY_EMBEDDINGS: dict[str, FloatArray] = {
    "red chair": _basis(0, CAPTION_EMBEDDING_DIM),
    "chair": _basis(0, CAPTION_EMBEDDING_DIM),
    "table": _basis(1, CAPTION_EMBEDDING_DIM),
    "wooden thing": _basis(1, CAPTION_EMBEDDING_DIM),
}
"""Known Query texts in the Caption space. Any other text resembles no Caption."""

UNKNOWN_CAPTION_QUERY_EMBEDDING = _basis(3, CAPTION_EMBEDDING_DIM)
"""Caption-space embedding of any other Query text: orthogonal to every fake Caption."""


def fake_reconstruction(with_captions: bool = True) -> Reconstruction:
    """Return a small fixed Reconstruction with three Instances and some background.

    Points: 0-3 belong to Instance 0, 4-7 to Instance 1, 8-9 to Instance 2,
    and 10-11 to no Instance.

    Args:
        with_captions: Give the chair and the table their :data:`FAKE_CAPTIONS`.
    """
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


class FakePipeline:
    """A Pipeline that reports every Stage and returns a fixed Reconstruction.

    Args:
        reconstruction: What to return; defaults to :func:`fake_reconstruction`.
        error: If set, raised after the first stage to simulate a failure.
        gate: If set, the run blocks until the event is set, so a job stays running.
    """

    def __init__(
        self,
        reconstruction: Reconstruction | None = None,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        """Create the fake; see the class docstring for the arguments."""
        self.reconstruction = reconstruction
        self.error = error
        self.gate = gate
        self.captures: list[Capture] = []

    def run(self, capture: Capture, on_stage: StageCallback) -> Reconstruction:
        """Record ``capture``, walk through every Stage and return the fixed result."""
        self.captures.append(capture)
        for i, stage in enumerate(Stage):
            on_stage(stage)
            if i == 0 and self.gate is not None:
                self.gate.wait(timeout=10)
            if i == 0 and self.error is not None:
                raise self.error
        return self.reconstruction or fake_reconstruction()


class FakeTextEncoder:
    """Maps the texts in :data:`FAKE_TEXT_EMBEDDINGS` to fixed vectors."""

    def encode(self, text: str) -> FloatArray:
        """Return the fixed embedding for ``text`` (case-insensitive)."""
        return FAKE_TEXT_EMBEDDINGS.get(text.strip().lower(), UNKNOWN_TEXT_EMBEDDING)


class FakeCaptionEncoder:
    """Maps the texts in :data:`FAKE_CAPTION_QUERY_EMBEDDINGS` to fixed vectors."""

    def encode_documents(self, texts: list[str]) -> FloatArray:
        """Return one fixed vector per document: its Query embedding, if known."""
        return np.stack([self.encode_query(t) for t in texts]).astype(np.float32)

    def encode_query(self, text: str) -> FloatArray:
        """Return the fixed embedding for ``text`` (case-insensitive)."""
        return FAKE_CAPTION_QUERY_EMBEDDINGS.get(
            text.strip().lower(), UNKNOWN_CAPTION_QUERY_EMBEDDING
        )
