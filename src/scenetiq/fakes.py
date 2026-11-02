"""Deterministic stand-ins for the GPU Pipeline and text encoder.

Used by the CPU test suite and selectable on a running backend with
``SCENETIQ_PIPELINE=fake`` so the whole system can be exercised without models.
"""

from __future__ import annotations

import threading

import numpy as np

from scenetiq.domain import (
    Capture,
    FloatArray,
    Reconstruction,
    Stage,
    StageCallback,
)
from scenetiq.geometry import build_instances, l2_normalize

EMBEDDING_DIM = 8


def _basis(i: int) -> FloatArray:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
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
}
"""Known Query texts. Any other text weakly resembles the chair (score 0.3)."""

UNKNOWN_TEXT_EMBEDDING = l2_normalize([0.3, 0, 0, 0, 0, 0, 0, 0.9539])


def fake_reconstruction() -> Reconstruction:
    """Return a small fixed Reconstruction with three Instances and some background.

    Points: 0-3 belong to Instance 0, 4-7 to Instance 1, 8-9 to Instance 2,
    and 10-11 to no Instance.
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
        instances=build_instances(points, instance_ids, FAKE_INSTANCE_EMBEDDINGS),
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
