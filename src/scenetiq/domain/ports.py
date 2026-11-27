from __future__ import annotations

from typing import Protocol

from scenetiq.domain.query import ParsedQuery
from scenetiq.domain.scene import Capture, FloatArray, Reconstruction, Stage


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


class QueryParser(Protocol):
    def parse(self, text: str) -> ParsedQuery: ...
