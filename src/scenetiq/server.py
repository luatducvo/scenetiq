"""Process entry point: build the API from environment configuration.

Run with ``uvicorn --factory scenetiq.server:build_app``.
"""

from typing import NamedTuple

from fastapi import FastAPI

from scenetiq.api import create_app
from scenetiq.config import Settings
from scenetiq.domain import CaptionEncoder, Pipeline, QueryParser, TextEncoder
from scenetiq.fakes import FakeCaptionEncoder, FakePipeline, FakeQueryParser, FakeTextEncoder
from scenetiq.spatial import NoRelationParser


class Components(NamedTuple):
    """The shared components the API is built from, in :func:`create_app`'s order.

    Attributes:
        pipeline: Turns a Capture into a Reconstruction.
        text_encoder: Encodes Query text against Instance image embeddings.
        caption_encoder: Encodes Query text against Descriptions.
        query_parser: Splits every Query into objects and Relations.
    """

    pipeline: Pipeline
    text_encoder: TextEncoder
    caption_encoder: CaptionEncoder
    query_parser: QueryParser


def build_components(settings: Settings) -> Components:
    """Return the shared components selected by ``settings.pipeline``.

    ``"real"`` needs the GPU stack from ``requirements-gpu.txt``; models load on first use.

    Raises:
        ValueError: If ``settings.pipeline`` names no known implementation.
    """
    if settings.pipeline == "fake":
        return Components(
            FakePipeline(), FakeTextEncoder(), FakeCaptionEncoder(), FakeQueryParser()
        )
    if settings.pipeline == "real":
        from scenetiq.pipeline.masking import SAM2Masker
        from scenetiq.pipeline.qwen_embedding import Qwen3TextEmbedder
        from scenetiq.pipeline.qwen_vl import QwenVLCaptioner
        from scenetiq.pipeline.real import RealPipeline, RealPipelineOptions
        from scenetiq.pipeline.reconstruct import DA3Reconstructor
        from scenetiq.pipeline.siglip import SigLIP2Encoder

        encoder = SigLIP2Encoder(settings.siglip)  # one model for Instance crops and Query text
        caption_encoder = Qwen3TextEmbedder(settings.caption_embedding)  # Descriptions, Queries
        # One vision-language model for Captioning and, later, the Query parser; it stays loaded.
        captioner = QwenVLCaptioner(settings.captioning)
        pipeline = RealPipeline(
            reconstructor=DA3Reconstructor(settings.da3),
            masker=SAM2Masker(settings.sam2),
            image_encoder=encoder,
            options=RealPipelineOptions(
                fusion=settings.fusion,
                alignment=settings.alignment,
                superpoints=settings.superpoints,
                merging=settings.merging,
                embedding=settings.embedding,
                captioning=settings.captioning,
            ),
            captioner=captioner,
            caption_encoder=caption_encoder,
        )
        # Until the parser on `captioner` exists, every Query is answered as a plain Query.
        return Components(pipeline, encoder, caption_encoder, NoRelationParser())
    raise ValueError(f"unknown pipeline {settings.pipeline!r}; expected 'fake' or 'real'")


def build_app() -> FastAPI:
    """Create the API configured by ``SCENETIQ_TOKEN`` and ``SCENETIQ_CONFIG``."""
    settings = Settings.from_env()
    return create_app(settings, *build_components(settings))
