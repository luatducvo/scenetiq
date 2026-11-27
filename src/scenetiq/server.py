from typing import NamedTuple

from fastapi import FastAPI

from scenetiq.api.app import create_app
from scenetiq.config import Settings
from scenetiq.domain import CaptionEncoder, Pipeline, QueryParser, TextEncoder
from scenetiq.fakes import FakeCaptionEncoder, FakePipeline, FakeQueryParser, FakeTextEncoder


class Components(NamedTuple):
    pipeline: Pipeline
    text_encoder: TextEncoder
    caption_encoder: CaptionEncoder
    query_parser: QueryParser


def build_components(settings: Settings) -> Components:
    if settings.pipeline == "fake":
        return Components(
            FakePipeline(), FakeTextEncoder(), FakeCaptionEncoder(), FakeQueryParser()
        )
    if settings.pipeline == "real":
        from scenetiq.pipeline.masking import SAM2Masker
        from scenetiq.pipeline.models.qwen3_embedding import Qwen3TextEmbedder
        from scenetiq.pipeline.models.qwen3_vl import QwenVLCaptioner
        from scenetiq.pipeline.models.siglip2 import SigLIP2Encoder
        from scenetiq.pipeline.real_pipeline import RealPipeline, RealPipelineOptions
        from scenetiq.pipeline.reconstructing import DA3Reconstructor
        from scenetiq.query.parser import VLMQueryParser

        encoder = SigLIP2Encoder(settings.siglip)  # one model for Instance crops and Query text
        caption_encoder = Qwen3TextEmbedder(settings.caption_embedding)  # Descriptions, Queries
        # One vision-language model for Captioning and the Query parser; it stays loaded.
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
        query_parser = VLMQueryParser(captioner, settings.query_parser)
        return Components(pipeline, encoder, caption_encoder, query_parser)
    raise ValueError(f"unknown pipeline {settings.pipeline!r}; expected 'fake' or 'real'")


def build_app() -> FastAPI:
    settings = Settings.from_env()
    return create_app(settings, *build_components(settings))
