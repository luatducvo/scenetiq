from typing import NamedTuple

from fastapi import FastAPI

from scenetiq.api.app import create_app
from scenetiq.config import Settings
from scenetiq.domain.ports import CaptionEncoder, Pipeline, QueryParser, TextEncoder


class Components(NamedTuple):
    pipeline: Pipeline
    text_encoder: TextEncoder
    caption_encoder: CaptionEncoder
    query_parser: QueryParser


def build_components(settings: Settings) -> Components:
    from scenetiq.pipeline.model_pipeline import ModelPipeline, ModelPipelineOptions
    from scenetiq.pipeline.models.qwen3_embedding import Qwen3TextEmbedder
    from scenetiq.pipeline.models.qwen3_vl import QwenVLCaptioner
    from scenetiq.pipeline.models.siglip2 import SigLIP2Encoder
    from scenetiq.pipeline.stages.masking import SAM2Masker
    from scenetiq.pipeline.stages.reconstructing import DA3Reconstructor
    from scenetiq.query.parser import VLMQueryParser

    encoder = SigLIP2Encoder(settings.siglip)  # one model for Instance crops and Query text
    caption_encoder = Qwen3TextEmbedder(settings.caption_embedding)  # Descriptions, Queries
    # One vision-language model for Captioning and the Query parser; it stays loaded.
    captioner = QwenVLCaptioner(settings.captioning)
    pipeline = ModelPipeline(
        reconstructor=DA3Reconstructor(settings.da3),
        masker=SAM2Masker(settings.sam2),
        image_encoder=encoder,
        options=ModelPipelineOptions(
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


def build_app() -> FastAPI:
    settings = Settings.from_env()
    return create_app(settings, *build_components(settings))
