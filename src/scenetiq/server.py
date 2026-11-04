"""Process entry point: build the API from environment configuration.

Run with ``uvicorn --factory scenetiq.server:build_app``.
"""

from fastapi import FastAPI

from scenetiq.api import create_app
from scenetiq.config import Settings
from scenetiq.domain import Pipeline, TextEncoder
from scenetiq.fakes import FakePipeline, FakeTextEncoder


def build_components(settings: Settings) -> tuple[Pipeline, TextEncoder]:
    """Return the Pipeline and text encoder selected by ``settings.pipeline``.

    ``"real"`` needs the GPU stack from ``requirements-gpu.txt``; models load on first use.

    Raises:
        ValueError: If ``settings.pipeline`` names no known implementation.
    """
    if settings.pipeline == "fake":
        return FakePipeline(), FakeTextEncoder()
    if settings.pipeline == "real":
        from scenetiq.pipeline.masking import SAM2Masker
        from scenetiq.pipeline.real import RealPipeline, RealPipelineOptions
        from scenetiq.pipeline.reconstruct import DA3Reconstructor
        from scenetiq.pipeline.siglip import SigLIP2Encoder

        encoder = SigLIP2Encoder(settings.siglip)  # one model for Instance crops and Query text
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
            ),
        )
        return pipeline, encoder
    raise ValueError(f"unknown pipeline {settings.pipeline!r}; expected 'fake' or 'real'")


def build_app() -> FastAPI:
    """Create the API configured by ``SCENETIQ_TOKEN`` and ``SCENETIQ_CONFIG``."""
    settings = Settings.from_env()
    pipeline, text_encoder = build_components(settings)
    return create_app(settings, pipeline, text_encoder)
