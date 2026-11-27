from fastapi import FastAPI

from scenetiq.api.app import create_app
from scenetiq.config import Settings
from scenetiq.server import Components
from tests.stubs.components import (
    StubCaptionEncoder,
    StubPipeline,
    StubQueryParser,
    StubTextEncoder,
)


def stub_components() -> Components:
    return Components(StubPipeline(), StubTextEncoder(), StubCaptionEncoder(), StubQueryParser())


# The local backend without a GPU: `uvicorn --factory tests.stubs.server:build_app`.
def build_app() -> FastAPI:
    settings = Settings.from_env()
    return create_app(settings, *stub_components())
