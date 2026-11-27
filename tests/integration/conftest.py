from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scenetiq.api.app import create_app
from scenetiq.config import Settings
from scenetiq.fakes import FakeCaptionEncoder, FakePipeline, FakeQueryParser, FakeTextEncoder

from .helpers import TOKEN


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage_root=tmp_path / "store", token=TOKEN)


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def make_client(settings: Settings, pipeline: FakePipeline) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(**overrides: object) -> TestClient:
        kwargs: dict[str, object] = {
            "settings": settings,
            "pipeline": pipeline,
            "text_encoder": FakeTextEncoder(),
            "caption_encoder": FakeCaptionEncoder(),
            "query_parser": FakeQueryParser(),
        }
        kwargs.update(overrides)
        client = TestClient(create_app(**kwargs))  # type: ignore[arg-type]
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
