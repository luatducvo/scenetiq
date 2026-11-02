import io
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenetiq.api import create_app
from scenetiq.config import Settings
from scenetiq.fakes import FakePipeline, FakeTextEncoder

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def png_bytes(color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def image_files(n: int, *, ext: str = "png") -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("images", (f"view_{i:03d}.{ext}", png_bytes(), f"image/{ext}")) for i in range(n)]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage_root=tmp_path / "store", token=TOKEN, match_threshold=0.5)


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


def upload(client: TestClient, n: int = 10, name: str = "Kitchen") -> dict:
    resp = client.post("/scenes", data={"name": name}, files=image_files(n), headers=AUTH)
    assert resp.status_code == 201, resp.text
    return resp.json()


def wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")
