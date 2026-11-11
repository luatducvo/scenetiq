import io
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image

from scenetiq.api import create_app
from scenetiq.config import Settings
from scenetiq.fakes import FakeCaptionEncoder, FakePipeline, FakeQueryParser, FakeTextEncoder

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def png_bytes(color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def start_capture(client: TestClient, view_count: int = 10) -> str:
    resp = client.post("/captures", json={"view_count": view_count}, headers=AUTH)
    assert resp.status_code == 201, resp.text
    return resp.json()["capture_id"]


def post_view(
    client: TestClient,
    capture_id: str,
    index: int,
    *,
    filename: str = "view.png",
    data: bytes | None = None,
) -> Response:
    return client.post(
        f"/captures/{capture_id}/views",
        data={"index": str(index)},
        files={"image": (filename, png_bytes() if data is None else data)},
        headers=AUTH,
    )


def staged_capture(client: TestClient, n: int = 10) -> str:
    capture_id = start_capture(client, n)
    for i in range(n):
        resp = post_view(client, capture_id, i)
        assert resp.status_code == 204, resp.text
    return capture_id


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


def upload(client: TestClient, n: int = 10, name: str = "Kitchen") -> dict:
    capture_id = staged_capture(client, n)
    resp = client.post("/scenes", json={"name": name, "capture_id": capture_id}, headers=AUTH)
    assert resp.status_code == 201, resp.text
    return resp.json()


def wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")
