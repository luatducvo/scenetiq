import io
import time

from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image

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
