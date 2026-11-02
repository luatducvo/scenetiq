import io
import threading
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenetiq.fakes import FakePipeline

from .conftest import AUTH, image_files, upload, wait_for_job


@pytest.mark.parametrize("count", [9, 301])
def test_capture_outside_10_to_300_images_is_rejected(client: TestClient, count: int):
    resp = client.post("/scenes", data={"name": "x"}, files=image_files(count), headers=AUTH)
    assert resp.status_code == 422
    assert "10-300" in resp.json()["detail"]


@pytest.mark.parametrize("count", [10, 300])
def test_capture_at_the_limits_is_accepted(client: TestClient, count: int):
    body = upload(client, count)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"


def test_non_image_file_is_rejected_with_its_name(client: TestClient):
    files = image_files(9)
    files.append(("images", ("notes.txt", b"hello", "text/plain")))
    resp = client.post("/scenes", data={"name": "x"}, files=files, headers=AUTH)
    assert resp.status_code == 415
    assert "notes.txt" in resp.json()["detail"]


def test_file_whose_content_does_not_match_its_extension_is_rejected(client: TestClient):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="BMP")
    files = image_files(9)
    files.append(("images", ("fake.jpg", buf.getvalue(), "image/jpeg")))
    resp = client.post("/scenes", data={"name": "x"}, files=files, headers=AUTH)
    assert resp.status_code == 415
    assert "fake.jpg" in resp.json()["detail"]


def test_rejected_upload_creates_no_scene(client: TestClient):
    client.post("/scenes", data={"name": "x"}, files=image_files(3), headers=AUTH)
    assert client.get("/scenes", headers=AUTH).json() == []


def test_job_succeeds_after_reporting_every_stage(
    make_client: Callable[..., TestClient],
):
    seen: list[str] = []
    gate = threading.Event()
    pipeline = FakePipeline(gate=gate)
    client = make_client(pipeline=pipeline)

    body = upload(client)
    running = client.get(f"/jobs/{body['job_id']}", headers=AUTH).json()
    seen.append(running["status"])
    assert running["stage"] in (None, "reconstructing")
    gate.set()

    done = wait_for_job(client, body["job_id"])
    assert done["status"] == "succeeded"
    assert done["stage"] == "embedding"
    assert done["error"] is None
    assert done["scene_id"] == body["scene_id"]
    assert seen[0] in ("queued", "running")


def test_pipeline_receives_every_uploaded_view(client: TestClient, pipeline: FakePipeline):
    body = upload(client, 12)
    wait_for_job(client, body["job_id"])
    (capture,) = pipeline.captures
    assert len(capture.views) == 12
    assert all(v.path.exists() for v in capture.views)


def test_failing_pipeline_marks_job_failed_with_reason(
    make_client: Callable[..., TestClient],
):
    client = make_client(pipeline=FakePipeline(error=RuntimeError("DA3 ran out of memory")))
    body = upload(client)
    done = wait_for_job(client, body["job_id"])
    assert done["status"] == "failed"
    assert done["error"] == "DA3 ran out of memory"
    assert done["stage"] == "reconstructing"


def test_upload_while_a_job_runs_is_refused(make_client: Callable[..., TestClient]):
    gate = threading.Event()
    client = make_client(pipeline=FakePipeline(gate=gate))
    first = upload(client)

    resp = client.post("/scenes", data={"name": "y"}, files=image_files(10), headers=AUTH)
    assert resp.status_code == 409

    gate.set()
    wait_for_job(client, first["job_id"])
    assert (
        client.post("/scenes", data={"name": "z"}, files=image_files(10), headers=AUTH).status_code
        == 201
    )


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/jobs/nope", headers=AUTH).status_code == 404


def test_new_scene_is_listed_with_its_name(client: TestClient):
    body = upload(client, name="Living room")
    wait_for_job(client, body["job_id"])
    (scene,) = client.get("/scenes", headers=AUTH).json()
    assert scene["scene_id"] == body["scene_id"]
    assert scene["name"] == "Living room"
    assert scene["status"] == "succeeded"
