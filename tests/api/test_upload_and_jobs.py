import io
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenetiq.fakes import FakePipeline

from .conftest import AUTH, post_view, staged_capture, start_capture, upload, wait_for_job


@pytest.mark.parametrize("count", [9, 301])
def test_capture_outside_10_to_300_images_is_rejected(client: TestClient, count: int):
    resp = client.post("/captures", json={"view_count": count}, headers=AUTH)
    assert resp.status_code == 422
    assert "10-300" in resp.json()["detail"]


@pytest.mark.parametrize("count", [10, 300])
def test_capture_at_the_limits_is_accepted(client: TestClient, count: int):
    body = upload(client, count)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"


def test_non_image_file_is_rejected_with_its_name(client: TestClient):
    resp = post_view(client, start_capture(client), 0, filename="notes.txt", data=b"hello")
    assert resp.status_code == 415
    assert "notes.txt" in resp.json()["detail"]


def test_file_whose_content_does_not_match_its_extension_is_rejected(client: TestClient):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="BMP")
    resp = post_view(client, start_capture(client), 0, filename="fake.jpg", data=buf.getvalue())
    assert resp.status_code == 415
    assert "fake.jpg" in resp.json()["detail"]


def test_capture_missing_a_view_creates_no_scene(client: TestClient):
    capture_id = start_capture(client, 10)
    for i in range(9):
        post_view(client, capture_id, i)

    resp = client.post("/scenes", json={"name": "x", "capture_id": capture_id}, headers=AUTH)

    assert resp.status_code == 422
    assert "9 of this Capture's 10" in resp.json()["detail"]
    assert client.get("/scenes", headers=AUTH).json() == []


def test_retried_view_is_counted_once_even_with_another_extension(
    client: TestClient, pipeline: FakePipeline
):
    capture_id = start_capture(client, 10)
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="JPEG")
    assert (
        post_view(client, capture_id, 0, filename="a.jpg", data=buf.getvalue()).status_code == 204
    )
    for i in range(10):
        assert post_view(client, capture_id, i).status_code == 204

    resp = client.post("/scenes", json={"name": "x", "capture_id": capture_id}, headers=AUTH)

    assert resp.status_code == 201
    wait_for_job(client, resp.json()["job_id"])
    (capture,) = pipeline.captures
    assert [v.path.name for v in capture.views] == [f"{i:04d}.png" for i in range(10)]


def test_views_sent_concurrently_and_out_of_order_all_reach_the_pipeline(
    client: TestClient, pipeline: FakePipeline
):
    capture_id = start_capture(client, 30)
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(
            pool.map(lambda i: post_view(client, capture_id, i).status_code, range(29, -1, -1))
        )
    assert set(codes) == {204}

    resp = client.post("/scenes", json={"name": "x", "capture_id": capture_id}, headers=AUTH)

    assert resp.status_code == 201
    wait_for_job(client, resp.json()["job_id"])
    (capture,) = pipeline.captures
    assert len(capture.views) == 30
    assert all(v.path.exists() for v in capture.views)


def test_view_index_outside_the_capture_is_rejected(client: TestClient):
    resp = post_view(client, start_capture(client, 10), 10)
    assert resp.status_code == 422
    assert "index 10" in resp.json()["detail"]


def test_view_for_unknown_capture_is_404(client: TestClient):
    assert post_view(client, "0" * 32, 0).status_code == 404


def test_capture_id_that_is_not_an_id_is_rejected(client: TestClient):
    assert post_view(client, "..", 0).status_code in (404, 422)
    resp = client.post("/scenes", json={"name": "x", "capture_id": "../scenes"}, headers=AUTH)
    assert resp.status_code == 422


def test_starting_a_capture_discards_an_unfinished_one(client: TestClient):
    earlier = start_capture(client)
    start_capture(client)
    assert post_view(client, earlier, 0).status_code == 404


def test_committed_capture_cannot_create_a_second_scene(client: TestClient):
    capture_id = staged_capture(client)
    payload = {"name": "x", "capture_id": capture_id}
    first = client.post("/scenes", json=payload, headers=AUTH)
    assert first.status_code == 201
    wait_for_job(client, first.json()["job_id"])

    assert client.post("/scenes", json=payload, headers=AUTH).status_code == 404


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


def test_capture_is_refused_before_any_view_is_sent_while_a_job_runs(
    make_client: Callable[..., TestClient],
):
    gate = threading.Event()
    client = make_client(pipeline=FakePipeline(gate=gate))
    first = upload(client)

    resp = client.post("/captures", json={"view_count": 10}, headers=AUTH)
    assert resp.status_code == 409

    gate.set()
    wait_for_job(client, first["job_id"])
    upload(client)


def test_scene_is_refused_while_the_job_slot_is_taken_and_its_capture_survives(client: TestClient):
    capture_id = staged_capture(client)
    payload = {"name": "y", "capture_id": capture_id}
    client.app.state.runner.reserve()  # type: ignore[attr-defined]

    assert client.post("/scenes", json=payload, headers=AUTH).status_code == 409

    client.app.state.runner.release()  # type: ignore[attr-defined]
    assert client.post("/scenes", json=payload, headers=AUTH).status_code == 201


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/jobs/nope", headers=AUTH).status_code == 404


def test_new_scene_is_listed_with_its_name(client: TestClient):
    body = upload(client, name="Living room")
    wait_for_job(client, body["job_id"])
    (scene,) = client.get("/scenes", headers=AUTH).json()
    assert scene["scene_id"] == body["scene_id"]
    assert scene["name"] == "Living room"
    assert scene["status"] == "succeeded"
