import io
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scenetiq.config import Settings
from scenetiq.fakes import FakePipeline
from scenetiq.runtime_log import PROGRESS_LOGGER, JobProgress, UploadProgress, parse_progress

from .conftest import AUTH, TOKEN, post_view, staged_capture, start_capture, upload, wait_for_job


@pytest.mark.parametrize("count", [0, 2001])
def test_capture_outside_1_to_2000_images_is_rejected(client: TestClient, count: int):
    resp = client.post("/captures", json={"view_count": count}, headers=AUTH)
    assert resp.status_code == 422
    assert "1-2000" in resp.json()["detail"]


def test_one_photo_capture_is_accepted(client: TestClient):
    body = upload(client, 1)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"


def test_capture_at_the_default_maximum_is_accepted_without_a_stride(client: TestClient):
    resp = client.post("/captures", json={"view_count": 2000}, headers=AUTH)
    assert resp.status_code == 201


@pytest.mark.parametrize(("count", "expected_status"), [(4, 422), (5, 201), (6, 201), (7, 422)])
def test_configured_view_limits_apply_to_the_uploaded_photos(
    make_client: Callable[..., TestClient], tmp_path: Path, count: int, expected_status: int
):
    settings = Settings(tmp_path / "s", token=TOKEN, min_views=5, max_views=6, view_stride=4)
    client = make_client(settings=settings)
    resp = client.post("/captures", json={"view_count": count}, headers=AUTH)
    assert resp.status_code == expected_status
    if expected_status == 422:
        assert "5-6" in resp.json()["detail"]


def test_view_stride_gives_the_pipeline_every_nth_view_from_the_first(
    make_client: Callable[..., TestClient], pipeline: FakePipeline, tmp_path: Path
):
    client = make_client(settings=Settings(tmp_path / "s", token=TOKEN, view_stride=3))
    body = upload(client, 10)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"
    assert [v.path.stem for v in pipeline.captures[0].views] == ["0000", "0003", "0006", "0009"]


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


def test_multi_picture_jpeg_from_a_phone_is_accepted(client: TestClient):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(
        buf, format="MPO", save_all=True, append_images=[Image.new("RGB", (4, 4))]
    )
    assert Image.open(io.BytesIO(buf.getvalue())).format == "MPO"

    resp = post_view(client, start_capture(client), 0, filename="phone.jpg", data=buf.getvalue())

    assert resp.status_code == 204, resp.text


def test_multi_picture_jpeg_named_png_is_rejected(client: TestClient):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(
        buf, format="MPO", save_all=True, append_images=[Image.new("RGB", (4, 4))]
    )

    resp = post_view(client, start_capture(client), 0, filename="phone.png", data=buf.getvalue())

    assert resp.status_code == 415
    assert "phone.png" in resp.json()["detail"]


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


def _reported_progress(caplog: pytest.LogCaptureFixture) -> list[JobProgress]:
    parsed = [parse_progress(r.getMessage()) for r in caplog.records if r.name == PROGRESS_LOGGER]
    return [p for p in parsed if isinstance(p, JobProgress)]


def _reported_uploads(caplog: pytest.LogCaptureFixture) -> list[UploadProgress]:
    parsed = [parse_progress(r.getMessage()) for r in caplog.records if r.name == PROGRESS_LOGGER]
    return [p for p in parsed if isinstance(p, UploadProgress)]


def test_each_stored_view_reports_the_upload_count_and_a_retry_does_not_add_to_it(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=PROGRESS_LOGGER)
    capture_id = start_capture(client, 10)

    for index in (0, 0, 1):
        assert post_view(client, capture_id, index).status_code == 204

    assert _reported_uploads(caplog) == [
        UploadProgress(capture_id, 1, 10),
        UploadProgress(capture_id, 1, 10),
        UploadProgress(capture_id, 2, 10),
    ]


def test_rejected_view_reports_no_upload_progress(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=PROGRESS_LOGGER)

    resp = post_view(client, start_capture(client), 0, filename="notes.txt", data=b"hello")

    assert resp.status_code == 415
    assert _reported_uploads(caplog) == []


def test_job_reports_every_stage_and_then_success(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=PROGRESS_LOGGER)

    body = upload(client)
    wait_for_job(client, body["job_id"])
    client.app.state.runner.join(5)  # type: ignore[attr-defined]

    assert [(p.status, p.stage, p.step, p.total) for p in _reported_progress(caplog)] == [
        ("running", "reconstructing", 1, 6),
        ("running", "fusing", 2, 6),
        ("running", "superpoints", 3, 6),
        ("running", "masking", 4, 6),
        ("running", "merging", 5, 6),
        ("running", "embedding", 6, 6),
        ("succeeded", "embedding", 6, 6),
    ]
    assert {(p.job_id, p.scene_id) for p in _reported_progress(caplog)} == {
        (body["job_id"], body["scene_id"])
    }


def test_failed_job_reports_the_stage_it_failed_at(
    make_client: Callable[..., TestClient], caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=PROGRESS_LOGGER)
    client = make_client(pipeline=FakePipeline(error=RuntimeError("boom")))

    wait_for_job(client, upload(client)["job_id"])
    client.app.state.runner.join(5)  # type: ignore[attr-defined]

    assert [(p.status, p.stage) for p in _reported_progress(caplog)] == [
        ("running", "reconstructing"),
        ("failed", "reconstructing"),
    ]


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
