from collections.abc import Callable

from fastapi.testclient import TestClient

from scenetiq.config import Settings
from scenetiq.storage import JobStatus, SceneStore

from .conftest import AUTH, upload, wait_for_job


def _leave_job_running(settings: Settings) -> str:
    """Simulate a process that died mid-job: the store says running, nothing runs it."""
    store = SceneStore(settings.storage_root)
    record = store.create_scene("Interrupted")
    store.update_job(record.job_id, status=JobStatus.RUNNING, stage="masking")
    store.close()
    return record.job_id


def test_job_left_running_by_a_dead_process_reads_failed_after_restart(
    make_client: Callable[..., TestClient], settings: Settings
):
    job_id = _leave_job_running(settings)

    client = make_client()
    job = client.get(f"/jobs/{job_id}", headers=AUTH).json()

    assert job["status"] == "failed"
    assert job["error"] == "interrupted by restart"
    assert job["stage"] == "masking"


def test_upload_is_accepted_after_restart_recovery(
    make_client: Callable[..., TestClient], settings: Settings
):
    _leave_job_running(settings)
    client = make_client()
    upload(client)


def test_succeeded_scene_survives_restart(make_client: Callable[..., TestClient]):
    first = make_client()
    body = upload(first)
    wait_for_job(first, body["job_id"])
    first.__exit__(None, None, None)

    second = make_client()
    (scene,) = second.get("/scenes", headers=AUTH).json()
    assert scene["status"] == "succeeded"
    assert second.get(f"/scenes/{body['scene_id']}/points", headers=AUTH).status_code == 200
