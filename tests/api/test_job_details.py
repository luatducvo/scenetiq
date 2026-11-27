"""Stage timings and job notes reported by GET /jobs/{job_id}."""

import sqlite3

from scenetiq.domain import Stage
from scenetiq.fakes import FAKE_NOTE, FakePipeline

from .conftest import AUTH, upload, wait_for_job

ALL_STAGES = [s.value for s in Stage]


def test_a_finished_job_lists_every_stage_in_order_back_to_back(client):
    job = wait_for_job(client, upload(client)["job_id"])

    assert job["status"] == "succeeded"
    assert [s["stage"] for s in job["stages"]] == ALL_STAGES
    assert job["stages"][0]["started_at"] >= job["started_at"]
    for current, following in zip(job["stages"], job["stages"][1:], strict=False):
        assert current["finished_at"] == following["started_at"]
    assert job["stages"][-1]["finished_at"] == job["finished_at"]


def test_a_note_is_listed_under_the_stage_that_recorded_it(client):
    job = wait_for_job(client, upload(client)["job_id"])

    assert [(n["stage"], n["message"]) for n in job["notes"]] == [("reconstructing", FAKE_NOTE)]


def test_a_failed_job_records_when_it_ended(make_client):
    client = make_client(pipeline=FakePipeline(error=RuntimeError("out of memory")))
    job = wait_for_job(client, upload(client)["job_id"])

    assert job["status"] == "failed"
    assert [s["stage"] for s in job["stages"]] == ["reconstructing"]
    assert job["finished_at"] is not None
    assert job["stages"][0]["finished_at"] == job["finished_at"]


def test_a_database_from_before_job_timings_still_lists_its_scenes(settings, make_client):
    settings.storage_root.mkdir(parents=True)
    db = sqlite3.connect(settings.storage_root / "scenetiq.db")
    with db:
        db.execute(
            "CREATE TABLE scenes (scene_id TEXT PRIMARY KEY, job_id TEXT UNIQUE NOT NULL,"
            " name TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL,"
            " stage TEXT, error TEXT)"
        )
        db.execute(
            "INSERT INTO scenes VALUES ('a' , 'b', 'Old kitchen',"
            " '2026-01-01T00:00:00+00:00', 'succeeded', 'captioning', NULL)"
        )
    db.close()
    client = make_client()

    [scene] = client.get("/scenes", headers=AUTH).json()
    assert scene["name"] == "Old kitchen"
    job = client.get("/jobs/b", headers=AUTH).json()
    assert (job["started_at"], job["finished_at"], job["stages"]) == (None, None, [])
