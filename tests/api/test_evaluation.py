import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scenetiq.evaluation import evaluate, load_labels, summarize

from .conftest import AUTH, upload, wait_for_job


def test_labelled_queries_are_scored_against_the_query_endpoint(client: TestClient, tmp_path: Path):
    body = upload(client)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"
    scene = body["scene_id"]
    # The fake Scene: 0 is a chair, 1 a table, 2 a lamp. "wooden thing" returns the lamp
    # and then the table (worked out in test_query.py); only the table is wooden.
    labels = load_labels(
        json.dumps(
            [
                {"scene_id": scene, "query": "red chair", "instance_ids": [0]},
                {"scene_id": scene, "query": "wooden thing", "instance_ids": [1]},
            ]
        )
    )

    def ask(scene_id: str, query: str) -> list[int]:
        resp = client.post(f"/scenes/{scene_id}/query", json={"text": query}, headers=AUTH)
        return [m["instance_id"] for m in resp.json()["matches"]]

    results = evaluate(labels, ask)

    assert [returned for _, returned, _ in results] == [[0], [2, 1]]
    summary = summarize([score for *_, score in results])
    # Precision 1 and 1/2, recall 1 and 1, best Match right for the first Query only.
    assert summary.precision == pytest.approx(0.75)
    assert summary.recall == pytest.approx(1.0)
    assert summary.best_correct == pytest.approx(0.5)
