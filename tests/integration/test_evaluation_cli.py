import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scenetiq.evaluation import cli

from .helpers import TOKEN, upload, wait_for_job

BASE = "http://backend"


@pytest.fixture
def scene_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> str:
    body = upload(client)
    assert wait_for_job(client, body["job_id"])["status"] == "succeeded"

    # The CLI's one network call, answered by the in-process API instead of a socket.
    def request(url: str, token: str, payload: dict[str, Any] | None = None) -> Any:
        assert url.startswith(f"{BASE}/"), url
        path = url.removeprefix(BASE)
        headers = {"Authorization": f"Bearer {token}"}
        if payload is None:
            resp = client.get(path, headers=headers)
        else:
            resp = client.post(path, json=payload, headers=headers)
        assert resp.status_code == 200, resp.text
        return resp.json()

    monkeypatch.setattr(cli, "_request", request)
    monkeypatch.setenv("SCENETIQ_TOKEN", TOKEN)
    return body["scene_id"]


def _labels(tmp_path: Path, scene_id: str) -> Path:
    path = tmp_path / "labels.json"
    rows = [
        {"scene_id": scene_id, "query": "red chair", "instance_ids": [0]},
        {"scene_id": scene_id, "query": "wooden thing", "instance_ids": [1]},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_the_cli_prints_each_query_score_and_the_summary(
    scene_id: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # The fake Scene: 0 is a chair, 1 a table, 2 a lamp. "wooden thing" returns the lamp and
    # then the table (worked out in test_query.py); only the table is wooden.
    code = cli.main(["--url", f"{BASE}/", "--labels", str(_labels(tmp_path, scene_id))])
    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        "P=1.00 R=1.00 best=yes  'red chair': wanted [0], returned [0]",
        "P=0.50 R=1.00 best=no   'wooden thing': wanted [1], returned [2, 1]",
        "2 Queries: precision 0.750, recall 1.000, best Match right 0.500",
    ]


def test_the_cli_passes_top_k_to_the_backend(
    scene_id: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    labels = _labels(tmp_path, scene_id)
    assert cli.main(["--url", BASE, "--labels", str(labels), "--top-k", "1"]) == 0
    # One Match each: the chair is right, the lamp alone is wrong for "wooden thing".
    assert capsys.readouterr().out.splitlines()[-1] == (
        "2 Queries: precision 0.500, recall 0.500, best Match right 0.500"
    )


def test_the_cli_lists_a_scenes_instances_with_name_and_centroid(
    scene_id: str, capsys: pytest.CaptureFixture[str]
):
    assert cli.main(["--url", BASE, "--instances", scene_id]) == 0
    # Centroids are the means of the fake points; the lamp has no Caption.
    pad = " " * 2
    assert capsys.readouterr().out.splitlines() == [
        "   0" + pad + "chair" + " " * 25 + pad + "(0.25, 0.50, 0.25)",
        "   1" + pad + "table" + " " * 25 + pad + "(2.50, 0.25, 0.20)",
        "   2" + pad + "(no Caption)" + " " * 18 + pad + "(5.00, 5.00, 1.70)",
    ]


def test_the_cli_refuses_to_run_without_a_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("SCENETIQ_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--url", BASE, "--instances", "abc"])
    assert exit_info.value.code == 2
    assert "SCENETIQ_TOKEN must be set" in capsys.readouterr().err


def test_the_cli_needs_labels_or_a_scene_to_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("SCENETIQ_TOKEN", TOKEN)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--url", BASE])
    assert exit_info.value.code == 2
    assert "give --labels FILE or --instances SCENE_ID" in capsys.readouterr().err
