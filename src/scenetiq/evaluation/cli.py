from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scenetiq.evaluation.labels import load_labels
from scenetiq.evaluation.metrics import evaluate, summarize


def _request(url: str, token: str, body: dict[str, Any] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scenetiq.evaluation",
        description="Score Queries against hand-labelled Instances on a running backend.",
    )
    parser.add_argument("--url", required=True, help="base URL of the backend")
    parser.add_argument("--labels", type=Path, help="JSON file of labelled Queries")
    parser.add_argument("--top-k", type=int, help="most Matches per Query (backend default)")
    parser.add_argument("--instances", metavar="SCENE_ID", help="list a Scene's Instances")
    args = parser.parse_args(argv)
    token = os.environ.get("SCENETIQ_TOKEN")
    if not token:
        parser.error("SCENETIQ_TOKEN must be set")
    base = args.url.rstrip("/")

    if args.instances:
        for inst in _request(f"{base}/scenes/{args.instances}/instances", token):
            centre = ", ".join(f"{c:.2f}" for c in inst["centroid"])
            print(f"{inst['instance_id']:>4}  {inst['name'] or '(no Caption)':<30}  ({centre})")
        return 0
    if args.labels is None:
        parser.error("give --labels FILE or --instances SCENE_ID")

    def ask(scene_id: str, query: str) -> list[int]:
        body: dict[str, Any] = {"text": query}
        if args.top_k is not None:
            body["top_k"] = args.top_k
        answer = _request(f"{base}/scenes/{scene_id}/query", token, body)
        return [m["instance_id"] for m in answer["matches"]]

    results = evaluate(load_labels(args.labels.read_text(encoding="utf-8")), ask)
    for label, returned, score in results:
        print(
            f"P={score.precision:.2f} R={score.recall:.2f} "
            f"best={'yes' if score.best_correct else 'no '}  {label.query!r}: "
            f"wanted {sorted(label.instance_ids)}, returned {returned}"
        )
    summary = summarize([score for *_, score in results])
    print(
        f"{summary.queries} Queries: precision {summary.precision:.3f}, "
        f"recall {summary.recall:.3f}, best Match right {summary.best_correct:.3f}"
    )
    return 0
