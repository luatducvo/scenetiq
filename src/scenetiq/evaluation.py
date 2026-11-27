from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelledQuery:
    scene_id: str
    query: str
    instance_ids: frozenset[int]


@dataclass(frozen=True)
class QueryScore:
    precision: float
    recall: float
    best_correct: bool


@dataclass(frozen=True)
class Summary:
    queries: int
    precision: float
    recall: float
    best_correct: float


Ask = Callable[[str, str], list[int]]


def score_query(wanted: frozenset[int] | set[int], returned: Sequence[int]) -> QueryScore:
    if not returned:
        return QueryScore(0.0, 0.0, False)
    right = len(wanted.intersection(returned))
    return QueryScore(right / len(returned), right / len(wanted), returned[0] in wanted)


def summarize(scores: Sequence[QueryScore]) -> Summary:
    if not scores:
        raise ValueError("no labelled Query to summarize")
    count = len(scores)
    return Summary(
        queries=count,
        precision=sum(s.precision for s in scores) / count,
        recall=sum(s.recall for s in scores) / count,
        best_correct=sum(s.best_correct for s in scores) / count,
    )


def load_labels(raw: str) -> list[LabelledQuery]:
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the labels are not valid JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("the labels must be a JSON list of labelled Queries")
    return [_label(row, number) for number, row in enumerate(rows, start=1)]


def _label(row: Any, number: int) -> LabelledQuery:
    if not isinstance(row, dict):
        raise ValueError(f"label {number} is not a JSON object: {row!r}")
    scene_id, query, ids = row.get("scene_id"), row.get("query"), row.get("instance_ids")
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError(f"label {number} has no text scene_id: {scene_id!r}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"label {number} has no query text: {query!r}")
    if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
        raise ValueError(
            f"label {number} needs instance_ids, a non-empty list of integers: {ids!r}"
        )
    return LabelledQuery(scene_id, query.strip(), frozenset(ids))


def evaluate(
    labels: Sequence[LabelledQuery], ask: Ask
) -> list[tuple[LabelledQuery, list[int], QueryScore]]:
    results = []
    for label in labels:
        returned = ask(label.scene_id, label.query)
        results.append((label, returned, score_query(label.instance_ids, returned)))
    return results


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


if __name__ == "__main__":
    raise SystemExit(main())
