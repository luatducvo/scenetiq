"""Measure Queries against hand-labelled ground truth, through the HTTP API (ADR 0011).

The Query settings are starting guesses. This tool is how to check them: label, by hand, the
Instances a few Queries should return, then run every Query against a running backend and
read how many Matches were right (precision), how many wanted Instances were found (recall)
and how often the best Match was a wanted one.

Labels are a JSON list, one object per Query::

    [{"scene_id": "0f3a...", "query": "cái ghế", "instance_ids": [3, 7]}]

Run it with ``python -m scenetiq.evaluation --url URL --labels FILE``; the token comes from
``SCENETIQ_TOKEN``. ``--instances SCENE_ID`` lists a Scene's Instance ids with their Names,
to label from. Only the standard library is used, so it runs anywhere the backend is
reachable.
"""

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
    """A Query and the Instances it should return, labelled by hand.

    Attributes:
        scene_id: The Scene the Query is asked against.
        query: The Query text.
        instance_ids: The Instances that fit the Query; never empty.
    """

    scene_id: str
    query: str
    instance_ids: frozenset[int]


@dataclass(frozen=True)
class QueryScore:
    """How well the Matches of one Query fit its label.

    Attributes:
        precision: Share of the Matches that are wanted Instances; 0 without Matches.
        recall: Share of the wanted Instances that are among the Matches.
        best_correct: Whether the best Match is a wanted Instance.
    """

    precision: float
    recall: float
    best_correct: bool


@dataclass(frozen=True)
class Summary:
    """The scores of several Queries, each Query counting once.

    Attributes:
        queries: How many Queries were scored.
        precision: Mean precision.
        recall: Mean recall.
        best_correct: Share of the Queries whose best Match is a wanted Instance.
    """

    queries: int
    precision: float
    recall: float
    best_correct: float


Ask = Callable[[str, str], list[int]]
"""Asks a Query (Scene id, text) and returns the Instance ids of its Matches, best first."""


def score_query(wanted: frozenset[int] | set[int], returned: Sequence[int]) -> QueryScore:
    """Score the Matches of one Query against the Instances it should return.

    Args:
        wanted: The labelled Instance ids; not empty.
        returned: The Instance ids of the Matches, best first.
    """
    if not returned:
        return QueryScore(0.0, 0.0, False)
    right = len(wanted.intersection(returned))
    return QueryScore(right / len(returned), right / len(wanted), returned[0] in wanted)


def summarize(scores: Sequence[QueryScore]) -> Summary:
    """Average the scores of several Queries.

    Raises:
        ValueError: If there is no score to average.
    """
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
    """Read labelled Queries from the JSON text of a labels file.

    Raises:
        ValueError: If the text is not a JSON list of objects with a text ``scene_id``, a
            non-empty text ``query`` and a non-empty list of integer ``instance_ids``.
    """
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
    """Ask every labelled Query and score its Matches.

    Args:
        labels: The labelled Queries.
        ask: Asks one Query against one Scene.

    Returns:
        Each label with the Instance ids returned for it and its score, in label order.
    """
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
    """Run the tool from the command line; see the module docstring.

    Returns:
        The process exit code: 0 after printing the scores or the Instances.
    """
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
