from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from scenetiq.evaluation.labels import LabelledQuery


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


def evaluate(
    labels: Sequence[LabelledQuery], ask: Ask
) -> list[tuple[LabelledQuery, list[int], QueryScore]]:
    results = []
    for label in labels:
        returned = ask(label.scene_id, label.query)
        results.append((label, returned, score_query(label.instance_ids, returned)))
    return results
