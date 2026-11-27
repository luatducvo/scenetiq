from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LabelledQuery:
    scene_id: str
    query: str
    instance_ids: frozenset[int]


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
