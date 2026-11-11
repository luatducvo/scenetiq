import json

import pytest

from scenetiq.evaluation import (
    LabelledQuery,
    QueryScore,
    evaluate,
    load_labels,
    score_query,
    summarize,
)


def test_a_query_is_scored_by_the_share_of_right_matches_and_of_found_instances():
    # Wanted 3, 7 and 9; returned 7, 4, 3, 5: two of four returned are right, two of three
    # wanted were found, and the best Match is a wanted one.
    score = score_query({3, 7, 9}, [7, 4, 3, 5])
    assert (score.precision, score.best_correct) == (0.5, True)
    assert score.recall == pytest.approx(0.666667, abs=1e-6)


def test_a_wrong_best_match_is_reported_even_when_a_wanted_instance_follows():
    assert score_query({3}, [4, 3]) == QueryScore(0.5, 1.0, False)


def test_a_query_without_matches_scores_zero():
    assert score_query({3}, []) == QueryScore(0.0, 0.0, False)


def test_the_summary_averages_over_queries():
    scores = [QueryScore(1.0, 0.5, True), QueryScore(0.5, 1.0, False), QueryScore(0.0, 0.0, False)]
    summary = summarize(scores)
    assert summary.queries == 3
    assert summary.precision == pytest.approx(0.5)
    assert summary.recall == pytest.approx(0.5)
    assert summary.best_correct == pytest.approx(1 / 3)


def test_no_queries_cannot_be_summarized():
    with pytest.raises(ValueError, match="no labelled Query"):
        summarize([])


def test_labels_are_read_from_json():
    raw = json.dumps(
        [
            {"scene_id": "abc", "query": "cái ghế", "instance_ids": [3, 7]},
            {"scene_id": "abc", "query": "lamp", "instance_ids": [2]},
        ]
    )
    assert load_labels(raw) == [
        LabelledQuery("abc", "cái ghế", frozenset({3, 7})),
        LabelledQuery("abc", "lamp", frozenset({2})),
    ]


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("{}", "a JSON list"),
        ("[1]", "label 1"),
        (json.dumps([{"scene_id": "abc", "query": "lamp"}]), "instance_ids"),
        (json.dumps([{"scene_id": "abc", "query": "lamp", "instance_ids": []}]), "instance_ids"),
        (json.dumps([{"scene_id": "abc", "query": "lamp", "instance_ids": ["2"]}]), "instance_ids"),
        (json.dumps([{"scene_id": "abc", "query": " ", "instance_ids": [2]}]), "query"),
        (json.dumps([{"scene_id": 5, "query": "lamp", "instance_ids": [2]}]), "scene_id"),
        ("not json", "not valid JSON"),
    ],
)
def test_unusable_labels_are_refused_with_the_reason(raw: str, reason: str):
    with pytest.raises(ValueError, match=reason):
        load_labels(raw)


def test_every_labelled_query_is_asked_and_scored_in_order():
    labels = [
        LabelledQuery("abc", "chair", frozenset({0})),
        LabelledQuery("xyz", "lamp", frozenset({2, 5})),
    ]
    answers = {("abc", "chair"): [0, 1], ("xyz", "lamp"): [5]}
    results = evaluate(labels, lambda scene_id, query: answers[scene_id, query])
    assert results == [
        (labels[0], [0, 1], QueryScore(0.5, 1.0, True)),
        (labels[1], [5], QueryScore(1.0, 0.5, True)),
    ]
