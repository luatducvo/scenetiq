import threading

import pytest

from scenetiq.pipeline.resident import ResidentModel


class _CountingLoader:
    """Stands in for a model's load: returns a new object and counts its calls."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.calls = 0
        self.gate = gate

    def __call__(self) -> object:
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(timeout=5)
        return object()


def test_constructing_loads_nothing():
    loader = _CountingLoader()
    ResidentModel(loader)
    assert loader.calls == 0


def test_two_threads_that_need_the_model_at_once_cause_one_load():
    gate = threading.Event()
    loader = _CountingLoader(gate)
    resident = ResidentModel(loader)
    seen: list[object] = []

    def use() -> None:
        with resident.use() as model:
            seen.append(model)

    threads = [threading.Thread(target=use) for _ in range(2)]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(timeout=5)

    assert loader.calls == 1
    assert len(seen) == 2
    assert seen[0] is seen[1]


def test_a_use_holds_the_lock_and_another_thread_runs_between_two_uses():
    resident = ResidentModel(_CountingLoader())
    inside, leave, other_done = threading.Event(), threading.Event(), threading.Event()
    order: list[str] = []

    def stage() -> None:
        # Two uses in a row, like two Instances of a Captioning stage.
        with resident.use():
            order.append("first")
            inside.set()
            leave.wait(timeout=5)
        other_done.wait(timeout=5)
        with resident.use():
            order.append("second")

    def query() -> None:
        with resident.use():
            order.append("query")
        other_done.set()

    job = threading.Thread(target=stage)
    job.start()
    inside.wait(timeout=5)
    other = threading.Thread(target=query)
    other.start()
    # While the first use runs, the other thread waits for the lock.
    assert not other_done.wait(timeout=0.2)
    leave.set()
    job.join(timeout=5)
    other.join(timeout=5)

    assert order == ["first", "query", "second"]


def test_a_failed_load_is_tried_again_on_the_next_use():
    calls = 0

    def flaky_load() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("out of memory")
        return "model"

    resident = ResidentModel(flaky_load)
    with pytest.raises(RuntimeError, match="out of memory"), resident.use():
        pass
    with resident.use() as model:
        assert model == "model"
    assert calls == 2


def test_a_failed_use_keeps_the_model_and_frees_the_lock():
    loader = _CountingLoader()
    resident = ResidentModel(loader)
    loaded: list[object] = []

    def failing_generation() -> None:
        with resident.use() as model:
            loaded.append(model)
            raise ValueError("generation failed")

    with pytest.raises(ValueError, match="generation failed"):
        failing_generation()
    first = loaded[0]

    # Another thread can take the lock, and gets the model loaded before the failure.
    seen: list[object] = []

    def use() -> None:
        with resident.use() as again:
            seen.append(again)

    other = threading.Thread(target=use)
    other.start()
    other.join(timeout=5)
    assert not other.is_alive()
    assert seen == [first]
    assert loader.calls == 1
