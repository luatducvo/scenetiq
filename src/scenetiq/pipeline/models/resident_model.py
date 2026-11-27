import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager


class ResidentModel[T]:
    def __init__(self, load: Callable[[], T]) -> None:
        self._load = load
        self._loaded: T | None = None
        self._lock = threading.Lock()

    @contextmanager
    def use(self) -> Generator[T]:
        with self._lock:
            if self._loaded is None:
                self._loaded = self._load()
            yield self._loaded
