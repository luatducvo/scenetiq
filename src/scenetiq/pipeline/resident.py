"""A model that loads on first use, stays loaded, and serves one caller at a time.

Queries and the job run on different threads. Holding the lock for each use, not for a
whole stage, lets another thread use the model between two uses of a stage.
"""

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager


class ResidentModel[T]:
    """Loads with ``load`` on the first :meth:`use` and keeps the result for every later one.

    Args:
        load: Builds the loaded model. Called at most once while it succeeds; if it
            raises, the next :meth:`use` calls it again.
    """

    def __init__(self, load: Callable[[], T]) -> None:
        """Keep ``load``; nothing is loaded until the first :meth:`use`."""
        self._load = load
        self._loaded: T | None = None
        self._lock = threading.Lock()

    @contextmanager
    def use(self) -> Generator[T]:
        """Hold the lock for the enclosed block and yield the model, loading it if needed.

        The lock is not re-entrant: never nest two uses of the same model.
        """
        with self._lock:
            if self._loaded is None:
                self._loaded = self._load()
            yield self._loaded
