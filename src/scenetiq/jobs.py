"""Runs one Scene reconstruction job at a time in a background thread."""

from __future__ import annotations

import logging
import threading

from scenetiq.domain import Capture, Pipeline, Stage
from scenetiq.storage import JobStatus, SceneRecord, SceneStore

logger = logging.getLogger(__name__)


class JobBusyError(RuntimeError):
    """Raised when a job is submitted while another one is still running."""


class JobRunner:
    """Runs the Pipeline for one Scene at a time and records progress in the store.

    A caller first claims the single slot with :meth:`reserve`, then either
    :meth:`start` s a job (the slot frees itself when the job ends) or
    :meth:`release` s it.

    Args:
        store: Where job status and results are persisted.
        pipeline: The Pipeline that turns a Capture into a Reconstruction.
    """

    def __init__(self, store: SceneStore, pipeline: Pipeline) -> None:
        """Create an idle runner."""
        self._store = store
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        """Whether the slot is claimed or a job is running."""
        with self._lock:
            return self._active

    def reserve(self) -> None:
        """Claim the single job slot.

        Raises:
            JobBusyError: If the slot is already claimed.
        """
        with self._lock:
            if self._active:
                raise JobBusyError("another job is running")
            self._active = True

    def release(self) -> None:
        """Free a slot claimed with :meth:`reserve` without starting a job."""
        with self._lock:
            self._active = False

    def start(self, record: SceneRecord, capture: Capture) -> None:
        """Process ``record``'s Capture in the background; requires a prior :meth:`reserve`."""
        thread = threading.Thread(
            target=self._run, args=(record, capture), name=f"job-{record.job_id}", daemon=True
        )
        with self._lock:
            self._thread = thread
        thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the most recently started job, if any, to finish."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, record: SceneRecord, capture: Capture) -> None:
        store = self._store

        def on_stage(stage: Stage) -> None:
            store.update_job(record.job_id, stage=stage.value)

        try:
            store.update_job(record.job_id, status=JobStatus.RUNNING)
            reconstruction = self._pipeline.run(capture, on_stage)
            store.save_reconstruction(record.scene_id, reconstruction)
        except Exception as exc:
            logger.exception("job %s failed", record.job_id)
            store.update_job(record.job_id, status=JobStatus.FAILED, error=str(exc) or repr(exc))
        else:
            store.update_job(record.job_id, status=JobStatus.SUCCEEDED)
        finally:
            self.release()
