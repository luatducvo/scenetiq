"""Runs one Scene reconstruction job at a time in a background thread."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

from scenetiq.domain import Capture, JobCancelledError, Pipeline, Stage
from scenetiq.runtime_log import JobProgress, report_progress
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
        self._job_id: str | None = None
        self._cancel = threading.Event()

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
            self._job_id = record.job_id
            self._cancel = threading.Event()
        thread.start()

    def cancel(self, job_id: str) -> bool:
        """Ask job ``job_id`` to stop at its next checkpoint.

        Returns:
            Whether ``job_id`` will stop: it holds the slot, or it was already asked to stop
            (also in the moment between freeing the slot and publishing its final status).
        """
        with self._lock:
            if self._job_id != job_id:
                return False
            if self._cancel.is_set():
                return True
            if not self._active:
                return False
            self._cancel.set()
            return True

    def cancel_requested(self, job_id: str) -> bool:
        """Whether job ``job_id`` is the latest job and was asked to stop."""
        with self._lock:
            return self._job_id == job_id and self._cancel.is_set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the most recently started job, if any, to finish."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, record: SceneRecord, capture: Capture) -> None:
        store = self._store
        with self._lock:
            cancel = self._cancel
        reporter = _JobReporter(store, record, cancel)
        outcome, error = JobStatus.FAILED, "job ended unexpectedly"
        try:
            store.start_job(record.job_id)
            reconstruction = self._pipeline.run(capture, reporter)
            # A cancellation during the last Stage still wins; once saving starts, it cannot.
            reporter.checkpoint()
            store.save_reconstruction(record.scene_id, reconstruction)
            outcome, error = JobStatus.SUCCEEDED, None
        except JobCancelledError:
            logger.info("job %s cancelled", record.job_id)
            store.discard_scene_data(record.scene_id)
            outcome, error = JobStatus.CANCELLED, None
        except Exception as exc:
            logger.exception("job %s failed", record.job_id)
            error = str(exc) or repr(exc)
        finally:
            # A client that sees the final status may start the next job at once: free the
            # slot first, and publish the status last.
            self.release()
            store.finish_job(record.job_id, outcome, error=error)
            reporter.report(outcome)


class _JobReporter:
    """Records what one job's Pipeline reports and forwards its progress to the notebook."""

    def __init__(self, store: SceneStore, record: SceneRecord, cancel: threading.Event) -> None:
        self._store = store
        self._job_id = record.job_id
        self._cancel = cancel
        self._progress = JobProgress(
            job_id=record.job_id,
            scene_id=record.scene_id,
            status=JobStatus.RUNNING.value,
            stage="",
            step=0,
            total=len(_STAGES),
        )

    def stage(self, stage: Stage) -> None:
        self.checkpoint()
        self._store.start_stage(self._job_id, stage.value)
        self._progress = replace(self._progress, stage=stage.value, step=_STAGES.index(stage) + 1)
        report_progress(self._progress)

    def checkpoint(self) -> None:
        if self._cancel.is_set():
            raise JobCancelledError(self._job_id)

    def note(self, message: str) -> None:
        logger.info("job %s note: %s", self._job_id, message)
        self._store.add_note(self._job_id, message)

    def report(self, status: JobStatus) -> None:
        """Send the job's final ``status`` to the notebook."""
        report_progress(replace(self._progress, status=status.value))


_STAGES = list(Stage)
