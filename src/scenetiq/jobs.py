"""Runs one Scene reconstruction job at a time in a background thread."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

from scenetiq.domain import Capture, Pipeline, Stage
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
        reporter = _JobReporter(store, record)
        try:
            store.update_job(record.job_id, status=JobStatus.RUNNING)
            reconstruction = self._pipeline.run(capture, reporter)
            store.save_reconstruction(record.scene_id, reconstruction)
        except Exception as exc:
            logger.exception("job %s failed", record.job_id)
            store.update_job(record.job_id, status=JobStatus.FAILED, error=str(exc) or repr(exc))
            reporter.report(JobStatus.FAILED)
        else:
            store.update_job(record.job_id, status=JobStatus.SUCCEEDED)
            reporter.report(JobStatus.SUCCEEDED)
        finally:
            self.release()


class _JobReporter:
    """Records what one job's Pipeline reports and forwards its progress to the notebook."""

    def __init__(self, store: SceneStore, record: SceneRecord) -> None:
        self._store = store
        self._job_id = record.job_id
        self._progress = JobProgress(
            job_id=record.job_id,
            scene_id=record.scene_id,
            status=JobStatus.RUNNING.value,
            stage="",
            step=0,
            total=len(_STAGES),
        )

    def stage(self, stage: Stage) -> None:
        self._store.update_job(self._job_id, stage=stage.value)
        self._progress = replace(self._progress, stage=stage.value, step=_STAGES.index(stage) + 1)
        report_progress(self._progress)

    def checkpoint(self) -> None:
        pass

    def note(self, message: str) -> None:
        logger.info("job %s note: %s", self._job_id, message)

    def report(self, status: JobStatus) -> None:
        """Send the job's final ``status`` to the notebook."""
        report_progress(replace(self._progress, status=status.value))


_STAGES = list(Stage)
