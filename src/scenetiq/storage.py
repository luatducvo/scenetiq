"""Persistent Scene and job metadata (SQLite) plus per-Scene directories."""

from __future__ import annotations

import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from scenetiq.artifacts import save_reconstruction
from scenetiq.domain import Reconstruction

INTERRUPTED_REASON = "interrupted by restart"
_TEMP_SUFFIX = ".tmp"


class UnknownCaptureError(LookupError):
    """Raised for a Capture that was never started, or was committed or discarded."""


class JobStatus(StrEnum):
    """Lifecycle of the job that reconstructs a Scene."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class SceneRecord:
    """Metadata of one Scene and the job that processes it.

    Attributes:
        scene_id: The Scene.
        job_id: The job that reconstructs it.
        name: The name the user gave it.
        created_at: When it was created, ISO 8601 UTC.
        status: Where its job is in its lifecycle.
        stage: The Stage its job is in or stopped at, if one has started.
        error: Why its job failed, if it did.
        started_at: When its job started running, ISO 8601 UTC.
        finished_at: When its job ended, ISO 8601 UTC; ``None`` while it runs, and for a
            job interrupted by a restart.
        instance_count: How many Instances its Reconstruction has, once saved.
    """

    scene_id: str
    job_id: str
    name: str
    created_at: str
    status: JobStatus
    stage: str | None
    error: str | None
    started_at: str | None = None
    finished_at: str | None = None
    instance_count: int | None = None


class JobEventKind(StrEnum):
    """What a job event records."""

    STAGE = "stage"
    NOTE = "note"


@dataclass(frozen=True)
class JobEvent:
    """One thing a job reported, in the order reported.

    Attributes:
        at: When, ISO 8601 UTC.
        kind: A Stage starting, or a note.
        stage: The Stage that started, or the Stage running when the note was recorded.
        message: The note; empty for a Stage event.
    """

    at: str
    kind: JobEventKind
    stage: str | None
    message: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    scene_id       TEXT PRIMARY KEY,
    job_id         TEXT UNIQUE NOT NULL,
    name           TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL,
    stage          TEXT,
    error          TEXT,
    started_at     TEXT,
    finished_at    TEXT,
    instance_count INTEGER
);
CREATE TABLE IF NOT EXISTS job_events (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL,
    at      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    stage   TEXT,
    message TEXT NOT NULL
);
"""

# Columns added after the first schema, so that an older database can be brought up to date.
_ADDED_SCENE_COLUMNS = {
    "started_at": "TEXT",
    "finished_at": "TEXT",
    "instance_count": "INTEGER",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SceneStore:
    """Stores Scene metadata in SQLite and Scene data in one directory per Scene.

    Args:
        root: Storage root; holds ``scenetiq.db`` and ``scenes/<scene_id>/``.
    """

    def __init__(self, root: Path) -> None:
        """Open (or create) the store under ``root``."""
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(root / "scenetiq.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)
            present = {row["name"] for row in self._db.execute("PRAGMA table_info(scenes)")}
            for column, kind in _ADDED_SCENE_COLUMNS.items():
                if column not in present:
                    self._db.execute(f"ALTER TABLE scenes ADD COLUMN {column} {kind}")

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def scene_dir(self, scene_id: str) -> Path:
        """Directory holding the data of ``scene_id``."""
        return self.root / "scenes" / scene_id

    def views_dir(self, scene_id: str) -> Path:
        """Directory holding the uploaded View images of ``scene_id``."""
        return self.scene_dir(scene_id) / "views"

    def create_scene(self, name: str) -> SceneRecord:
        """Register a new Scene with a queued job and create its directory."""
        record = SceneRecord(
            scene_id=uuid.uuid4().hex,
            job_id=uuid.uuid4().hex,
            name=name,
            created_at=_now(),
            status=JobStatus.QUEUED,
            stage=None,
            error=None,
        )
        self.views_dir(record.scene_id).mkdir(parents=True)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO scenes (scene_id, job_id, name, created_at, status)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    record.scene_id,
                    record.job_id,
                    record.name,
                    record.created_at,
                    record.status.value,
                ),
            )
        return record

    def start_capture(self, view_count: int) -> str:
        """Stage a new Capture of ``view_count`` Views and return its id.

        Only one Capture is staged at a time: an earlier unfinished one is discarded.
        """
        shutil.rmtree(self._captures_dir(), ignore_errors=True)
        capture_id = uuid.uuid4().hex
        directory = self._capture_dir(capture_id)
        (directory / "views").mkdir(parents=True)
        (directory / "view_count").write_text(str(view_count))
        return capture_id

    def capture_view_count(self, capture_id: str) -> int:
        """Return how many Views the staged Capture was started with.

        Raises:
            UnknownCaptureError: If ``capture_id`` is not staged.
        """
        try:
            return int((self._capture_dir(capture_id) / "view_count").read_text())
        except FileNotFoundError as exc:
            raise UnknownCaptureError(capture_id) from exc

    def save_view(self, capture_id: str, index: int, suffix: str, data: bytes) -> None:
        """Store one View of a staged Capture, replacing any earlier copy of ``index``.

        Replacing makes a retried request safe. The write is atomic, so an interrupted
        request never counts as a received View.

        Raises:
            UnknownCaptureError: If ``capture_id`` is not staged.
        """
        views = self._capture_dir(capture_id) / "views"
        try:
            temp = views / f"{uuid.uuid4().hex}{_TEMP_SUFFIX}"
            temp.write_bytes(data)
            for earlier in views.glob(f"{index:04d}.*"):
                earlier.unlink(missing_ok=True)
            temp.replace(views / f"{index:04d}{suffix}")
        except FileNotFoundError as exc:
            raise UnknownCaptureError(capture_id) from exc

    def received_view_count(self, capture_id: str) -> int:
        """Return how many Views of the staged Capture have been stored.

        Raises:
            UnknownCaptureError: If ``capture_id`` is not staged.
        """
        directory = self._capture_dir(capture_id) / "views"
        if not directory.is_dir():
            raise UnknownCaptureError(capture_id)
        return len(_view_paths(directory))

    def create_scene_from_capture(
        self, name: str, capture_id: str
    ) -> tuple[SceneRecord, list[Path]]:
        """Register a new Scene and move the staged Capture's Views into it.

        Returns:
            The Scene's record and the final paths of its Views, in index order.

        Raises:
            UnknownCaptureError: If ``capture_id`` is not staged.
        """
        staged = self._capture_dir(capture_id) / "views"
        if not staged.is_dir():
            raise UnknownCaptureError(capture_id)
        record = self.create_scene(name)
        target = self.views_dir(record.scene_id)
        try:
            target.rmdir()
            shutil.move(staged, target)
        except BaseException:
            self.delete_scene(record.scene_id)
            raise
        shutil.rmtree(self._capture_dir(capture_id), ignore_errors=True)
        return record, _view_paths(target)

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update the status, stage and/or error of ``job_id`` (``None`` keeps a field)."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE scenes SET status = COALESCE(?, status), stage = COALESCE(?, stage),"
                " error = COALESCE(?, error) WHERE job_id = ?",
                (status.value if status else None, stage, error, job_id),
            )

    def start_job(self, job_id: str) -> None:
        """Mark ``job_id`` running and record when it started."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE scenes SET status = ?, started_at = ? WHERE job_id = ?",
                (JobStatus.RUNNING.value, _now(), job_id),
            )

    def finish_job(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        """Record that ``job_id`` ended with ``status`` (and ``error``, if it failed)."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE scenes SET status = ?, error = COALESCE(?, error), finished_at = ?"
                " WHERE job_id = ?",
                (status.value, error, _now(), job_id),
            )

    def start_stage(self, job_id: str, stage: str) -> None:
        """Record that ``job_id`` entered ``stage``."""
        with self._lock, self._db:
            self._db.execute("UPDATE scenes SET stage = ? WHERE job_id = ?", (stage, job_id))
            self._db.execute(
                "INSERT INTO job_events (job_id, at, kind, stage, message) VALUES (?, ?, ?, ?, ?)",
                (job_id, _now(), JobEventKind.STAGE.value, stage, ""),
            )

    def add_note(self, job_id: str, message: str) -> None:
        """Record a note of ``job_id``, under the Stage it is in."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO job_events (job_id, at, kind, stage, message)"
                " SELECT job_id, ?, ?, stage, ? FROM scenes WHERE job_id = ?",
                (_now(), JobEventKind.NOTE.value, message, job_id),
            )

    def job_events(self, job_id: str) -> list[JobEvent]:
        """Return everything ``job_id`` reported, in the order reported."""
        with self._lock:
            rows = self._db.execute(
                "SELECT at, kind, stage, message FROM job_events WHERE job_id = ? ORDER BY seq",
                (job_id,),
            ).fetchall()
        return [
            JobEvent(
                at=r["at"], kind=JobEventKind(r["kind"]), stage=r["stage"], message=r["message"]
            )
            for r in rows
        ]

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        """Return the Scene with ``scene_id``, or ``None``."""
        return self._one("SELECT * FROM scenes WHERE scene_id = ?", scene_id)

    def get_job(self, job_id: str) -> SceneRecord | None:
        """Return the Scene whose job is ``job_id``, or ``None``."""
        return self._one("SELECT * FROM scenes WHERE job_id = ?", job_id)

    def list_scenes(self) -> list[SceneRecord]:
        """Return every Scene, newest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM scenes ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [_record(r) for r in rows]

    def delete_scene(self, scene_id: str) -> None:
        """Remove ``scene_id``'s metadata and its directory."""
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM job_events WHERE job_id IN"
                " (SELECT job_id FROM scenes WHERE scene_id = ?)",
                (scene_id,),
            )
            self._db.execute("DELETE FROM scenes WHERE scene_id = ?", (scene_id,))
        shutil.rmtree(self.scene_dir(scene_id), ignore_errors=True)

    def fail_unfinished_jobs(self) -> int:
        """Mark every queued or running job failed as interrupted; return how many."""
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE scenes SET status = ?, error = ? WHERE status IN (?, ?)",
                (
                    JobStatus.FAILED.value,
                    INTERRUPTED_REASON,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                ),
            )
        return cur.rowcount

    def save_reconstruction(self, scene_id: str, reconstruction: Reconstruction) -> None:
        """Persist ``reconstruction`` as the result of ``scene_id`` and count its Instances."""
        save_reconstruction(self.scene_dir(scene_id) / "reconstruction", reconstruction)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE scenes SET instance_count = ? WHERE scene_id = ?",
                (len(reconstruction.instances), scene_id),
            )

    def reconstruction_dir(self, scene_id: str) -> Path:
        """Directory holding the Reconstruction artifacts of ``scene_id``."""
        return self.scene_dir(scene_id) / "reconstruction"

    def _captures_dir(self) -> Path:
        return self.root / "captures"

    def _capture_dir(self, capture_id: str) -> Path:
        return self._captures_dir() / capture_id

    def _one(self, sql: str, arg: str) -> SceneRecord | None:
        with self._lock:
            row = self._db.execute(sql, (arg,)).fetchone()
        return _record(row) if row else None


def _view_paths(directory: Path) -> list[Path]:
    views = (p for p in directory.iterdir() if p.suffix != _TEMP_SUFFIX)
    return sorted(views, key=lambda p: int(p.stem))  # by name, View 10000 would precede 9999


def _record(row: sqlite3.Row) -> SceneRecord:
    return SceneRecord(
        scene_id=row["scene_id"],
        job_id=row["job_id"],
        name=row["name"],
        created_at=row["created_at"],
        status=JobStatus(row["status"]),
        stage=row["stage"],
        error=row["error"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        instance_count=row["instance_count"],
    )
