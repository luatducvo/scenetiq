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


class JobStatus(StrEnum):
    """Lifecycle of the job that reconstructs a Scene."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class SceneRecord:
    """Metadata of one Scene and the job that processes it."""

    scene_id: str
    job_id: str
    name: str
    created_at: str
    status: JobStatus
    stage: str | None
    error: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    scene_id   TEXT PRIMARY KEY,
    job_id     TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    stage      TEXT,
    error      TEXT
)
"""


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
            self._db.execute(_SCHEMA)

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
            created_at=datetime.now(UTC).isoformat(),
            status=JobStatus.QUEUED,
            stage=None,
            error=None,
        )
        self.views_dir(record.scene_id).mkdir(parents=True)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO scenes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.scene_id,
                    record.job_id,
                    record.name,
                    record.created_at,
                    record.status.value,
                    None,
                    None,
                ),
            )
        return record

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
        """Persist ``reconstruction`` as the result of ``scene_id``."""
        save_reconstruction(self.scene_dir(scene_id) / "reconstruction", reconstruction)

    def reconstruction_dir(self, scene_id: str) -> Path:
        """Directory holding the Reconstruction artifacts of ``scene_id``."""
        return self.scene_dir(scene_id) / "reconstruction"

    def _one(self, sql: str, arg: str) -> SceneRecord | None:
        with self._lock:
            row = self._db.execute(sql, (arg,)).fetchone()
        return _record(row) if row else None


def _record(row: sqlite3.Row) -> SceneRecord:
    return SceneRecord(
        scene_id=row["scene_id"],
        job_id=row["job_id"],
        name=row["name"],
        created_at=row["created_at"],
        status=JobStatus(row["status"]),
        stage=row["stage"],
        error=row["error"],
    )
