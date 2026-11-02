"""HTTP API: upload Captures, track jobs, fetch Reconstructions and run Queries."""

import io
import logging
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from scenetiq.artifacts import INSTANCE_IDS_FILE, POINTS_FILE, load_instances
from scenetiq.config import Settings
from scenetiq.domain import Capture, Pipeline, TextEncoder, View
from scenetiq.jobs import JobBusyError, JobRunner
from scenetiq.query import MissingEmbeddingsError, run_query
from scenetiq.storage import JobStatus, SceneRecord, SceneStore

logger = logging.getLogger(__name__)

MIN_VIEWS = 10
MAX_VIEWS = 300
_ALLOWED_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


def create_app(settings: Settings, pipeline: Pipeline, text_encoder: TextEncoder) -> FastAPI:
    """Build the SceneTIQ API.

    Args:
        settings: Storage root, shared token, Match threshold and CORS origins.
        pipeline: Turns an uploaded Capture into a Reconstruction.
        text_encoder: Encodes Query text for matching against Instances.

    Returns:
        A FastAPI app; every route requires ``Authorization: Bearer <token>``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        store = SceneStore(settings.storage_root)
        interrupted = store.fail_unfinished_jobs()
        if interrupted:
            logger.warning("marked %d unfinished job(s) failed after restart", interrupted)
        app.state.store = store
        app.state.runner = JobRunner(store, pipeline)
        yield
        app.state.runner.join(timeout=5)
        store.close()

    bearer = HTTPBearer(auto_error=False)

    def require_token(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if creds is None or not secrets.compare_digest(creds.credentials, settings.token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid token")

    app = FastAPI(title="SceneTIQ", lifespan=lifespan, dependencies=[Depends(require_token)])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.text_encoder = text_encoder

    @app.post("/scenes", status_code=status.HTTP_201_CREATED)
    def create_scene(
        request: Request,
        name: Annotated[str, Form(min_length=1, max_length=200)],
        images: Annotated[list[UploadFile], File()],
    ) -> dict[str, str]:
        """Upload a Capture for a new Scene and start reconstructing it."""
        if not MIN_VIEWS <= len(images) <= MAX_VIEWS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"a Capture needs {MIN_VIEWS}-{MAX_VIEWS} images, got {len(images)}",
            )
        payloads = [(f, f.file.read()) for f in images]
        for upload, data in payloads:
            _validate_image(upload.filename or "", data)

        runner: JobRunner = request.app.state.runner
        store: SceneStore = request.app.state.store
        try:
            runner.reserve()
        except JobBusyError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        try:
            record = store.create_scene(name)
            views_dir = store.views_dir(record.scene_id)
            views = []
            for i, (upload, data) in enumerate(payloads):
                path = views_dir / f"{i:04d}{PurePath(upload.filename or '').suffix.lower()}"
                path.write_bytes(data)
                views.append(View(path=path))
        except BaseException:
            runner.release()
            raise
        runner.start(record, Capture(views=views))
        return {"scene_id": record.scene_id, "job_id": record.job_id}

    @app.get("/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict[str, str | None]:
        """Report a job's status, current stage and error."""
        record = request.app.state.store.get_job(job_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        return {
            "job_id": record.job_id,
            "scene_id": record.scene_id,
            "status": record.status.value,
            "stage": record.stage,
            "error": record.error,
        }

    @app.get("/scenes")
    def list_scenes(request: Request) -> list[dict[str, str]]:
        """List every Scene, newest first."""
        return [_scene_summary(r) for r in request.app.state.store.list_scenes()]

    @app.delete("/scenes/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_scene(request: Request, scene_id: str) -> None:
        """Delete a Scene and all its data; refused while its job is unfinished."""
        store: SceneStore = request.app.state.store
        record = store.get_scene(scene_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown Scene")
        if record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise HTTPException(status.HTTP_409_CONFLICT, "Scene job still running")
        store.delete_scene(scene_id)

    @app.get("/scenes/{scene_id}/points")
    def get_points(request: Request, scene_id: str) -> FileResponse:
        """Return the Reconstruction's point cloud as binary PLY (xyz metres, rgb)."""
        directory = _ready_reconstruction_dir(request.app.state.store, scene_id)
        return FileResponse(directory / POINTS_FILE, media_type="application/octet-stream")

    @app.get("/scenes/{scene_id}/instance-ids")
    def get_instance_ids(request: Request, scene_id: str) -> FileResponse:
        """Return the Instance id of every point (little-endian int32, PLY order, -1 = none)."""
        directory = _ready_reconstruction_dir(request.app.state.store, scene_id)
        return FileResponse(directory / INSTANCE_IDS_FILE, media_type="application/octet-stream")

    @app.get("/scenes/{scene_id}/instances")
    def get_instances(request: Request, scene_id: str) -> list[InstanceOut]:
        """List the Scene's Instances with centroid and bounding box in metres."""
        directory = _ready_reconstruction_dir(request.app.state.store, scene_id)
        return [
            InstanceOut(
                instance_id=inst.instance_id,
                centroid=inst.centroid.tolist(),
                bbox_min=inst.bbox_min.tolist(),
                bbox_max=inst.bbox_max.tolist(),
            )
            for inst in load_instances(directory)
        ]

    @app.post("/scenes/{scene_id}/query")
    def query_scene(request: Request, scene_id: str, body: QueryIn) -> QueryOut:
        """Find the Instances that match a free-text Query."""
        directory = _ready_reconstruction_dir(request.app.state.store, scene_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Query text is empty")
        try:
            result = run_query(
                load_instances(directory),
                text_encoder.encode(text),
                settings.match_threshold,
                text_encoder.to_score,
            )
        except MissingEmbeddingsError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return QueryOut(
            matches=[MatchOut(instance_id=m.instance_id, score=m.score) for m in result.matches],
            low_confidence=result.low_confidence,
        )

    return app


class InstanceOut(BaseModel):
    """An Instance's position and extent, in metres."""

    instance_id: int
    centroid: list[float]
    bbox_min: list[float]
    bbox_max: list[float]


class QueryIn(BaseModel):
    """A free-text Query."""

    text: str = Field(max_length=500)


class MatchOut(BaseModel):
    """One Match: an Instance id and its relevance score."""

    instance_id: int
    score: float


class QueryOut(BaseModel):
    """The Matches for a Query, flagged when only a Low-confidence Match was found."""

    matches: list[MatchOut]
    low_confidence: bool


def _ready_reconstruction_dir(store: SceneStore, scene_id: str) -> Path:
    record = store.get_scene(scene_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown Scene")
    if record.status is not JobStatus.SUCCEEDED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Scene is not ready: its job is {record.status.value}",
        )
    return store.reconstruction_dir(scene_id)


def _scene_summary(record: SceneRecord) -> dict[str, str]:
    return {
        "scene_id": record.scene_id,
        "job_id": record.job_id,
        "name": record.name,
        "created_at": record.created_at,
        "status": record.status.value,
    }


def _validate_image(filename: str, data: bytes) -> None:
    expected = _ALLOWED_FORMATS.get(PurePath(filename).suffix.lower())
    if expected is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{filename!r} is not a JPG or PNG file"
        )
    try:
        with Image.open(io.BytesIO(data)) as img:
            actual = img.format
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{filename!r} is not a readable image"
        ) from exc
    if actual != expected:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{filename!r} is not a valid {expected} file"
        )
