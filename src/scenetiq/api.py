"""HTTP API: upload Captures, track jobs, fetch Reconstructions and run Queries."""

import io
import logging
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi import Path as FastApiPath
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
from scenetiq.storage import JobStatus, SceneRecord, SceneStore, UnknownCaptureError

logger = logging.getLogger(__name__)

MIN_VIEWS = 10
MAX_VIEWS = 300
_ALLOWED_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
_ID_PATTERN = "^[0-9a-f]{32}$"  # capture ids become directory names, so never accept a path


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

    @app.get("/health")
    def health(request: Request) -> dict[str, str | bool]:
        """Report that the backend is up and whether a job holds the single job slot."""
        return {"status": "ok", "job_running": request.app.state.runner.busy}

    @app.post("/captures", status_code=status.HTTP_201_CREATED)
    def start_capture(request: Request, body: CaptureIn) -> dict[str, str]:
        """Start staging a Capture, refusing it before any View is sent if it cannot run."""
        if not MIN_VIEWS <= body.view_count <= MAX_VIEWS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"a Capture needs {MIN_VIEWS}-{MAX_VIEWS} images, got {body.view_count}",
            )
        if request.app.state.runner.busy:
            raise HTTPException(status.HTTP_409_CONFLICT, "another job is running")
        return {"capture_id": request.app.state.store.start_capture(body.view_count)}

    # One URL for every View, so that browsers reuse a single cached CORS preflight.
    @app.post("/captures/{capture_id}/views", status_code=status.HTTP_204_NO_CONTENT)
    def add_view(
        request: Request,
        capture_id: Annotated[str, FastApiPath(pattern=_ID_PATTERN)],
        index: Annotated[int, Form(ge=0)],
        image: Annotated[UploadFile, File()],
    ) -> None:
        """Validate and store one View; sending the same ``index`` again replaces it."""
        store: SceneStore = request.app.state.store
        filename = image.filename or ""
        try:
            view_count = store.capture_view_count(capture_id)
            if index >= view_count:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"View index {index} is outside this Capture of {view_count} Views",
                )
            data = image.file.read()
            _validate_image(filename, data)
            store.save_view(capture_id, index, PurePath(filename).suffix.lower(), data)
        except UnknownCaptureError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown Capture") from exc

    @app.post("/scenes", status_code=status.HTTP_201_CREATED)
    def create_scene(request: Request, body: SceneIn) -> dict[str, str]:
        """Create a Scene from a fully received Capture and start reconstructing it."""
        runner: JobRunner = request.app.state.runner
        store: SceneStore = request.app.state.store
        try:
            view_count = store.capture_view_count(body.capture_id)
            received = store.received_view_count(body.capture_id)
            if received != view_count:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"received only {received} of this Capture's {view_count} Views",
                )
            runner.reserve()
        except JobBusyError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except UnknownCaptureError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown Capture") from exc
        try:
            record, paths = store.create_scene_from_capture(body.name, body.capture_id)
        except UnknownCaptureError as exc:
            runner.release()
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown Capture") from exc
        except BaseException:
            runner.release()
            raise
        runner.start(record, Capture(views=[View(path=p) for p in paths]))
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


class CaptureIn(BaseModel):
    """The size of a Capture about to be staged."""

    view_count: int


class SceneIn(BaseModel):
    """A name for a new Scene and the staged Capture to create it from."""

    name: str = Field(min_length=1, max_length=200)
    capture_id: str = Field(pattern=_ID_PATTERN)


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
            # Phones write JPEGs with embedded pictures, which Pillow reports as MPO.
            actual = "JPEG" if img.format == "MPO" else img.format
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{filename!r} is not a readable image"
        ) from exc
    if actual != expected:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{filename!r} is not a valid {expected} file"
        )
