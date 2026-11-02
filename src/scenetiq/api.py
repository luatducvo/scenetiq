"""HTTP API: upload Captures, track jobs, fetch Reconstructions and run Queries."""

import io
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

from scenetiq.artifacts import POINTS_FILE
from scenetiq.config import Settings
from scenetiq.domain import Capture, Pipeline, TextEncoder, View
from scenetiq.jobs import JobBusyError, JobRunner
from scenetiq.storage import JobStatus, SceneRecord, SceneStore

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

    @app.get("/scenes/{scene_id}/points")
    def get_points(request: Request, scene_id: str) -> FileResponse:
        """Return the Reconstruction's point cloud as binary PLY (xyz metres, rgb)."""
        directory = _ready_reconstruction_dir(request.app.state.store, scene_id)
        return FileResponse(directory / POINTS_FILE, media_type="application/octet-stream")

    return app


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
