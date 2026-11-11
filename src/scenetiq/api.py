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
from scenetiq.domain import (
    Aspect,
    CaptionEncoder,
    Capture,
    ParsedQuery,
    Pipeline,
    QueryParseError,
    QueryParser,
    TextEncoder,
    View,
)
from scenetiq.jobs import JobBusyError, JobRunner
from scenetiq.query import QUERY_LOGGER, Match, MissingEmbeddingsError, run_query
from scenetiq.runtime_log import UploadProgress, report_progress
from scenetiq.spatial import run_spatial_query
from scenetiq.storage import (
    JobEvent,
    JobEventKind,
    JobStatus,
    SceneRecord,
    SceneStore,
    UnknownCaptureError,
)

logger = logging.getLogger(__name__)
_query_logger = logging.getLogger(QUERY_LOGGER)

_ALLOWED_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
_ID_PATTERN = "^[0-9a-f]{32}$"  # capture ids become directory names, so never accept a path


def create_app(
    settings: Settings,
    pipeline: Pipeline,
    text_encoder: TextEncoder,
    caption_encoder: CaptionEncoder,
    query_parser: QueryParser,
) -> FastAPI:
    """Build the SceneTIQ API.

    Args:
        settings: Storage root, shared token, Query ranking and CORS origins.
        pipeline: Turns an uploaded Capture into a Reconstruction.
        text_encoder: Encodes Query text for matching against Instance image embeddings.
        caption_encoder: Encodes Query text for matching against Instance Captions.
        query_parser: Splits every Query into objects and Relations (ADR 0009).

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

    @app.get("/options")
    def get_options() -> OptionsOut:
        """Report the settings the frontend configures itself from."""
        return OptionsOut(
            min_views=settings.min_views,
            max_views=settings.max_views,
            upload_max_side=settings.upload_max_side,
            upload_jpeg_quality=settings.upload_jpeg_quality,
        )

    @app.post("/captures", status_code=status.HTTP_201_CREATED)
    def start_capture(request: Request, body: CaptureIn) -> dict[str, str]:
        """Start staging a Capture, refusing it before any View is sent if it cannot run."""
        if not settings.min_views <= body.view_count <= settings.max_views:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"a Capture needs {settings.min_views}-{settings.max_views} images, "
                f"got {body.view_count}",
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
            received = store.received_view_count(capture_id)
            report_progress(UploadProgress(capture_id, received, view_count))
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
        selected = paths[:: settings.view_stride]
        runner.start(record, Capture(views=[View(path=p) for p in selected]))
        return {"scene_id": record.scene_id, "job_id": record.job_id}

    @app.get("/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> JobOut:
        """Report a job's status, current stage, error, stage timings and notes."""
        store: SceneStore = request.app.state.store
        record = store.get_job(job_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        events = store.job_events(job_id)
        return JobOut(
            job_id=record.job_id,
            scene_id=record.scene_id,
            status=record.status.value,
            stage=record.stage,
            error=record.error,
            started_at=record.started_at,
            finished_at=record.finished_at,
            stages=_stage_timings(events, record.finished_at),
            notes=[
                NoteOut(at=e.at, stage=e.stage, message=e.message)
                for e in events
                if e.kind is JobEventKind.NOTE
            ],
        )

    @app.post("/jobs/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
    def cancel_job(request: Request, job_id: str) -> dict[str, str]:
        """Ask a queued or running job to stop at its next checkpoint; returns at once."""
        record = request.app.state.store.get_job(job_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
        unfinished = record.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        if not unfinished or not request.app.state.runner.cancel(job_id):
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"job is {record.status.value}, not running"
            )
        return {"job_id": job_id, "status": "cancelling"}

    @app.get("/scenes")
    def list_scenes(request: Request) -> list[SceneOut]:
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
        """List the Scene's Instances with centroid, bounding box (metres) and Caption Name."""
        directory = _ready_reconstruction_dir(request.app.state.store, scene_id)
        return [
            InstanceOut(
                instance_id=inst.instance_id,
                centroid=inst.centroid.tolist(),
                bbox_min=inst.bbox_min.tolist(),
                bbox_max=inst.bbox_max.tolist(),
                name=None if inst.caption is None else inst.caption.name,
            )
            for inst in load_instances(directory)
        ]

    @app.post("/scenes/{scene_id}/query")
    def query_scene(request: Request, scene_id: str, body: QueryIn) -> QueryOut:
        """Find the Instances that match a free-text Query, honouring its Relations."""
        directory = _ready_reconstruction_dir(request.app.state.store, scene_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Query text is empty")
        instances = load_instances(directory)
        # A Scene where no Instance got a Caption has no Caption ranking: skip encoding.
        captioned = any(inst.caption_embeddings for inst in instances)

        def rank(phrase: str, top_k: int) -> list[Match]:
            return run_query(
                instances,
                text_encoder.encode(phrase),
                top_k,
                settings.rrf_k,
                caption_encoder.encode_query(phrase) if captioned else None,
            )

        # Only a parse failure the parser reports falls back; any other exception is a bug.
        parsed: ParsedQuery | None = None
        try:
            parsed = query_parser.parse(text)
        except QueryParseError as exc:
            _query_logger.warning("query parse failed, answered as a plain Query: %s", exc)
        try:
            matches = None
            if parsed is not None and parsed.constraints:
                matches = run_spatial_query(
                    instances, parsed, rank, settings.query_top_k, settings.spatial
                )
            relations_applied = matches is not None
            if matches is None:  # no Relation, no valid assignment or no parse: plain Query
                matches = rank(text, settings.query_top_k)
        except MissingEmbeddingsError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return QueryOut(
            matches=[
                MatchOut(
                    instance_id=m.instance_id,
                    score=m.score,
                    image_score=m.image_score,
                    caption_score=m.caption_score,
                    caption_aspect=m.caption_aspect,
                    anchors=[
                        AnchorOut(instance_id=a.instance_id, phrase=a.phrase) for a in m.anchors
                    ],
                )
                for m in matches
            ],
            relations_applied=relations_applied,
            relations_stated=parsed is not None and bool(parsed.constraints),
            unsupported=[] if parsed is None else list(parsed.unsupported),
            parse_failed=parsed is None,
        )

    return app


class OptionsOut(BaseModel):
    """The View limits of a Capture, and how the frontend prepares photos (ADR 0010).

    The frontend shrinks every photo so that its long side is at most ``upload_max_side``
    pixels and re-encodes it as JPEG at ``upload_jpeg_quality`` in (0, 1].
    """

    min_views: int
    max_views: int
    upload_max_side: int
    upload_jpeg_quality: float


class CaptureIn(BaseModel):
    """The size of a Capture about to be staged."""

    view_count: int


class SceneIn(BaseModel):
    """A name for a new Scene and the staged Capture to create it from."""

    name: str = Field(min_length=1, max_length=200)
    capture_id: str = Field(pattern=_ID_PATTERN)


class SceneOut(BaseModel):
    """One Scene in the list.

    ``started_at`` and ``finished_at`` (ISO 8601 UTC) are null until its job starts and
    ends; ``finished_at`` stays null for a job interrupted by a restart.
    ``instance_count`` is null unless its job succeeded.
    """

    scene_id: str
    job_id: str
    name: str
    created_at: str
    status: str
    started_at: str | None
    finished_at: str | None
    instance_count: int | None


class StageOut(BaseModel):
    """When one Stage of a job started and ended (null while it runs)."""

    stage: str
    started_at: str
    finished_at: str | None


class NoteOut(BaseModel):
    """One note of a job, with the Stage running when it was recorded."""

    at: str
    stage: str | None
    message: str


class JobOut(BaseModel):
    """A job's status and what it reported.

    ``stages`` lists every Stage that started, in order; a Stage ends when the next starts
    or the job ends. ``notes`` are in the order recorded. Timestamps are ISO 8601 UTC;
    ``started_at`` and ``finished_at`` are null until the job starts and ends.
    """

    job_id: str
    scene_id: str
    status: str
    stage: str | None
    error: str | None
    started_at: str | None
    finished_at: str | None
    stages: list[StageOut]
    notes: list[NoteOut]


class InstanceOut(BaseModel):
    """An Instance's position and extent, in metres, and its Caption's Name (or null)."""

    instance_id: int
    centroid: list[float]
    bbox_min: list[float]
    bbox_max: list[float]
    name: str | None


class QueryIn(BaseModel):
    """A free-text Query."""

    text: str = Field(max_length=500)


class AnchorOut(BaseModel):
    """An Instance a Match was measured against, and the phrase of the object it stands for."""

    instance_id: int
    phrase: str


class MatchOut(BaseModel):
    """One Match: an Instance id, its score, the per-channel similarities and its Anchors.

    ``score`` is the normalised fused score, or the assignment score when Relations were
    applied (ADR 0009); either way it is relative to one Query. ``caption_score`` is the
    similarity of the best-matching Description and ``caption_aspect`` its Aspect; both are
    null for an Instance without a Caption. When Relations were applied, the similarities are
    those of the wanted object's phrase. ``anchors`` is empty when they were not.
    """

    instance_id: int
    score: float
    image_score: float
    caption_score: float | None
    caption_aspect: Aspect | None
    anchors: list[AnchorOut]


class QueryOut(BaseModel):
    """The top-K Matches for a Query, best first, and what of the Query was taken into account.

    ``relations_applied`` is true when the Query's Relations decided the Matches.
    ``relations_stated`` is true when the parsed Query had at least one Relation, so stated
    but not applied means no assignment was valid and the Query was answered as a plain
    Query. ``unsupported`` lists the parts of the Query the Relation set cannot express, as
    the parser quoted them; they are never applied. ``parse_failed`` is true when the parser
    could not read the Query, which was then answered as a plain Query; the other three
    are then false and empty.
    """

    matches: list[MatchOut]
    relations_applied: bool
    relations_stated: bool
    unsupported: list[str]
    parse_failed: bool


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


def _stage_timings(events: list[JobEvent], finished_at: str | None) -> list[StageOut]:
    starts = [e for e in events if e.kind is JobEventKind.STAGE and e.stage is not None]
    # A Stage ends when the next one starts; the last one when the job ends.
    ends = [e.at for e in starts[1:]] + [finished_at][: len(starts)]
    return [
        StageOut(stage=e.stage or "", started_at=e.at, finished_at=end)
        for e, end in zip(starts, ends, strict=True)
    ]


def _scene_summary(record: SceneRecord) -> SceneOut:
    return SceneOut(
        scene_id=record.scene_id,
        job_id=record.job_id,
        name=record.name,
        created_at=record.created_at,
        status=record.status.value,
        started_at=record.started_at,
        finished_at=record.finished_at,
        instance_count=record.instance_count,
    )


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
