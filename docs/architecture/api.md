# API

`backend/services/scene-api/src/scene_api/`. The contract is `backend/openapi/openapi.json`; errors follow [docs/agents/failures.md](../agents/failures.md).

## Endpoints

All under `/api/v1/scenes`.

| Request | Does |
| --- | --- |
| `POST /` `{name}` | create a Scene (`created`) |
| `GET /` | list Scenes, newest first |
| `GET /{scene}` | Scene with Stage Runs, `assets` URLs, `failure`, `ingest_report` |
| `DELETE /{scene}` | delete an idle Scene, its Publications and, in the background, its Target Workspace |
| `POST /{scene}/uploads` (multipart `files`) | store photos or one video on the Target |
| `POST /{scene}/process` | queue a Job; returns `job_id` |
| `GET /{scene}/events` | SSE progress stream |
| `GET /{scene}/objects`, `GET /{scene}/objects/{object}` | Objects of the current Publication |
| `GET /{scene}/search?q=` | Search Hits |
| `GET /{scene}/publications/{job}/files/{path}` | a public file of a Publication |

## Scene lifecycle

`created` → upload → `uploaded` → process → `processing` → `ready` or `failed`. A `ready` or `failed` Scene can be processed again.

| Action | Allowed when | Otherwise |
| --- | --- | --- |
| upload | `created` or `uploaded`, no active Job | `SCENE_LOCKED`, `SCENE_BUSY` |
| process | `uploaded`, `ready` or `failed`, no active Job | `SCENE_NOT_UPLOADED`, `SCENE_BUSY` |
| delete | no active Job | `SCENE_BUSY` |

Uploads accept `.jpg .jpeg .png .webp .heic` photos or one `.mp4 .mov .mkv .webm` video, never both and never a second video (`INVALID_UPLOAD`, `UNSUPPORTED_MEDIA`). Each stored name is the sanitised original stem plus a random suffix, so ingest keeps capture order.

## Progress events

`GET /events` first replays the Scene's state, then, while it is `processing`, streams new `scene_events` rows until a `ready` or `failed` status. Replayed and live events have the same shapes:

- `stage`: one Stage Run's `stage`, `algorithm`, `status`, `manifest`, `error`, `started_at`, `updated_at`
- `progress`: `stage`, `done`, `total`, `unit`
- `status`: the Scene's `status`, plus `error` (the failure) when failed

Only live events carry an SSE `id`.

## Publication files

`assets` gives `mesh_url` (`mesh/mesh.glb`), `points_url` (`mesh/points.ply`) and `membership_url` (`objects/membership.bin`), each a URL that names the Publication, so a new Job's files never reuse an old URL. Only `mesh/` and `objects/membership.bin` are public (`SceneResults.PUBLIC`); everything else, including the Objects file, is served through the Objects endpoints.

## Search

The query is embedded on the Target with the embed Stage's Algorithm and compared with each Object's embedding by cosine similarity in pgvector. The `SEARCH_CANDIDATES` (default 20) most similar Objects and every label match are then reranked on the Target, by the same Algorithm, against their Object Descriptions (`objects.description`), each getting a relevance in 0..1. A label match is the query containing the Object's label, or the reverse.

- `GET /search` returns every reranked label match (score = relevance + 1), then the other reranked Objects within `SEARCH_MARGIN` (default 0.05) of the best relevance, at most 10 hits in all; an Object that was not reranked is never a hit. Each hit carries `score` and `raw_similarity`, the relevance without the label bonus.

## Database

`backend/migrations/001_init.sql` is the whole schema. The Colab notebook's backend cell runs it on a new, empty database each time it starts the backend; the test command does the same for its disposable container. The API and the worker never change the schema. A schema change edits this file and takes effect when the backend cell runs again.

| Table | Holds |
| --- | --- |
| `scenes` | name, status, `failure`, `publication` (Job id) |
| `jobs` | status `queued`, `running`, `done`, `failed`; at most one queued or running per Scene |
| `stage_runs` | one row per Stage of the latest Job: Algorithm, status, Manifest, error, progress, times |
| `objects` | the current Publication's Objects with a `vector(1024)` embedding (HNSW cosine index) and their Object Description |
| `scene_events` | the progress outbox read by `GET /events` |

## Object membership file

`objects/membership.bin`, written by the objects Stage, encoded by `scene_core/membership.py` and decoded by `frontend/src/lib/api/membership.ts`. All integers are little-endian `uint32`, and every section starts on a 4-byte boundary so the browser can view each array in place:

```
magic "SCMB" | version 1 | object_count N | point_count | vertex_count | ids_byte_length L
ids: UTF-8 JSON array of the N Object ids, space-padded to L bytes
point_offsets[N + 1]  | point_indices[point_offsets[N]]
vertex_offsets[N + 1] | vertex_indices[vertex_offsets[N]]
```

Object `i` owns `point_indices[point_offsets[i] : point_offsets[i + 1]]`, sorted and unique; vertices work the same way. Points index `mesh/points.ply`, vertices index every primitive of `mesh/mesh.glb` in order. The ids list the Objects of `objects/objects.json` in the same order.
