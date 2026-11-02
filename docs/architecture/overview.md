# System overview

A user creates a Scene, uploads photos or one video of a room, and starts a Job. The worker runs the eight Stages on the pipeline's Target, publishes the result, and the browser then shows the room in 3D and finds Objects by text. Terms are defined in [GLOSSARY.md](../../GLOSSARY.md).

## Processes

| Process | Code | Role |
| --- | --- | --- |
| frontend | `frontend/` | React SPA; talks only to scene-api |
| scene-api | `backend/services/scene-api` | REST and SSE API; stores uploads on the Target; serves Publication files; Search |
| scene-worker | `backend/services/scene-worker` | claims queued Jobs, runs every Stage on the Target, publishes |
| Postgres + pgvector | `backend/migrations/001_init.sql` | Scenes, Jobs, Stage Runs, Objects with embeddings, events |
| scene-compute | `backend/services/scene-compute` | optional; runs Algorithms on a GPU host behind an HTTP contract |

Compose (`compose.yaml`) runs these under three profiles: `backend` (Postgres, scene-api, scene-worker), `frontend`, and `local-gpu` (scene-compute on this machine). `uv run main.py up` starts the first two.

## Data flow

1. **Upload**: scene-api stores files under `raw/` in the Scene's Target Workspace. The browser first shrinks photos to the long side ingest expects.
2. **Process**: `POST /process` inserts a queued Job. The worker claims it, clears earlier Stage outputs (keeping `raw/`), and runs ingest → recon → align → mesh → segment → lift → objects → embed on the one Target. Each Stage is one Compute Run and leaves a Manifest and a Stage Run.
3. **Publish**: the worker downloads only `objects/`, `embeddings/` and `mesh/` into `DATA_DIR/scenes/{scene}/{job}/`, then writes Objects and embeddings to the database and points the Scene at that Publication, in one transaction. Frames and every other intermediate stay on the Target.
4. **Explore**: the browser loads the mesh, point cloud and Object membership from the Publication, lists Objects, and runs Searches (the query is embedded on the Target, then compared in pgvector).

Progress reaches the browser as `scene_events` rows that the store writes with each transition and `GET /events` streams as SSE.

## Code map

| Path | Holds |
| --- | --- |
| `main.py` | project commands (`uv run main.py --help`) |
| `backend/packages/scene-core/src/scene_core/` | `domain.py` Object models · `contracts.py` Stages, Workspace paths, Manifests · `ports.py` `StageAlgorithm` · `config.py` pipeline and compute config · `executor.py` runs an Algorithm in a Workspace · `registry.py` entry-point lookup · `results.py` Publication files · `membership.py` membership binary · `geometry.py` box and plane fitting · `errors.py` `SceneError` |
| `backend/packages/scene-stages/src/scene_stages/` | `ingest.py` · `gpu.py` recon, align, segment · `mesh.py` · `lift.py` · `objects.py` Object boxes and membership · `placement.py` placement facts · `mg3d_doc.py` embed: Object Descriptions, embedding, reranking · `fake.py` Fake Algorithms · `arrays.py` NumPy/CuPy choice |
| `backend/packages/scene-store/` | `SceneStore`: every write, the event stream |
| `backend/packages/scene-target/` | `TargetWorkspace`, `open_target_workspace`, in-process and HTTP adapters |
| `backend/services/scene-api/` | `main.py` routes and Search ranking · `models.py` API models |
| `backend/services/scene-worker/` | `runner.py` the Job loop |
| `backend/services/scene-compute/` | `server.py` the compute contract · `download_models.py` |
| `backend/config/` | pipeline and compute YAML, Vocabularies |
| `backend/deploy/` | the Colab notebook and README; the local GPU README |
| `backend/tests/` | pytest suite |
| `frontend/src/pages/` | route screens |
| `frontend/src/components/` | screens and panels; `viewer/` for the 3D canvas |
| `frontend/src/lib/` | `api/` client, types, mock · `sceneProgress.ts` · `store/viewerStore.ts` · `upload/` · `quality/` |
| `docs/diagrams/` | rendered architecture diagrams |

## Further reading

- [pipeline.md](pipeline.md): Stages, Workspace layout, Jobs, publishing
- [compute.md](compute.md): Targets, the compute contract, the GPU image, Providers
- [api.md](api.md): endpoints, lifecycle, events, Search, database, membership format
- [frontend.md](frontend.md): routes, screens, viewer
- [../extending.md](../extending.md): recipes for common additions
