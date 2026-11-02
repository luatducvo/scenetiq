# Extending

Each recipe lists every file the change touches. Finish with the done check in [agents/testing.md](agents/testing.md).

## A new Algorithm for an existing Stage

1. Write a `StageAlgorithm` subclass in `backend/packages/scene-stages/src/scene_stages/` (`gpu.py` for model-backed Algorithms, otherwise the Stage's module). Set `name = "<stage>.<method>"`, `stage`, `reads` and `writes` equal to the Stage's row in `OUTPUTS`, and `manifest_type` to the Stage's Manifest from `scene_core/contracts.py`. Write the same files, with the same names, as the Algorithm it replaces: later Stages read them.
2. Import heavy libraries inside `setup`, `run` or `download_weights`. Load models in `setup`; fetch weights into `MODELS_DIR` in `download_weights`; report progress with `self.progress`.
3. Register it under `[project.entry-points.scene_stages]` in `backend/packages/scene-stages/pyproject.toml`. Put its dependencies in a new optional extra there, and add that extra to the `gpu` extra in `backend/services/scene-compute/pyproject.toml` if it runs on the GPU image. Run `uv lock`.
4. Select it in `backend/config/pipeline.yaml` with its params.
5. Test it on small synthetic inputs in `backend/tests/test_<stage>.py`.

## A new API endpoint

1. Add the route to `scene_api/main.py` and its models to `scene_api/models.py`; writes go through a `SceneStore` method.
2. `uv run main.py openapi`, then `npm run api:sync` in `frontend/`.
3. Add a `sceneApi` method in `frontend/src/lib/api/http.ts` and a handler in `frontend/src/lib/api/mock/handlers.ts`.
4. Test the route in `backend/tests/test_api.py`.

## A new database column or table

Change `backend/migrations/001_init.sql`, the one schema file, and the `SceneStore` writes and the API reads that use it. The Colab notebook's backend cell creates the database from it on every run, and the test database is recreated on each `main.py test`.

## A new viewer layer

1. If it needs a new Publication file, write it from a Stage into `mesh/` or `objects/`, add it to `SceneResults.PUBLIC` and `ASSETS` in `scene_core/results.py` and to `SceneAssets` in `scene_api/models.py`, then regenerate the contract.
2. Write a component taking `SceneLayerProps` in `frontend/src/components/viewer/layers/index.tsx` and add it to `sceneLayers`.
3. Put its show and hide rules in `viewer/view.ts` with tests in `view.test.ts`. Geometry layers must preserve the single Points/Mesh choice; always-visible overlays can follow the `FloorGridLayer` example. If the layer needs a control, add it to the toolbar or spatial tools in `SceneExplorer.tsx`, with strings in `messages/en.json`. Add interaction tests in `SceneExplorer.interactions.test.tsx` when the control changes viewer state.

## A new Target or Provider

A Provider hosts the whole backend, as Colab does ([ADR 0002](adr/0002-the-backend-runs-on-one-colab-runtime.md)).

1. Add the Target to `backend/config/compute.yaml` with `backend: http` and the URL scene-compute listens on. Add `token: ${COMPUTE_TOKEN}` and a `<PROVIDER>_URL` variable in `backend/.env.example` only if scene-compute is reachable beyond the host.
2. Write `backend/deploy/<provider>/README.md` and its setup: how to install and start Postgres with pgvector, scene-api, scene-worker and scene-compute there, how the frontend reaches scene-api, and how to check `/ping` and `/info`.

## A new Stage

The eight Stages are fixed across both sides. A ninth touches `STAGES`, `OUTPUTS` and a new Manifest in `scene_core/contracts.py`; a Fake Algorithm and a real one; every `config/pipeline*.yaml`; `STAGES` in `frontend/src/lib/api/types.ts`; the Stage strings in `messages/en.json`; the mock Scene in `mock/handlers.ts`; and the API contract.
