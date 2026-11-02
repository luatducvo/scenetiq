# backend/

Run `uv run main.py …` from the repository root, not from here: the uv workspace and `main.py` live there.

## Packages

Libraries in `packages/`, processes in `services/`. Dependencies point one way: `scene-core` ← `scene-stages`, `scene-store`, `scene-target` ← the services.

| Package | Holds |
| --- | --- |
| `scene-core` | domain models, Stage contracts and Manifests, ports, config, errors, Publication files, membership format, geometry |
| `scene-stages` | every Algorithm, real and fake |
| `scene-store` | every database write, the `scene_events` outbox |
| `scene-target` | `TargetWorkspace`, the only way to reach a Target Workspace |
| `scene-api` | FastAPI app: routes, read queries, Search ranking |
| `scene-worker` | the Job runner |
| `scene-compute` | the HTTP server that runs Algorithms on a GPU host |

## Rules

- `scene-core` imports no torch, FastAPI or database driver. A heavy dependency goes in an optional extra of `scene-stages`, one per Algorithm, and is imported inside the Algorithm's methods, not at module top.
- An Algorithm is a `StageAlgorithm` named `<stage>.<method>`, registered as a `scene_stages` entry point in `packages/scene-stages/pyproject.toml`. Its `reads` and `writes` match its Stage's folders in `OUTPUTS`. Every Stage has a Fake Algorithm in `scene_stages/fake.py`.
- Every database write goes through `SceneStore`, which commits a transition's rows and its events in one transaction. Read queries for the API stay in `scene-api`.
- The whole schema is `migrations/001_init.sql`; never add a second migration file. A schema change edits it; the Colab notebook's backend cell creates an empty database from it on every run, and the API and the worker never touch the schema.
- Reach a Scene's inputs and Stage outputs only through `TargetWorkspace`. Only `SceneResults` in `scene_core/results.py` knows the names of Publication files.
- Errors and fallbacks: [docs/agents/failures.md](../docs/agents/failures.md). Tests: [docs/agents/testing.md](../docs/agents/testing.md).

## Read before changing

- Stages, Jobs and publishing: [docs/architecture/pipeline.md](../docs/architecture/pipeline.md)
- Targets, scene-compute, `deploy/`: [docs/architecture/compute.md](../docs/architecture/compute.md)
- Routes, events, Search, database: [docs/architecture/api.md](../docs/architecture/api.md)
