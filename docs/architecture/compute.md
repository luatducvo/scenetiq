# Compute

## Targets

The pipeline file names one Target, and everything that touches a Target Workspace uses it: uploads, all eight Stages, deleting a Scene, and embedding Search queries. Keeping the whole pipeline on one machine means Frames and intermediate arrays never cross the network; only a Publication does.

`backend/config/compute.yaml` lists the Targets:

| Target | Kind | URL variable |
| --- | --- | --- |
| `cpu` | in-process, under `DATA_DIR/target` on the worker and API host | none |
| `local_gpu` | HTTP, scene-compute in the `local-gpu` Compose profile | fixed `http://scene-compute:9000` |
| `colab` | HTTP, scene-compute on a Google Colab runtime behind a Cloudflare Tunnel | `COLAB_URL` |

`scene_target.open_target_workspace` builds a `TargetWorkspace` for the pipeline's Target and raises a `ValueError` naming what to set in `backend/.env` when the Target is unset, unknown or has no URL. The API builds it on first use and waits at most 10 seconds for a cold Target; the worker waits up to the Target's `startup_timeout`.

## Adapters

`TargetWorkspace` delegates to one of two adapters in `scene_target`:

- `InProcessAdapter` runs Algorithms through `scene_core.executor.Executor` in the calling process.
- `HttpAdapter` speaks the compute contract:
  - before each call it polls `GET /ping` every `poll_interval` until `startup_timeout`, then fails with `COMPUTE_UNAVAILABLE`;
  - it uploads in `chunk_size` pieces with the file's SHA-256, and downloads with `Range` requests (a file within one chunk in a single gzip-compressed request), checking every SHA-256;
  - it moves `transfer_concurrency` files at once;
  - it starts a Compute Run, polls it, forwards its progress, and fails with `COMPUTE_TIMEOUT` after six hours;
  - it raises the Algorithm's own error code when a Compute Run fails.

## The compute contract

`backend/services/scene-compute/src/scene_compute/server.py`. Every request except `/ping` needs `X-Compute-Token` when `COMPUTE_TOKEN` is set.

| Request | Does |
| --- | --- |
| `GET /ping` | readiness |
| `GET /info` | image version, GPU name, installed Algorithms |
| `GET /workspaces/{scene}/files?dirs=a,b&cursor=n` | paged list of files with size and SHA-256 |
| `PUT /workspaces/{scene}/files/{path}?offset&total&sha256` | append one chunk; the file appears when the last chunk's checksum matches |
| `GET /workspaces/{scene}/files/{path}` | download, with optional `Range` |
| `DELETE /workspaces/{scene}/files?dirs=a,b` | delete folders or files |
| `DELETE /workspaces/{scene}` | delete the Target Workspace |
| `POST /runs` `{algorithm, scene_id, params}` | queue a Compute Run; returns `run_id` |
| `GET /runs/{run_id}` | `state` (`queued`, `running`, `done`, `failed`), `progress`, `manifest` or `error` |
| `POST /invoke/{algorithm}/embed_text` `{texts}` | `{vectors}`, for Search queries |

Server behaviour:

- One thread runs Compute Runs and `embed_text` calls, one at a time, so models are never used concurrently.
- While a Scene has a queued or running Compute Run, writes to its Target Workspace and new runs for it fail with `WORKSPACE_BUSY`.
- Request bodies and responses are capped at `CHUNK_SIZE_MB` (`BODY_TOO_LARGE`); each request is capped at `REQUEST_TIMEOUT_S` (`REQUEST_TIMEOUT`), which is why Stage work is asynchronous.
- Run records persist in `DATA_DIR/runs/`. On restart, a run left queued or running becomes failed with `COMPUTE_INTERRUPTED`. `CLEAN_ON_START=1` empties `scenes/` and `runs/` first, for disposable hosts.

## The GPU image

`Dockerfile.compute` builds the image for the `local_gpu` Compose profile from the repository root, on CUDA 12.8; Colab installs `scene-compute` from the repository instead.

- The `EXTRAS` build argument chooses what it contains: empty for Fake Algorithms only, `gpu` for every real Algorithm (`scene-compute[gpu]` pins torch, torchvision and xformers together).
- `uv run main.py compute-image [--extras gpu]` builds it and `push-compute-image` pushes it, tagged `<variant>-<git short sha>` under `COMPUTE_IMAGE_REPO`.
- Weights are cached under `MODELS_DIR` (`HF_HOME`, `TORCH_HOME` inside it). `python -m scene_compute.download_models` fetches them for every Algorithm in `pipeline.yaml` that the image has installed; locally, `uv run main.py download-models` runs it into `HOST_MODELS_DIR`.

## Providers

Colab is the only Provider: `backend/deploy/colab/` holds its README and the notebook `scene_compute_colab.ipynb`. `backend/deploy/local-gpu/` covers the `local_gpu` Compose profile and building the image. Point the worker at a new Target with `pipeline.fake-gpu.yaml` first: it checks transport, authentication and file transfer without weights.
