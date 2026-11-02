# Google Colab GPU Target

Runs `scene-compute` on a Google Colab GPU runtime and exposes it to your machine through a free
Cloudflare Tunnel. [scene_compute_colab.ipynb](scene_compute_colab.ipynb) is the notebook; this
guide explains the steps and what to configure locally.

## How it works

```
your machine                                        Google Colab
web · scene-api · postgres                          scene-compute (uvicorn, port 9001)
scene-worker ── HTTPS (tunnel URL, 100 MB chunks) ─> cloudflared → https://<random>.trycloudflare.com
```

Every Stage runs on Colab, and uploads are stored there too; only the results return to your
machine after the last Stage.
Every request finishes in under 300 s and bodies stay under 100 MB, so the tunnel's limits never apply.

## Run the notebook

Open `scene_compute_colab.ipynb` in Colab (File → Upload notebook, or File → Open notebook →
GitHub) and run the cells in order:

0. **Download weights.** Once, on any runtime (a CPU runtime is enough), fetch every model's
   weights (over 25 GB, of which the embed Stage's Qwen3-VL, bge-m3 and bge-reranker-v2-m3 take
   about 22 GB, so a free 15 GB Drive is not enough) into the Drive cache that step 5 checks, so GPU time is not spent
   downloading. It needs only `HF_TOKEN` in Colab secrets: no clone or install. It downloads to
   the runtime SSD and then copies each file to Drive, because the Drive mount mishandles the
   chunked writes of large Hugging Face downloads.
1. **GPU check.** Runtime → Change runtime type → pick a GPU, then `nvidia-smi`.
2. **uv and Python 3.12.**
3. **Clone the repository** into `/content/3D-scene` (set `REPO_URL`; for a private repository use a
   token URL or upload a zip without `.venv`, `.git`, `data` and `models`).
4. **Install** `uv sync --python 3.12 --package scene-compute --extra gpu`. Drop `--extra gpu` for
   the fake image's dependencies when you only want to test the connection.
5. **Model cache check.** Mounts Google Drive and sets `MODELS_DIR` there. It never downloads:
   it prints `cached` for each model complete on Drive and `MISSING` with the missing files
   otherwise, then raises an error and stops if any weight is missing; run step 0 and this cell
   again. When every weight is present it sets `HF_HUB_OFFLINE=1`, so the server never downloads
   at run time.
6. **Tunnel.** Starts `cloudflared` for port 9001 and prints the public URL.
7. **Server.** Read `COMPUTE_TOKEN` from Colab secrets, then start uvicorn on port 9001, reusing
   the cache configuration from step 5 without printing the token. The cell sets
   `CLEAN_ON_START=1`, so every server start empties `/content/data`. Keep this cell running while
   you use the web app.

## Configure your machine

In `backend/.env`:

```dotenv
GPU_TARGET=colab
COLAB_URL=https://<random>.trycloudflare.com   # from step 6
COMPUTE_TOKEN=<same token as step 7>
PIPELINE_CONFIG=/app/backend/config/pipeline.fake-gpu.yaml   # first, to test the connection
# PIPELINE_CONFIG=/app/backend/config/pipeline.yaml          # then, real models
```

Then run `uv run main.py up` so the worker reads the new values.

## Smoke test

```bash
curl -i https://<random>.trycloudflare.com/ping
curl -H "X-Compute-Token: <token>" https://<random>.trycloudflare.com/info
```

`/ping` returns 200; `/info` lists the GPU name and the registered Algorithms. Then process a Scene
from <http://localhost:3000>.

## Limits

- **VRAM.** The Executor keeps every model loaded, and the real pipeline is sized for at least
  48 GB. A T4 (about
  15 GB) is enough for the fake pipeline, not for every real model at once; for real models use an
  80 GB A100 runtime and keep `max_keyframes` in [pipeline.yaml](../../config/pipeline.yaml) at 80
  or lower.
- **Session lifetime.** Colab disconnects idle or closed tabs. A runtime restart or a server restart
  deletes uploads and intermediate outputs: create a new Scene and upload again. Already processed
  Scenes stay viewable locally; search needs the server running.
- **New URL per tunnel.** Each run of step 6 prints a new URL: update `COLAB_URL` and run
  `uv run main.py up` again.

