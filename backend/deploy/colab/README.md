# Google Colab backend

Runs the whole backend on one Google Colab GPU runtime: PostgreSQL with pgvector, scene-api,
scene-worker and scene-compute. A free Cloudflare Tunnel exposes scene-api to the frontend on your
machine. [GPU.ipynb](GPU.ipynb) is the notebook; this guide explains its steps.

## How it works

```
your machine                         Google Colab runtime
npm run dev (:3000) ── HTTPS ──>     cloudflared → scene-api (127.0.0.1:8000)
                                                    │
                                     scene-worker ──┼── Postgres + pgvector (127.0.0.1:5432)
                                                    │
                                                    └─ scene-compute (127.0.0.1:9001), the colab Target
```

Only scene-api is public. scene-compute and Postgres listen on localhost, so the `colab` Target in
`backend/config/compute.yaml` has a fixed URL and no token. scene-compute stays a separate process so
the models are loaded once, for the worker's Stages and the API's Search alike.

## Run the notebook

Open `GPU.ipynb` in Colab (File → Upload notebook, or File → Open notebook → GitHub) and run the
cells in order:

0. **Download weights.** Once, on any runtime (a CPU runtime is enough), fetch every model's
   weights (over 25 GB, of which the embed Stage's Qwen3-VL, bge-m3 and bge-reranker-v2-m3 take
   about 22 GB, so a free 15 GB Drive is not enough) into the Drive cache that step 5 checks, so GPU
   time is not spent downloading. It needs only `HF_TOKEN` in Colab secrets: no clone or install. It
   downloads to the runtime SSD and then copies each file to Drive, because the Drive mount
   mishandles the chunked writes of large Hugging Face downloads.
1. **GPU check.** Runtime → Change runtime type → pick a GPU, then `nvidia-smi`.
2. **uv and Python 3.12.**
3. **Clone the repository** into `/content/3D-scene` (set `REPO_URL`; for a private repository use a
   token URL or upload a zip without `.venv`, `.git`, `data` and `models`).
4. **Install** every backend process and the GPU Algorithms:
   `uv sync --python 3.12 --all-packages --extra gpu --no-dev`.
5. **Model cache check.** Mounts Google Drive and sets `MODELS_DIR` there. It never downloads:
   it prints `cached` for each model complete on Drive and `MISSING` with the missing files
   otherwise, then raises an error and stops if any weight is missing; run step 0 and this cell
   again. When every weight is present it sets `HF_HUB_OFFLINE=1`, so the server never downloads
   at run time.
6. **PostgreSQL.** Installs PostgreSQL 16 and pgvector from the PostgreSQL apt repository, starts it
   and creates the `scene` role. Once per runtime.
7. **Backend.** Stops any running backend process, creates an empty `scene` database from
   `backend/migrations/001_init.sql`, empties `/content/data`, then starts scene-compute, scene-api
   and scene-worker in the background with the variables listed in `backend/.env.example`. It waits
   until scene-compute and scene-api answer. Logs go to `/content/logs/<process>.log`.
8. **Tunnel.** Starts `cloudflared` for scene-api and prints `VITE_API_URL=<public URL>`.
9. **Logs.** Prints the end of each log; run it when a Job fails or the API does not answer.

## Configure your machine

In `frontend/.env`:

```dotenv
VITE_API_URL=https://<random>.trycloudflare.com   # from step 8
VITE_API_MODE=http
```

Then run `npm run dev` in `frontend/` and open <http://localhost:3000>. scene-api allows
`http://localhost:3000` as its only CORS origin.

## Smoke test

```bash
curl -i https://<random>.trycloudflare.com/api/v1/scenes
```

It returns 200 and `[]` on a fresh backend. Then create and process a Scene from the web app.

## Limits

- **No persistence.** Scenes, Jobs, Objects and Publications live on the runtime. Running step 7
  again, restarting the runtime or losing it deletes them all.
- **No authentication.** Anyone with the tunnel URL can create, process and delete Scenes while the
  runtime runs. The URL is random and changes with every run of step 8.
- **VRAM.** The Executor keeps every model loaded, and the real pipeline is sized for at least
  48 GB. A T4 (about 15 GB) cannot hold every real model at once; use an 80 GB A100 runtime and keep
  `max_keyframes` in [pipeline.yaml](../../config/pipeline.yaml) at 80 or lower.
- **Session lifetime.** Colab disconnects idle or closed tabs, which ends the backend too.
- **New URL per tunnel.** Each run of step 8 prints a new URL: update `VITE_API_URL` in
  `frontend/.env` and restart `npm run dev`.
