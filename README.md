# SceneTIQ

Turns uncalibrated photos of a static scene into a metric 3D point cloud split into objects (Instances) that free-text Queries can find, such as "red chair" or "the box on the shelf". The GPU backend runs on a remote GPU runtime; the React viewer runs locally.

> The GPU Pipeline has not been run end to end yet.

## Layout

| Path | What |
|---|---|
| `src/scenetiq/` | FastAPI backend: API, storage, Query ranking, spatial solver |
| `src/scenetiq/pipeline/` | GPU stages: DA3 → TSDF fusion → gravity alignment → SAM2 + MV3DIS → SigLIP2 + Qwen3-VL Captions |
| `frontend/` | React + Vite + TypeScript viewer |
| `notebooks/` | Starts the backend on a GPU runtime |
| `tests/` | `api/` and `unit/` run on CPU; `eval/` needs a GPU |
| `GLOSSARY.md`, `docs/adr/` | Domain terms and design decisions |

## Local development

Needs [uv](https://docs.astral.sh/uv/) and Node.js. The local backend uses a fake Pipeline, so no GPU or model is needed.

```bash
uv sync
uv run pre-commit install
SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8001

cp frontend/.env.example frontend/.env      # already points at localhost:8001 with token "dev"
npm --prefix frontend install
npm --prefix frontend run dev               # http://localhost:3000
```

Checks (also run by the pre-commit hooks):

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
npm --prefix frontend run lint && npm --prefix frontend run typecheck && npm --prefix frontend run build
```

## Running on a GPU

1. Create a Cloudflare named tunnel (Zero Trust → Networks → Tunnels) with a public hostname pointing to `http://localhost:8001`.
2. Add two secrets to the runtime: `SCENETIQ_TOKEN` (any long random string) and `CLOUDFLARE_TUNNEL_TOKEN`.
3. Open the notebook in `notebooks/` on a GPU runtime. It clones `main`, so the repo must be public.
4. Edit `CONFIG` in the first cell (`"pipeline": "fake"` or `"real"`; every key is in `src/scenetiq/config.py`), then run all cells.
5. Point `frontend/.env` at `https://<tunnel hostname>` with your `SCENETIQ_TOKEN`, and restart `npm run dev`.

Scenes live on the runtime's disk and are lost when it disconnects. The first real job downloads about 15 GB of weights. Vite bakes the token into the bundle, so never host a build publicly.

## License

DA3 Nested weights are CC BY-NC 4.0 (ADR 0001), so SceneTIQ is for non-commercial use only.
