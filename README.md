# 3D Scene Understanding

Turn photos or one short video of a room into a metric 3D Scene: a point cloud and mesh, the Objects in it with their sizes, and text Search over those Objects.

The pipeline runs Depth Anything 3 for reconstruction, GeoCalib for gravity alignment, TSDF fusion for the mesh, SAM 3 for open-vocabulary segmentation, mask clustering to lift masks into 3D Objects, Qwen3-VL to write each Object's Object Description, bge-m3 to embed it and bge-reranker-v2-m3 to rerank Search Hits. Every model Stage also has a fake that runs on a CPU, so the whole system works without a GPU.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv)
- Docker with Compose
- Node.js 24 and npm, for frontend development
- For real models: a CUDA GPU host with at least 48 GB of VRAM, local or rented

## Quick start

```bash
uv run main.py up        # Postgres, scene-api, scene-worker and the frontend
```

Open http://localhost:3000. On first run `main.py` creates `backend/.env` from `backend/.env.example`; with the defaults every Stage runs as a fake on the CPU.

`uv run main.py --help` lists every command, including `down`, `reset` (deletes all Scenes), `test` and `lint`.

To remove generated files on the host, run either command from the repository root:

```bash
uv run main.py clean python  # Python/tool caches and bytecode; also clears the global uv cache
uv run main.py clean react   # Frontend build output, Vite/Next.js caches and TypeScript build metadata
```

These commands keep `.venv`, installed dependencies in `node_modules`, source files, environment files and project data. They do not clean Docker images, containers or volumes.

## Frontend only

```bash
cd frontend
npm install
VITE_API_MODE=mock npm run dev
```

Mock mode serves a demo Scene from the browser; no backend is needed. In PowerShell, use `$env:VITE_API_MODE="mock"; npm run dev`.

The frontend is a React SPA built with Vite, React Router and Three.js through React Three Fiber. `npm run build` produces `frontend/dist/`; the production container serves it through Nginx. Set `VITE_API_URL` and `VITE_API_MODE` before building.

## Real models on a GPU

1. Run scene-compute on Google Colab following `backend/deploy/colab/README.md`.
2. In `backend/.env`, set `COLAB_URL` to the notebook's tunnel URL, `COMPUTE_TOKEN`, `GPU_TARGET=colab`, and `PIPELINE_CONFIG=/app/backend/config/pipeline.yaml`. Try `pipeline.fake-gpu.yaml` first to check the connection.
3. `uv run main.py up`.

With a local NVIDIA GPU, `uv run main.py download-models` and `uv run main.py up-local-gpu` run scene-compute on this machine (`GPU_TARGET=local_gpu`, `LOCAL_GPU_EXTRAS=gpu`).

## Documentation

- [GLOSSARY.md](GLOSSARY.md): the domain terms
- [docs/architecture/overview.md](docs/architecture/overview.md): how it fits together
- [docs/extending.md](docs/extending.md): adding Algorithms, endpoints and Targets
