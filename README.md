# 3D Scene Understanding

Turn photos or one short video of a room into a metric 3D Scene: a point cloud and mesh, the Objects in it with their sizes, and text Search over those Objects.

The pipeline runs Depth Anything 3 for reconstruction, GeoCalib for gravity alignment, TSDF fusion for the mesh, SAM 3 for open-vocabulary segmentation, mask clustering to lift masks into 3D Objects, Qwen3-VL to write each Object's Object Description, bge-m3 to embed it and bge-reranker-v2-m3 to rerank Search Hits. Every model Stage also has a fake that runs on a CPU, which the tests use.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv)
- Node.js 24 and npm, for the frontend
- A Google account with Colab, for the backend; real models need an A100 runtime
- Docker, only for `uv run main.py test`

## Quick start

The backend (Postgres, scene-api, scene-worker and scene-compute) runs on one Google Colab runtime; only the frontend runs on your machine.

1. Run `backend/deploy/colab/GPU.ipynb` following [its README](backend/deploy/colab/README.md). Step 8 prints the API's public URL.
2. On your machine:

   ```bash
   cd frontend
   npm install
   cp .env.example .env   # set VITE_API_URL to the URL from step 8
   npm run dev
   ```

3. Open http://localhost:3000.

Every Scene lives only as long as the Colab runtime: restarting the backend cell or the runtime deletes them all.

`uv run main.py --help` lists every command, including `test` and `lint`.

To remove generated files on the host, run either command from the repository root:

```bash
uv run main.py clean python  # Python/tool caches and bytecode; also clears the global uv cache
uv run main.py clean react   # Frontend build output, Vite/Next.js caches and TypeScript build metadata
```

These commands keep `.venv`, installed dependencies in `node_modules`, source files, environment files and project data.

## Frontend only

```bash
cd frontend
npm install
VITE_API_MODE=mock npm run dev
```

Mock mode serves a demo Scene from the browser; no backend is needed. In PowerShell, use `$env:VITE_API_MODE="mock"; npm run dev`.

The frontend is a React SPA built with Vite, React Router and Three.js through React Three Fiber. `npm run build` produces `frontend/dist/`. Set `VITE_API_URL` and `VITE_API_MODE` before building.

## Documentation

- [GLOSSARY.md](GLOSSARY.md): the domain terms
- [docs/architecture/overview.md](docs/architecture/overview.md): how it fits together
- [docs/extending.md](docs/extending.md): adding Algorithms, endpoints and Targets
