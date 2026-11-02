# SceneTIQ

SceneTIQ turns a set of ordinary 2D photos of a static scene into a metric 3D point cloud, splits it into individual objects, and lets you find those objects with free-text queries such as "red chair" or "the potted plant".

You upload 10–300 photos and SceneTIQ estimates the camera poses itself. Search is open-vocabulary: it is not limited to a fixed list of object classes.

> **Status:** The backend API, storage, job handling and web frontend are implemented and tested on CPU, using a fake pipeline. The real GPU pipeline (Depth Anything 3, SAM2, MV3DIS, SigLIP2) is implemented but **has not yet been run end to end on a GPU**. Expect to fix installation or runtime issues on the first Colab run. The Match threshold (`0.5`) is a placeholder until the evaluation harness has been run.

## How it works

```text
Photos ──► Depth Anything 3 ──► TSDF fusion ──► ScanNet Segmentator ──► SAM2 masks ──► MV3DIS ──► SigLIP2
(Capture)  metric depth +       mesh / point    superpoints              per photo      3D objects  embedding
           camera poses         cloud                                                   (Instances) per object
```

| Stage | Model / method | Notes |
|---|---|---|
| Reconstructing | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) `DA3NESTED-GIANT-LARGE` | Depth in metres, no camera poses needed |
| Fusing | TSDF fusion (Open3D), 2 cm voxels | The mesh vertices become the point cloud |
| Superpoints | ScanNet Segmentator (ported to Python) | Groups points by surface normal |
| Masking | [SAM2](https://github.com/facebookresearch/sam2) automatic mask generation | Prompt-free, so no vocabulary limit |
| Merging | [MV3DIS](https://github.com/zybjn/MV3DIS) 3D-guided mask matching (vendored) | Class-agnostic Instances |
| Embedding | [SigLIP2](https://huggingface.co/google/siglip2-giant-opt-patch16-384) giant | Multi-view crops averaged per Instance |

A Query encodes your text with the same SigLIP2 model and scores every Instance. You get back every Match above the threshold. If none qualifies, you get the single closest Instance, flagged as a *Low-confidence Match*.

The terms used here (Scene, Capture, View, Reconstruction, Instance, Query, Match) are defined in [GLOSSARY.md](GLOSSARY.md). The key design decisions are recorded in [docs/adr/](docs/adr/).

### System layout

```text
┌──────────────────────┐   HTTPS + bearer token   ┌───────────────────────────────┐
│ React frontend       │ ───────────────────────► │ Cloudflare named tunnel       │
│ (your machine,       │                          └──────────────┬────────────────┘
│  localhost:5173)     │                                         ▼
└──────────────────────┘                          ┌───────────────────────────────┐
                                                  │ Google Colab (A100 80 GB)     │
                                                  │  FastAPI backend + GPU models │
                                                  │  Scenes + SQLite on Google    │
                                                  │  Drive (survive restarts)     │
                                                  └───────────────────────────────┘
```

## Requirements

- **Local machine:** Python 3.12 with [uv](https://docs.astral.sh/uv/), and Node.js 20+ with npm.
- **GPU backend:** a Google Colab runtime with an A100 80 GB GPU and Google Drive.
- **Remote access:** a domain managed by Cloudflare, for a stable tunnel URL.
- **Disk:**
  - Model weights download automatically from Hugging Face on first use: about 15 GB per Colab session (DA3 6.8 GB, SigLIP2 7.5 GB, SAM2 0.9 GB). None of them are gated.
  - Every Scene keeps its original photos on Drive. Plan on about 1–1.5 GB for each 300-photo Scene.

## Quick start: run locally without a GPU

The backend can run with a **fake pipeline**. It returns a small fixed Reconstruction, so you can try the full upload → progress → view → query flow on any machine.

```bash
# Backend (terminal 1)
uv sync
SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8000

# Frontend (terminal 2)
npm --prefix frontend install
npm --prefix frontend run dev
```

1. Open http://localhost:5173.
2. In **Settings**, set the URL to `http://localhost:8000` and the token to `dev`.
3. Under **New Scene**, upload 10 or more JPG/PNG images. Any images will do, because the fake pipeline ignores their content.
4. Open the Scene and try the Queries `red chair`, `table`, `lamp` or `chair`. Any other text returns a Low-confidence Match.

On Windows PowerShell, set the token with `$env:SCENETIQ_TOKEN="dev"` before the `uv run` line.

## Deploying the GPU backend on Colab

### One-time setup

1. **Push this repository to GitHub.** The notebook clones `REPO_URL` / `BRANCH`. Both default to `https://github.com/luatducvo/scenetiq.git` and `main`. The repository must be public, or you must adapt the clone step to use a token.
2. **Create a Cloudflare named tunnel:**
   1. In Cloudflare Zero Trust, go to **Networks → Tunnels → Create a tunnel → Cloudflared**.
   2. Add a public hostname (for example `scenetiq.example.com`) with service `http://localhost:8000`.
   3. Copy the tunnel token.
3. **Add Colab secrets.** Open the 🔑 sidebar, add both secrets, and enable notebook access for each:
   - `SCENETIQ_TOKEN`: any long random string. You will enter the same value in the frontend.
   - `CLOUDFLARE_TUNNEL_TOKEN`: the token from step 2.
4. **Pick the runtime:** go to **Runtime → Change runtime type**, and choose GPU (A100).

There is nothing to create on Google Drive in advance. The notebook creates `MyDrive/scenetiq/data` (Scenes and the SQLite database) and `MyDrive/scenetiq/logs`.

### Every session

1. Open [`notebooks/colab_server.ipynb`](notebooks/colab_server.ipynb) in Colab.
2. Check the parameters cell. `PIPELINE = "real"` uses the GPU models; `"fake"` checks Drive, the tunnel and the frontend without downloading any models. Using `"fake"` is recommended for your very first run.
3. Run all cells. The notebook:
   1. mounts Drive
   2. clones or updates the repo
   3. installs the package with `uv pip install --system` (adding [`requirements-gpu.txt`](requirements-gpu.txt) when `PIPELINE` is `"real"`). This installs into Colab's own Python, so Colab's preinstalled CUDA build of torch is kept
   4. starts the API and waits until it is ready
   5. starts the tunnel
4. On your machine, run the frontend (`npm --prefix frontend run dev`). In **Settings**, set the URL to `https://<your tunnel hostname>` and the token to your `SCENETIQ_TOKEN`.

If the Colab runtime disconnects, run the notebook again. Finished Scenes come back at the same URL. A job that was running when the runtime died is shown as *failed (interrupted by restart)*; upload it again.

If the frontend is served from somewhere other than `localhost:5173`, add its origin to `SCENETIQ_CORS_ORIGINS`.

## Using SceneTIQ

### Taking good photos

- **Static scenes only.** Nothing should move between shots.
- **10–300 photos.** Walk around the space and keep plenty of overlap between neighbouring photos.
- Vary the viewpoint, not just the rotation: move between shots instead of spinning on the spot.
- Use even lighting and avoid motion blur.

### Workflow

1. **New Scene:** give the Scene a name and select its photos. The browser checks the count (10–300) and the file type (JPG/PNG) before uploading.
2. **Processing:** the progress view shows the current stage: reconstructing → fusing → superpoints → masking → merging → embedding. Only one job runs at a time; a second upload is refused while a job runs.
3. **Viewer:**
   - Drag to orbit, right-drag to pan, scroll to zoom.
   - Type a Query and press **Query**. The points of each Match are highlighted and boxed, and a ranked list shows the scores.
   - A **low confidence** badge means nothing cleared the threshold; you are seeing the closest Instance.
4. **Scenes:** the landing page lists every Scene. Open finished Scenes or delete ones you no longer need. A Scene cannot be deleted while its job is running.

### HTTP API

Every request needs `Authorization: Bearer <SCENETIQ_TOKEN>`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/scenes` | Multipart `name` + `images` (10–300 JPG/PNG). Returns `{scene_id, job_id}` |
| `GET` | `/jobs/{job_id}` | `{status, stage, error}`; `status` is `queued`, `running`, `succeeded` or `failed` |
| `GET` | `/scenes` | All Scenes, newest first |
| `DELETE` | `/scenes/{scene_id}` | Delete a Scene and its data |
| `GET` | `/scenes/{scene_id}/points` | Point cloud as binary PLY (xyz in metres, rgb) |
| `GET` | `/scenes/{scene_id}/instance-ids` | Little-endian int32 Instance id per point (PLY order, `-1` = none) |
| `GET` | `/scenes/{scene_id}/instances` | `[{instance_id, centroid, bbox_min, bbox_max}]` in metres |
| `POST` | `/scenes/{scene_id}/query` | Body `{"text": "red chair"}` → `{matches: [{instance_id, score}], low_confidence}` |

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SCENETIQ_TOKEN` | *(required)* | Shared bearer token |
| `SCENETIQ_STORAGE_ROOT` | `./data` | Holds the SQLite database and one directory per Scene |
| `SCENETIQ_PIPELINE` | `fake` | `fake` or `real` (the notebook sets `real`) |
| `SCENETIQ_MATCH_THRESHOLD` | `0.5` | Minimum score for a Match. With the real pipeline, the score is SigLIP2's match probability (0–1) |
| `SCENETIQ_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins |

## Testing and evaluation

### CPU tests (any machine)

```bash
uv run pytest                    # API with a fake pipeline, plus pure-function unit tests
```

### GPU smoke tests (Colab)

You need a folder of photos of one room. Pick an object that is clearly visible, and note its bounding box in one photo. That photo's `view` index is its position when the photos are sorted by file name.

```bash
SCENETIQ_SMOKE_CAPTURE=/content/drive/MyDrive/smoke_room \
SCENETIQ_SMOKE_QUERY="chair" \
SCENETIQ_SMOKE_QUERY_BOX="12,830,410,1290,1050" \
SCENETIQ_SMOKE_ABSENT_QUERY="giraffe" \
pytest -m gpu -s
```

`SCENETIQ_SMOKE_QUERY_BOX` is `view,left,top,right,bottom`, in pixels of the original photo. `SCENETIQ_SMOKE_ABSENT_QUERY` is optional. The Query test requires the following:

- a confident Match at the configured threshold
- at least 20 % of the best Match's points project into that photo
- at least half of those land inside the box

### ScanNet200 evaluation harness (Colab)

This measures class-agnostic instance AP (AP / AP50 / AP25) on real scans and recommends a Match threshold.

1. Accept the [ScanNet](http://www.scan-net.org/) Terms of Use and download 5–10 validation scenes. For each scene you need `<scene>_vh_clean_2.ply`, `<scene>.aggregation.json`, `<scene>_vh_clean_2.0.010000.segs.json` and the `.sens` file.
2. Export the RGB frames and poses with ScanNet's SensReader, into `<frames>/<scene>/color/*.jpg` and `<frames>/<scene>/pose/*.txt`.
3. Run:

   ```bash
   python -m scenetiq.evaluation \
     --scans-dir /data/scannet/scans --frames-dir /data/scannet/frames \
     --scenes scene0011_00,scene0015_00 \
     --out /content/drive/MyDrive/scenetiq/eval/run1 \
     [--baseline /content/drive/MyDrive/scenetiq/eval/baseline/report.json]
   ```

The harness uses only RGB: each scene is sampled down to at most 300 frames, and ScanNet's depth and poses are hidden from the pipeline. It writes `report.md` and `report.json`. With `--baseline`, it exits non-zero if AP drops by more than `--tolerance` (default 0.02). To use the result, set `SCENETIQ_MATCH_THRESHOLD`, or `DEFAULT_MATCH_THRESHOLD` in `src/scenetiq/config.py`, to the `recommended_threshold` the report gives.

Never commit or redistribute ScanNet data.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run ty check                                       # type check
uv run pytest                                         # tests
npm --prefix frontend run lint && npm --prefix frontend run typecheck && npm --prefix frontend run build
```

[AGENTS.md](AGENTS.md) holds the project conventions: layout, test seams, Definition of Done, and the vendoring rules. The feature spec and implementation tickets are in [`.scratch/scene-reconstruction-query/`](.scratch/scene-reconstruction-query/).

```text
src/scenetiq/          API, storage, jobs, Query, domain types
src/scenetiq/pipeline/ real GPU pipeline, one module per stage
src/scenetiq/evaluation/ ScanNet200 evaluation harness
src/scenetiq/vendor/   vendored MV3DIS and ScanNet Segmentator (see PROVENANCE.md)
frontend/              React + Vite + react-three-fiber client
notebooks/             Colab bootstrap notebook
tests/unit, tests/api, tests/eval   pure functions, API (fake pipeline), GPU smoke tests
```

## Licensing

- **Depth Anything 3:**
  - The code is Apache-2.0.
  - The `DA3NESTED-GIANT-LARGE` weights are licensed **CC BY-NC 4.0**, so a SceneTIQ deployment with the real pipeline is limited to **non-commercial use** ([ADR 0001](docs/adr/0001-non-commercial-da3-nested.md)).
- **SAM2:** Apache-2.0. Install it only from Meta's GitHub repository; the `sam2` package on PyPI is a third-party upload.
- **SigLIP2:** Apache-2.0.
- **Vendored code:** MV3DIS (MIT) and the ScanNet Segmentator (MIT-style ScanNet licence). See the `LICENSE` and `PROVENANCE.md` files under `src/scenetiq/vendor/`.
- **ScanNet data:** subject to the ScanNet Terms of Use, and never included in this repository.
