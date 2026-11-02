## Agent skills

### Issue tracker

Issues and specs are tracked as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Development

Python 3.12, managed with uv. Use the vocabulary in `GLOSSARY.md` (Scene, Capture, View, Reconstruction, Mask, Instance, Query, Match) in names, docstrings and tests. Respect the decisions in `docs/adr/`. The current feature spec is `.scratch/scene-reconstruction-query/spec.md`.

### Commands

| Purpose | Command |
|---|---|
| Install / sync dev env | `uv sync` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run ty check` |
| Tests (CPU, default) | `uv run pytest` |
| GPU tests (Colab only) | `uv run pytest -m gpu` |
| Run backend locally | `SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8000` |
| Frontend install | `npm --prefix frontend install` |
| Frontend dev server | `npm --prefix frontend run dev` (http://localhost:5173) |
| Frontend lint | `npm --prefix frontend run lint` |
| Frontend type check | `npm --prefix frontend run typecheck` |
| Frontend build | `npm --prefix frontend run build` |

### Configuration

The backend reads environment variables (see `scenetiq.config.Settings`):

| Variable | Default | Meaning |
|---|---|---|
| `SCENETIQ_TOKEN` | (required) | Shared bearer token every request must send |
| `SCENETIQ_STORAGE_ROOT` | `./data` | SQLite database plus one directory per Scene |
| `SCENETIQ_MATCH_THRESHOLD` | `0.5` | Minimum relevance score for a Match |
| `SCENETIQ_PIPELINE` | `fake` | `fake` (no models) or `real` (GPU model stack) |
| `SCENETIQ_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated browser origins |

### Docstrings

Google convention, enforced by ruff's `D` rules. Every public module, class, function and method in `src/scenetiq` needs a docstring. Tests are exempt.

### Layout

- `src/scenetiq/`: first-party package.
  - `domain.py`: the Capture, View, Camera, Instance, Reconstruction and Stage types, plus the `Pipeline` and `TextEncoder` protocols.
  - `api.py`: the FastAPI app factory `create_app(settings, pipeline, text_encoder)`.
  - `server.py`: the process entry point, which reads env config and picks the Pipeline.
  - `storage.py`: SQLite metadata and per-Scene directories.
  - `artifacts.py`: the on-disk Reconstruction format.
  - `jobs.py`: the single-slot background job runner.
  - `fakes.py`: the deterministic fake Pipeline and text encoder.
  - `config.py`: `Settings` read from `SCENETIQ_*` environment variables, and the default Match threshold.
  - `colab.py`: helpers the Colab notebook calls (server process, cloudflared tunnel).
  - `pipeline/`: the real GPU Pipeline (see *Real Pipeline* below).
  - `evaluation/`: the ScanNet200 evaluation harness (see *Evaluation harness* below).
  - `query.py`: scores Instances against a Query embedding, applies the Match threshold, and returns the Low-confidence Match fallback.
  - `geometry.py`: Instance centroid and bounding-box computation, plus L2 normalisation.
- `frontend/`: React + Vite + TypeScript client. It has no router; `App.tsx` switches between screens.
- `src/scenetiq/vendor/`: vendored third-party code. Excluded from ruff and ty. Every subdirectory needs a `PROVENANCE.md` with the upstream URL, pinned commit, licence and local modifications. First-party wrappers around vendored code live outside `vendor/` and are fully checked.
- `tests/unit/`: deterministic pure functions that need no model and no I/O beyond temporary files: coordinate and unit adapters, geometry, the pure helpers inside stage modules (Mask selection, primitives, superpoints, crop and view selection, label conversion), and evaluation metrics.
- `tests/api/`: **Seam A**. The HTTP API driven through FastAPI's TestClient, with a fake Pipeline and a fake text encoder on a temporary storage root. This is the main suite and must pass on CPU.
- `tests/eval/`: **Seam B**. The real Pipeline on a GPU, marked `@pytest.mark.gpu` and deselected by default.

Test only at these seams. A pure, model-free function may be unit-tested wherever it lives, including inside a stage module. Do not unit-test anything that loads a model or needs a GPU (DA3, SAM2, SigLIP2, the whole MV3DIS run, `RealPipeline.run`); those are covered only by Seam B. Never mock a model to test a stage's internals.

### Definition of Done (every ticket)

1. `uv run ruff check .` and `uv run ruff format --check .` pass.
2. `uv run ty check` passes with zero diagnostics.
3. Every new public first-party symbol has a Google-style docstring.
4. Tests are added at the right seam, and `uv run pytest` passes.
5. If the ticket touches `frontend/`: `npm --prefix frontend run lint`, `run typecheck` and `run build` pass.
6. AGENTS.md is updated whenever the ticket adds a command, directory, convention or configuration value that a future agent needs to know.

### Colab (backend deployment, ADR 0004)

`notebooks/colab_server.ipynb` is the only way the backend is deployed. It holds parameters and calls into `scenetiq.colab`; put any new logic in the package, not the notebook. Notebooks are excluded from ruff and ty.

One-time setup:
1. In the Cloudflare Zero Trust dashboard, create a named tunnel (Networks → Tunnels → Create → Cloudflared). Add a public hostname (e.g. `scenetiq.<your-domain>`) whose service is `http://localhost:8000`. Copy the tunnel token.
2. In Colab, add the secrets `SCENETIQ_TOKEN` (any long random string, also entered in the frontend Settings) and `CLOUDFLARE_TUNNEL_TOKEN`, and grant the notebook access to both.
3. Pick a GPU runtime.

Each run: open the notebook, set `PIPELINE` (`fake` or `real`) and run all cells. Scenes, the SQLite database and logs live under `DRIVE_ROOT` on Google Drive, so they survive runtime restarts. Jobs that were running when the runtime died are marked failed at the next start. Then run the frontend locally with the backend URL set to `https://<tunnel hostname>`. Add the frontend's origin to `SCENETIQ_CORS_ORIGINS` if it is not on `localhost:5173`.

### Real Pipeline (GPU)

- `src/scenetiq/pipeline/` holds the real Pipeline. `real.py` chains the stages. `view_data.py` defines `ViewData` (images, depth and cameras of every View at the depth resolution) and `FusedGeometry`, the types passed between stages. `adapters.py` holds the pure conversions (unit-tested): DA3 world-to-camera ↔ camera-to-world, and confidence filtering of depth. Each stage file holds one stage: `reconstruct.py` (DA3, ADR 0001), `fusion.py` (TSDF, ADR 0002), `superpoints.py` (ScanNet Segmentator ported to Python, on the fused mesh), `primitives.py` (voxel-grid fallback, `RealPipelineOptions.primitives="voxel"`), `masking.py` (SAM2 automatic Masks, ADR 0003, at most 255 per View), `merging.py` (wraps vendored MV3DIS; turns its labels into Instance ids), `embedding.py` (multi-view crops from the original photos at two context scales, averaged) and `siglip.py` (SigLIP2 giant: one model encodes both Instance crops and Query text).
- Model libraries (`torch`, `depth_anything_3`, `open3d`, `sam2`, `transformers`) are **imported lazily inside functions**, so the package still imports and tests on CPU. ty is told to allow these unresolved imports in `[tool.ty.analysis]`.
- The GPU stack is pinned in `requirements-gpu.txt`, not in `pyproject.toml`, because `uv lock` must stay resolvable on CPU-only machines. The notebook installs it when `PIPELINE = "real"`. On a GPU machine: `pip install -e . -r requirements-gpu.txt`.
- GPU smoke tests: `SCENETIQ_SMOKE_CAPTURE=/path/to/photos uv run pytest -m gpu`. They are skipped when the variable is unset.
- Tuning knobs live in the stage option dataclasses (`DA3Options`, `FusionOptions`). Defaults: process at 504 px, drop depth below the 30th confidence percentile or beyond 12 m, 2 cm TSDF voxels.
- Relevance scores: `TextEncoder.to_score` maps cosine similarity to the score that the Match threshold applies to. SigLIP2 uses its own `sigmoid(scale * cos + bias)`, a match probability in [0, 1]. The fake encoder uses cosine similarity as is. Instances that no View sees well enough get no embedding and are dropped.
- The end-to-end Query smoke tests use these variables:
  - `SCENETIQ_SMOKE_QUERY=<object clearly visible in the smoke Capture>`.
  - `SCENETIQ_SMOKE_QUERY_BOX=view,left,top,right,bottom`: the object's box, in original-photo pixels, in one photo. `view` is the photo's index in the Capture sorted by file name.
  - With both set, the test requires a confident Match (no Low-confidence fallback) at the configured threshold, and at least half of the best Match's points must project inside the box.
  - Optionally, `SCENETIQ_SMOKE_ABSENT_QUERY=<object not in the Capture>` requires that Query to come back Low-confidence.

### Evaluation harness (Seam B, GPU)

`python -m scenetiq.evaluation --scans-dir <scannet>/scans --frames-dir <exported frames> --scenes <id,id,...> --out <dir> [--baseline <earlier report.json>] [--tolerance 0.02]`

- Data: ScanNet v2 `scans/<scene>/` (`_vh_clean_2.ply`, `.aggregation.json`, `_vh_clean_2.0.010000.segs.json`) plus 2D frames exported from the `.sens` files with ScanNet's SensReader (`<scene>/color/<i>.jpg`, `<scene>/pose/<i>.txt`). Put them on Google Drive or local Colab disk. **Never commit or redistribute ScanNet data**; it is under the ScanNet Terms of Use.
- Per scene: sample ≤300 valid frames evenly, run the real Pipeline on RGB only, align the Reconstruction to ScanNet with a Umeyama similarity on camera centres, transfer Instances to GT vertices (nearest neighbour ≤10 cm), and compute class-agnostic AP / AP50 / AP25. Walls, floors and ceilings are ignored. Then run one Query per GT label; a Match counts as right at IoU ≥ 0.5 with a GT instance of that label.
- Output: `report.json` and `report.md` with per-scene and aggregate metrics and the F1-optimal threshold. With `--baseline` the command exits 1 if any metric drops more than `--tolerance`. Keep the last accepted `report.json` as the baseline.
- **Updating the threshold:** after an accepted run, set `DEFAULT_MATCH_THRESHOLD` in `src/scenetiq/config.py` (or `SCENETIQ_MATCH_THRESHOLD` on Colab) to `recommended_threshold`, and note the report it came from in the commit message. Until a run has been made, the default is 0.5 on SigLIP2's sigmoid match probability.
- The pure parts (`sampling`, `alignment`, `transfer`, `metrics`, `scannet` loaders) have CPU unit tests in `tests/unit/test_evaluation.py`.
