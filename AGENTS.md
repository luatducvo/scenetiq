SceneTIQ turns uncalibrated photos into a 3D Reconstruction split into Instances that free-text Queries can find; the GPU backend runs on Colab, the React viewer runs in the browser.

Python 3.13 with **uv** (never `uv sync` on Colab, see [GPU stack](docs/agents/gpu-stack.md)); React + Vite + TS in `frontend/`.

## Commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest                      # CPU suite; `-m gpu` = GPU smoke tests (Colab only)
npm --prefix frontend run lint && npm --prefix frontend run typecheck && npm --prefix frontend run build
SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8001   # local backend, fake Pipeline
```

## Every task

- Name things with the `GLOSSARY.md` terms and steer clear of its *Avoid* list ([domain docs](docs/agents/domain.md)).
- Respect `docs/adr/`; when a decision changes, update or supersede its ADR in the same change.
- Import `torch`, `open3d`, `sam2`, `transformers`, `xgrammar` and `scenetiq.vendor.depth_anything_3` only inside functions, so the package imports and tests on CPU.
- Done means: every command above passes (frontend ones when `frontend/` changed), tests sit at the right [seam](docs/agents/testing.md), and this folder of docs records any new non-obvious constraint (never what the code already says).
- Commit no model weights.

## Read before touching

- **Tests**, choosing a seam, fakes, open3d-only suites → [testing.md](docs/agents/testing.md)
- **GPU stack**: dependencies, vendored DA3, SAM2, xgrammar pins, Open3D TSDF, VRAM and resident models → [gpu-stack.md](docs/agents/gpu-stack.md)
- **Pipeline stages**: Masking, Merging, Embedding, Captioning, alignment, point order, job cancelling → [pipeline.md](docs/agents/pipeline.md)
- **Queries**: Match scoring, Semantic Labels, the parser, spatial Relations → [query.md](docs/agents/query.md)
- **Colab deployment**, configuration, logging, uploads → [deployment.md](docs/agents/deployment.md)
- **Issues and specs** in `.scratch/<feature>/` → [issue-tracker.md](docs/agents/issue-tracker.md), [triage-labels.md](docs/agents/triage-labels.md)
- **Unverified defaults**: settings and formulas still waiting on Colab or ground truth → [unverified.md](docs/agents/unverified.md)
