## Agent skills

- Issue tracker: local markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.
- Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`, recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.
- Domain docs: single context, with one `GLOSSARY.md` and `docs/adr/` at the root. See `docs/agents/domain.md`.

## Development

Python 3.12 with uv; React + Vite + TS in `frontend/`. Human-facing setup and usage are in `README.md`. The current spec is `.scratch/scene-reconstruction-query/spec.md`.

- Use the `GLOSSARY.md` terms (Scene, Capture, View, Reconstruction, Mask, Instance, Query, Match) in names, docstrings and tests, and avoid the terms it lists under *Avoid*. ScanNet's own "frame" is the only exception.
- Respect `docs/adr/`. When a decision changes, update or supersede its ADR in the same change.

### Commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest                      # CPU suite; `-m gpu` = GPU smoke tests (Colab only)
npm --prefix frontend run lint && npm --prefix frontend run typecheck && npm --prefix frontend run build
SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8001   # local backend, fake Pipeline
```

### Definition of Done

1. Ruff lint and format, ty, and pytest all pass. The frontend checks also pass when `frontend/` changed.
2. Every public symbol in `src/scenetiq` has a Google-style docstring.
3. Tests are added at the right seam (see below).
4. This file is updated when a new rule or non-obvious constraint appears. Do not add anything an agent can read from the code.

### Test seams

- **Seam A**, `tests/api/`: the HTTP API through TestClient, with `FakePipeline` and `FakeTextEncoder` on a temp storage root. Most behaviour is tested here.
- **Unit**, `tests/unit/`: pure, model-free functions, wherever they live (stage modules included).
- **Seam B**, `tests/eval/` plus the ScanNet harness: the real Pipeline on GPU, marked `@pytest.mark.gpu`, deselected by default.
- Never unit-test or mock anything that loads a model (DA3, SAM2, SigLIP2, a full MV3DIS run, `RealPipeline.run`). Only Seam B covers those.
- Test expectations must come from hand-worked values, never from recomputing the code's own formula.

### Non-obvious constraints

- **GPU imports are lazy.** `torch`, `depth_anything_3`, `open3d`, `sam2` and `transformers` are imported only inside functions, so the package imports and tests on CPU. ty whitelists them in `[tool.ty.analysis]`. Never import them at module top level.
- **The GPU stack lives in `requirements-gpu.txt`, not `pyproject.toml`**, so that `uv lock` still resolves on CPU-only machines.
  - Install it with `uv pip install --system -e . -r requirements-gpu.txt`. Do not use `uv sync`: `--system` keeps Colab's CUDA-matched torch.
  - DA3 needs `numpy<2`, so keep `numpy>=1.26` in `pyproject.toml`; do not raise the lower bound.
  - Install SAM2 only from Meta's GitHub repo. The PyPI `sam2` package is a third-party upload.
- **Vendored code** is in `src/scenetiq/vendor/`, excluded from ruff and ty.
  - Each directory needs `LICENSE` and `PROVENANCE.md` (upstream URL, pinned commit, every local modification).
  - Keep changes minimal and record each one in `PROVENANCE.md`.
  - Wrappers live outside `vendor/` and are fully checked.
- **MV3DIS limits:**
  - `point_level=True` is broken upstream and would allocate N×N matrices, so always pass primitive ids.
  - The mask label map is `uint8`, so the Masking stage caps Masks at 255 per View.
- **Scores:** `TextEncoder.to_score` turns cosine similarity into the score the Match threshold applies to. SigLIP2 uses its sigmoid match probability; the fake encoder uses cosine similarity.
  - `DEFAULT_MATCH_THRESHOLD` (0.5) is a placeholder. Replace it only with the `recommended_threshold` of an accepted run of `python -m scenetiq.evaluation`, and cite that report in the commit.
- **Point order is a contract.** The PLY, `instance_ids.bin` and the MV3DIS input all use the TSDF mesh vertex order. Never reorder or filter points after fusion.
- **Deployment** is `notebooks/colab_server.ipynb` only. The notebook holds parameters; its logic belongs in `scenetiq.colab`.
- **Never commit ScanNet data or model weights.**
