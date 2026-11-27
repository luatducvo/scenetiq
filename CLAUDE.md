SceneTIQ turns uncalibrated photos into a 3D Reconstruction split into Instances that free-text Queries can find; the GPU backend runs on a GPU runtime, the React viewer runs in the browser.

Python 3.13 with **uv** (never `uv sync` on the GPU runtime, see [GPU stack](#gpu-stack)); React + Vite + TS in `frontend/`.

## Commands

```bash
uv run pre-commit install         # once per clone: lint + ty on commit, pytest + frontend build on push
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest                      # unit + integration, on CPU
uv run pytest evaluation           # evaluation suite: the real Pipeline (GPU runtime only)
npm --prefix frontend run lint && npm --prefix frontend run typecheck && npm --prefix frontend run build
SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8001   # local backend, fake Pipeline
```

## Every task

- Name things with the `GLOSSARY.md` terms and steer clear of its *Avoid* list.
- Respect `docs/adr/`; when a decision changes, update or supersede its ADR in the same change.
- Import `torch`, `open3d`, `sam2`, `transformers`, `xgrammar` and `depth_anything_3` only inside functions, so the package imports and tests on CPU.
- Done means: every command above passes (frontend ones when `frontend/` changed), tests sit at the right [layer](#tests), and this file records any new non-obvious constraint (never what the code already says).
- Commit no model weights.
- Write no docstrings (outside `src/scenetiq/vendor/`); when the *why* is not obvious, use a `#` comment.

## Agent skills

### Issue tracker

Local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the root. See `docs/agents/domain.md`.

## Constraints the code does not show

### Tests

- Three layers (ADR 0008). **Unit**, `tests/unit/`: pure, model-free functions. **Integration**, `tests/integration/`: the HTTP API through TestClient, with the fakes of `scenetiq.fakes` on a temp storage root; most behaviour is tested here. **Evaluation**, `evaluation/` at the root, outside `testpaths`: the real Pipeline on a GPU, skipped as a whole without CUDA.
- Model-loading code (DA3, SAM2, SigLIP2, Qwen3-VL, Qwen3-Embedding, MV3DIS, `RealPipeline.run`) is covered by the evaluation suite alone. Elsewhere stand it in only with `scenetiq.fakes`, at the integration layer. Constructing those classes is fine anywhere: models load on first use.
- Shared test helpers live in a `helpers.py` or `support.py` next to the tests; a `conftest.py` holds fixtures only and is never imported.
- Derive every expectation from hand-worked values, so a test checks the formula instead of restating it.
- `tests/unit/test_fusion.py` and `test_plane_search.py` need open3d: `uv run --with open3d==0.20.0 pytest tests/unit/test_fusion.py tests/unit/test_plane_search.py`.

### GPU stack

- The stack is the `gpu` extra, locked in `uv.lock`; `uv sync` does not select it. Never `uv sync` on the GPU runtime: the lock's torch would replace the runtime's CUDA build. `install_packages` exports the lock (`--frozen`) without the torch family and holds the runtime's `torch` and `torchvision` only (its full freeze pins a clashing `transformers`). Rerun `uv lock` after changing the extra.
- DA3 comes from our fork pinned by commit (ADR 0001); its `requires-python <3.14` caps ours. Leave `xformers` out: its wheels pin a torch.
- SAM2 only from Meta's GitHub (PyPI `sam2` is a third-party upload). Its metadata in `[[tool.uv.dependency-metadata]]` must follow `setup.py` when the pin moves. It builds with `SAM2_BUILD_CUDA=0` and `--no-build-isolation-package sam-2` on the command line, since the notebook does not run uv from the checkout; `sam2._C` is unused.
- `xgrammar` and `transformers` are pinned together; before moving either, check on the GPU runtime that `torch.__version__` stays the same.
- Fusion uses Open3D's tensor `VoxelBlockGrid`: the legacy `ScalableTSDFVolume` keeps almost nothing with metric depth in 0.20.0 ([Open3D#7592](https://github.com/isl-org/Open3D/issues/7592)).
- `src/scenetiq/vendor/` (excluded from ruff and ty): each directory has `LICENSE` and `PROVENANCE.md` recording every local change; wrappers live outside it.
- DA3 and SAM2 leave the GPU when their stage ends, also on failure. SigLIP2, Qwen3-Embedding and Qwen3-VL stay resident (about 17 GB less for DA3 and SAM2), built once in `server.build_components` and shared by the Pipeline and the API. Their loads take a lock, because Queries and the job run on different threads; Qwen3-VL's lock is held per generation through `ResidentModel.use()` and is not re-entrant.
- A stage's output holds only CPU data. Nothing checks the View count against GPU memory: DA3 takes every View in one pass.

### Pipeline

- Points keep the TSDF mesh vertex order and count from fusion onward (ADR 0002): never reorder or filter them.
- Always pass MV3DIS primitive ids; `point_level=True` is broken upstream.
- `plan_views` needs the fusion voxel size to give splat radii; without it the Embedding and Captioning stages silently fall back to box crops. A stage that changes the point spacing must pass the new one.
- Schema-forced generation (Captioning and the Query parser): build a new xgrammar `LogitsProcessor` per `generate` and reuse the compiled grammar; size its vocabulary by the model's output layer, not the tokenizer. Keep `parse_caption` and `read_parsed_query` as second checks: a reply can be cut off at `max_new_tokens`.
- `Aspect` order is a storage contract: `caption_embeddings.npy` keeps one row per Aspect in that order. Never reorder or rename its members.
- GeoCalib's `gravity.vec3d` points up despite its name.
- Cancelling is cooperative: the Pipeline stops only at the reporter's `stage()` or `checkpoint()`. A long loop added to a stage takes a checkpoint callable. A cancelled job's Scene directory is deleted; its record and events stay.

### Queries

- Stored Scenes keep the embeddings of the models that built them: changing `siglip.model_id` or `caption_embedding.model_id` ranks older Scenes wrongly, without an error.
- SigLIP2 text is truncated explicitly (`truncation=True, max_length=64`): `SiglipProcessor` does not truncate. The parser gets the Query whole.
- The parser raises `QueryParseError` only for an unusable reply, answered as a plain Query; a model or CUDA failure fails the request.
- Route on the parsed Query, never on keywords. Relation scores stay in [0, 1] and relative to the Scene extent; candidate margins apply only to Queries with a Relation and always leave a candidate; `query_score_ratio` never cuts a spatial Query's phrase rankings.

### Deployment and configuration

- Deployment is the notebook in `notebooks/` only; its logic belongs in `scenetiq.deployment`. That package, `scenetiq.config` and `scenetiq.runtime_log` use only the standard library: the notebook imports them before installing anything.
- Logs never go to files. `src/scenetiq/deployment/logging.json` decides what reaches stdout, so anything the owner must see goes through a logger it lets through.
- Every default lives in `scenetiq.config`; pass a new setting to its stage in `server.build_components`. The notebook overrides them through `SCENETIQ_CONFIG`; only the token travels separately.

### Unverified

Nothing has run end to end on a GPU. The ranking and spatial defaults named in ADR 0006 and ADR 0007, and DA3 from the fork on numpy 2 without xformers, are guesses until `pytest evaluation` and `python -m scenetiq.evaluation` say otherwise.
