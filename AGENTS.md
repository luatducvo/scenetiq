## Agent skills

- Issue tracker: local markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.
- Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`, recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.
- Domain docs: single context, with one `GLOSSARY.md` and `docs/adr/` at the root. See `docs/agents/domain.md`.

## Development

Python 3.13 with uv (Colab's Python); React + Vite + TS in `frontend/`. Human-facing setup and usage are in `README.md`. The current spec is `.scratch/scene-reconstruction-query/spec.md`.

- Use the `GLOSSARY.md` terms (Scene, Capture, View, Reconstruction, Mask, Instance, Caption, Aspect, Description, Query, Match) in names, docstrings and tests, and avoid the terms it lists under *Avoid*.
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

- **Seam A**, `tests/api/`: the HTTP API through TestClient, with `FakePipeline`, `FakeTextEncoder` and `FakeCaptionEncoder` on a temp storage root. Most behaviour is tested here.
- **Unit**, `tests/unit/`: pure, model-free functions, wherever they live (stage modules included).
- **Seam B**, `tests/eval/`: the real Pipeline on GPU, marked `@pytest.mark.gpu`, deselected by default.
- Never unit-test or mock anything that loads a model (DA3, SAM2, SigLIP2, Qwen3-VL, Qwen3-Embedding, a full MV3DIS run, `RealPipeline.run`). Only Seam B covers those. Constructing them is fine, because models load on first use.
- Test expectations must come from hand-worked values, never from recomputing the code's own formula.

### Non-obvious constraints

- **GPU imports are lazy.** `torch`, `open3d`, `sam2`, `transformers`, `xgrammar` and the vendored `scenetiq.vendor.depth_anything_3` are imported only inside functions, so the package imports and tests on CPU. ty whitelists the third-party ones in `[tool.ty.analysis]`. Never import them at module top level.
- **The GPU stack lives in `requirements-gpu.txt`, not `pyproject.toml`**, so that `uv lock` still resolves on CPU-only machines.
  - Install it with `uv pip install --system -e . -r requirements-gpu.txt`. Do not use `uv sync`: `--system` keeps Colab's CUDA-matched torch.
  - DA3 is vendored, not installed (ADR 0005): upstream pins `numpy<2` and Python `<=3.13`, which Colab no longer satisfies. Its runtime dependencies are listed by hand in `requirements-gpu.txt`. Do not add `xformers`: its wheels pin a torch version and can replace Colab's CUDA torch.
  - Install SAM2 only from Meta's GitHub repo. The PyPI `sam2` package is a third-party upload.
  - `xgrammar` and `transformers` are pinned together. Before changing either pin, check on Colab that installing it leaves `torch.__version__` unchanged.
- **TSDF uses Open3D's tensor `VoxelBlockGrid`.** The legacy `ScalableTSDFVolume` keeps almost nothing with metric depth in 0.20.0 ([Open3D#7592](https://github.com/isl-org/Open3D/issues/7592)), and no older release has Python 3.13 wheels. `tests/unit/test_fusion.py` needs open3d, so the CPU suite skips it: after changing fusion, run `uv run --with open3d==0.20.0 pytest tests/unit/test_fusion.py`.
- **Vendored code** is in `src/scenetiq/vendor/`, excluded from ruff and ty.
  - Each directory needs `LICENSE` and `PROVENANCE.md` (upstream URL, pinned commit, every local modification).
  - Keep changes minimal and record each one in `PROVENANCE.md`.
  - Wrappers live outside `vendor/` and are fully checked.
- **MV3DIS limits:**
  - `point_level=True` is broken upstream and would allocate N×N matrices, so always pass primitive ids.
  - The mask label map is `uint8`, so the Masking stage caps Masks at 255 per View.
- **Matches are top-K by Reciprocal Rank Fusion, with no threshold** (ADR 0008). A Match's score is the fused value divided by its maximum, `rankings / (rrf_k + 1)`, so it is relative to one Query and says nothing absolute. Do not reintroduce a relevance threshold or a Low-confidence Match (the candidate margin of a spatial Query, below, is not one).
  - The Caption ranking scores an Instance by its best-matching Description (max), not a mean and not one channel per Aspect. An Instance without a Caption gets only its image rank. A Reconstruction where no Instance got a Caption is normalised over one ranking.
  - **Aspect order is a storage contract:** `caption_embeddings.npy` keeps one row per Aspect in `Aspect` order, and the prompt and parser use the Aspect values as JSON keys. Never reorder or rename `Aspect` members.
  - `query_top_k`, `rrf_k` and the Captioning defaults have not been validated against ground truth, and the repo has no tool that measures them.
- **Spatial Queries are solved on stored geometry** (ADR 0009). Route only on the parsed Query's constraints, never on keywords: a Query without one must return exactly the plain Matches, with its original text.
  - Every Relation returns a score in [0, 1], never a yes/no, and measures distances against the Scene extent, never in absolute metres. Relations are not an RRF channel.
  - The candidate margin (`spatial.caption_margin`, `spatial.image_margin`) is not a threshold on Matches, so it does not break the rule above: it only limits which shortlisted Instances may stand in for an object of a Query with a Relation, relative to the best of that object's shortlist, and always leaves at least one candidate. Never apply it to a Query without a Relation, and never turn it into an absolute similarity threshold.
  - `spatial.candidates_per_object` (M), both candidate margins, `spatial.falloff` and the Relation formulas have not been validated against ground truth.
- **Configuration** lives only in `scenetiq.config`: every default, including the stage Options. Add a new setting there, and pass it to its stage in `server.build_components`. The notebook's `CONFIG` cell overrides it through `SCENETIQ_CONFIG`; only the token travels separately.
- **Captioning and Query parser replies are schema-forced** by xgrammar's transformers logits processor (ADR 0008). A processor keeps matcher state, so build a new one for every `generate` call; the compiled grammar is reused. Its vocabulary is sized by the model's output layer, not the tokenizer, which is smaller; this is not yet verified on Colab (`.scratch/aspect-captioning/issues/03-colab-acceptance.md`). Keep `parse_caption` as a second check: a reply can still be cut off at `max_new_tokens`.
- **Every Query waits for one Qwen3-VL generation**, the Query parser's (ADR 0009), including a Query without a Relation, and the first Query after a restart also pays the model's load. During a job's Captioning stage a Query waits for the current Instance's generation.
  - The parser gets the Query whole; never truncate it (the 64-token limit is SigLIP2's alone).
  - `read_parsed_query` is the parser's second check; keep it, for the same reason as `parse_caption`. A parser raises `QueryParseError` only for an unusable reply, which the endpoint answers as a plain Query; a model or CUDA failure must propagate and fail the request.
  - The text-only generation through Qwen3-VL's chat template, and the xgrammar vocabulary size with it, have not been run on Colab (`.scratch/spatial-reasoning/issues/08-colab-acceptance.md`).
- **Cancelling a job is cooperative.** The job runner sets a flag; the Pipeline stops only where it calls the reporter's `stage()` or `checkpoint()`, which raise `JobCancelledError`. Never kill or interrupt the job thread. A long loop added to a stage should take a checkpoint callable, as Masking (per View) and Captioning (per Instance) do; a single model call cannot be interrupted. A cancelled job's Scene directory is deleted, and its record and events stay.
- **GPU memory:** DA3 and SAM2 leave the GPU when their stage ends, also on failure. SigLIP2, Qwen3-Embedding and Qwen3-VL stay for Queries, so their loads take a lock (Queries and the job run on different threads). One instance of each is shared by the Pipeline and the API: build them once in `server.build_components`.
  - Qwen3-VL's lock is held for each single generation, through `ResidentModel.use()`, never for a whole stage, so a Query can generate between two Instances. The lock is not re-entrant: never generate from inside a generation.
  - With Qwen3-VL resident, a job starts with about 17 GB less free memory for DA3 and SAM2.
  - A stage's output must hold no GPU tensors, or releasing its model frees nothing.
  - Each job logs peak VRAM per stage through `scenetiq.gpu`.
- **View limits apply to uploaded photos.** Nothing checks the View count against GPU memory: DA3 takes every View left after the View stride in one forward pass, so an oversized Capture fails with OOM.
- **Up is −Y in every Reconstruction** (ADR 0007), so "above" means a smaller y. A Scene whose Views gave no consensus stays in DA3's frame, and its vertical Relations (above, below, on, under, tallest, lowest) are still computed on DA3's Y axis: they can be wrong, with no warning at Query time. Only the job log and the job's notes (Processing page) say a Scene was left unaligned.
- **SigLIP2 Query text must be truncated explicitly.** The checkpoint loads `SiglipProcessor`, which does not truncate, and the text tower has 64 positions: without `truncation=True, max_length=64` a long Query raises `ValueError`.
- **GeoCalib's `gravity.vec3d` points up, not down,** despite its name: it is `(0, -1, 0)` for a level camera. It follows a photo's roll only to about ±45°; beyond that it still answers, but wrongly (measured on one photo: 90° gave −15.5°, 180° gave −0.1°). Consensus does not catch this if every View is rolled the same way.
- **Nothing applies EXIF orientation.** DA3, the Embedding stage and the Captioning stage read a photo's raw pixels with `Image.open`, so a portrait photo stored as landscape plus an EXIF flag reaches every stage sideways.
- **Stored Scenes keep the embeddings of the models that built them.** Changing `siglip.model_id` or `caption_embedding.model_id` makes older Scenes rank wrongly, without an error.
- **`tests/unit/test_plane_search.py` also needs open3d,** so the CPU suite skips it: after changing the plane search, run `uv run --with open3d==0.20.0 pytest tests/unit/test_plane_search.py`.
- **Point order is a contract.** The PLY, `instance_ids.bin` and the MV3DIS input all use the TSDF mesh vertex order. Never reorder or filter points after fusion.
- **Request size:** Cloudflare returns 413 for large bodies (105 MB was refused), so never send a whole Capture in one request (ADR 0006).
- **Deployment** is `notebooks/colab_server.ipynb` only. The notebook holds parameters; its logic belongs in `scenetiq.colab`.
  - `scenetiq.colab`, `scenetiq.config` and `scenetiq.runtime_log` use only the standard library, because the notebook imports them before anything is installed.
- **Colab logs** are never written to files or Drive. `colab_logging.json` decides what reaches the server's stdout, which the notebook's last cell displays. Stderr is kept only to explain a crash, so anything the owner must see has to go through a logger that file lets through.
- **Never commit model weights.**
