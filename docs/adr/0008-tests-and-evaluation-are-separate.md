# Tests are unit and integration; evaluation is separate

The checks split by what they need. `tests/unit/` holds pure, model-free functions; `tests/integration/` drives the HTTP API through TestClient with the fakes of `scenetiq.fakes`. Both run on CPU, on every push. `evaluation/` at the root runs the real Pipeline and its models on a GPU runtime and is not in `testpaths`, so `uv run pytest` never collects it; `uv run pytest evaluation` runs it, and without CUDA it skips as a whole. Scoring Queries against hand-labelled Instances is product code, `scenetiq.evaluation` (`python -m scenetiq.evaluation` against a running backend), and is itself tested at both CPU layers.

## Considered Options

- One `tests/` tree with a `gpu` marker deselected by default: the boundary hid in `addopts`, `tests/eval/` was easy to confuse with `scenetiq.evaluation`, and a forgotten marker pulled model loads into the CPU run.
- Testing the deployment code (subprocesses, a real uvicorn server, the tunnel): slow and tied to the GPU runtime's process table; the notebook is its check, so `scenetiq.deployment` has no tests.

## Consequences

- The evaluation suite shares its models across files through session fixtures; a test that loads a second copy of a model must free it before it returns.
- Nothing on CPU checks `scenetiq.deployment`; a change there is verified by running the notebook.
