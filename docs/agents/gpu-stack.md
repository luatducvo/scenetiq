# GPU stack

## Dependencies

The GPU stack lives in `requirements-gpu.txt`, not `pyproject.toml`, so `uv lock` still resolves on CPU-only machines. ty whitelists the lazily imported third-party packages in `[tool.ty.analysis]`.

- Install with `uv pip install --system -e . -r requirements-gpu.txt`. `--system` keeps Colab's CUDA-matched torch, which `uv sync` would replace.
- DA3 is vendored, not installed (ADR 0005). Its runtime dependencies are listed by hand in `requirements-gpu.txt`. Leave `xformers` out: its wheels pin a torch version and can replace Colab's CUDA torch.
- Install SAM2 from Meta's GitHub repo only; the PyPI `sam2` package is a third-party upload.
- `xgrammar` and `transformers` are pinned together. Before changing either pin, check on Colab that installing it leaves `torch.__version__` unchanged.

## Vendored code

Vendored code is in `src/scenetiq/vendor/`, excluded from ruff and ty.

- Each directory needs `LICENSE` and `PROVENANCE.md` (upstream URL, pinned commit, every local modification).
- Keep changes minimal and record each one in `PROVENANCE.md`.
- Wrappers live outside `vendor/` and are fully checked.

## TSDF

Fusion uses Open3D's tensor `VoxelBlockGrid`. The legacy `ScalableTSDFVolume` keeps almost nothing with metric depth in 0.20.0 ([Open3D#7592](https://github.com/isl-org/Open3D/issues/7592)), and no older release has Python 3.13 wheels.

## Memory and resident models

- DA3 and SAM2 leave the GPU when their stage ends, also on failure.
- SigLIP2, Qwen3-Embedding and Qwen3-VL stay resident for Queries. Build one instance of each in `server.build_components`, shared by the Pipeline and the API; their loads take a lock, because Queries and the job run on different threads.
- Qwen3-VL's lock is held for one generation at a time, through `ResidentModel.use()`, so a Query can generate between two Instances. The lock is not re-entrant: each generation runs outside any other.
- With Qwen3-VL resident, a job starts with about 17 GB less free memory for DA3 and SAM2.
- A stage's output holds only CPU data; a GPU tensor left in it keeps its model's memory allocated.
- Nothing checks the View count against GPU memory: DA3 takes every View left after the View stride in one forward pass, so an oversized Capture fails with OOM.
