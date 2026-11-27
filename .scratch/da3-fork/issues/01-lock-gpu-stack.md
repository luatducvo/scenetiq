# 01: Lock the GPU stack and install it on Colab from the lock

**What to build:** The GPU stack moves from `requirements-gpu.txt` into a `gpu` extra resolved in `uv.lock`, and the Colab install uses the exported lock while holding Colab's torch. DA3 stays vendored in this ticket; its dependencies move into the extra as they are. See "The `gpu` extra" and "Installing on Colab from the lock" in `.scratch/da3-fork/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] `pyproject.toml` has a `gpu` extra with everything `requirements-gpu.txt` lists, and `requirements-gpu.txt` is deleted.
- [x] SAM2's static metadata is in `[[tool.uv.dependency-metadata]]`, and `uv lock` succeeds on a CPU-only machine without building SAM2.
- [x] `uv sync` still installs no torch.
- [x] `scenetiq.colab` selects the torch family from `uv.lock`, builds the `uv export` command, the held constraints and the `uv pip install` command, and `install_packages` runs export then install for the real Pipeline only.
- [x] Unit tests cover the torch-family selection (fixture lock), the held constraints (local segment dropped, missing package skipped) and both commands for the real and the fake Pipeline.
- [x] A failing export or install still raises with uv's last lines.
- [x] `docs/agents/gpu-stack.md`, the `[tool.ty.analysis]` comment and the `server.build_components` docstring name the `gpu` extra.
- [x] Ruff, ty and pytest pass.
