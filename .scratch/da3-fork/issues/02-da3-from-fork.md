# 02: Install DA3 from the fork and delete the vendored copy

**What to build:** DA3 comes from `luatducvo/Depth-Anything-3` at a pinned commit through the `gpu` extra, and `src/scenetiq/vendor/depth_anything_3/` is gone. See "DA3 from the fork" and "Documentation" in `.scratch/da3-fork/spec.md`.

**Blocked by:** 01: Lock the GPU stack and install it on Colab from the lock

**Status:** ready-for-agent

- [x] The extra declares the fork at a tag, or at commit `879fcab51a72518a9249989e6dd352a9aac32f7d` if the owner has not tagged it, and no longer lists DA3's own dependencies.
- [x] `requires-python` is `>=3.12,<3.14` and `uv lock` succeeds.
- [x] `RealReconstructor` imports `depth_anything_3.api`, still inside `_load`.
- [x] `src/scenetiq/vendor/depth_anything_3/` is deleted, and nothing in the repo refers to `scenetiq.vendor.depth_anything_3`.
- [x] ty allows `depth_anything_3.**`.
- [x] ADR 0013 supersedes ADR 0005, and ADR 0005 carries a superseded note.
- [x] `AGENTS.md`, `docs/agents/gpu-stack.md` and `docs/agents/unverified.md` are updated as the spec lists.
- [x] Ruff, ty and pytest pass.
