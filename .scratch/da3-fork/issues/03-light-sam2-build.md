# 03: Build SAM2 without its CUDA extension, against Colab's torch

**What to build:** The Colab install builds SAM2 in seconds: no CUDA compile and no torch download into a build environment. See "SAM2 build" in `.scratch/da3-fork/spec.md`.

**Blocked by:** 01: Lock the GPU stack and install it on Colab from the lock

**Status:** ready-for-agent

- [x] The real Pipeline's install runs with `SAM2_BUILD_CUDA=0` and passes `--no-build-isolation-package sam-2` on the command line.
- [x] The unit tests of the install command and environment cover both.
- [x] `docs/agents/gpu-stack.md` records why: SAM2 has no wheel, its build requires torch, and SceneTIQ does not use `sam2._C`.
- [x] Ruff, ty and pytest pass.
