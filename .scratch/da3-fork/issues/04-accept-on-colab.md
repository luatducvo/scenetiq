# 04: Accept the locked GPU stack and the DA3 fork on Colab

**What to build:** The owner confirms on a fresh Colab runtime that the new install keeps Colab's torch, that DA3 from the fork gives the same Reconstruction as the vendored copy, and that SAM2 works without its CUDA extension. See "Testing Decisions" in `.scratch/da3-fork/spec.md`.

**Blocked by:** 02: Install DA3 from the fork and delete the vendored copy, 03: Build SAM2 without its CUDA extension, against Colab's torch

**Status:** ready-for-human

- [ ] Install time on a fresh runtime is recorded before (current `main`) and after.
- [ ] `torch.__version__` and `torchvision.__version__` are the same before and after the install.
- [ ] `pytest -m gpu` passes.
- [ ] One Capture is reconstructed before and after; depth maps and cameras match, and any difference is recorded here.
- [ ] SAM2's Masks for the same Capture match before and after, and the warning about the missing `sam2._C`, if any, is recorded.
- [ ] Two installs in separate sessions give the same `pip freeze` for the GPU stack.
- [ ] The `unverified.md` row for DA3 is removed once all of the above hold.
