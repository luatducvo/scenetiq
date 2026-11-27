# 07: Accept Captioning on Colab

**What to build:** The owner confirms on a real GPU that Captions are sensible and that Caption-aware ranking finds objects that image similarity alone missed. This is the first real run of both new models in this system. See "Testing Decisions" (Acceptance) and "Further Notes" in `.scratch/instance-captioning/spec.md`.

**Blocked by:** 06: Caption Instances with Qwen3-VL

**Status:** ready-for-human

- [ ] The notebook installs the stack on a fresh runtime, and Colab's CUDA torch is not replaced.
- [ ] `pytest -m gpu` passes on Colab.
- [ ] Two or three real Captures are compared, each as a Scene from before the change and one from after it, with synonym, material, colour and use Queries; results are recorded under `## Comments`.
- [ ] The captioning log line appears in the notebook, and the share of Instances without a Caption is recorded.
- [ ] Peak VRAM of the Captioning stage and the job's added runtime are recorded.
- [ ] Names in the Match list and on the bounding boxes are checked by eye.
- [ ] If K, the RRF constant, Views per Instance or the context fraction need changing, the new defaults and the reason are recorded, or a follow-up ticket is opened.
