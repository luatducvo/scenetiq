# 04: Accept Gravity alignment on Colab

**What to build:** The owner confirms on a real GPU that new Scenes come out level, and records how large the correction really is. This is the first real run of GeoCalib in this system, so it is also where a sign error in the up direction or an install problem would first show. See "Testing Decisions" and "Further Notes" in `.scratch/gravity-alignment/spec.md`.

**Blocked by:** 03: Refine up with a RANSAC plane

**Status:** ready-for-human

- [ ] The notebook installs the stack on a fresh runtime, and Colab's CUDA torch is not replaced.
- [ ] `pytest -m gpu` passes on Colab, including the up-direction test and the check that only SigLIP2 stays on the GPU after a run.
- [ ] Two or three real Captures are compared in the viewer, each as a Scene from before the change and one from after it, using the same photos.
- [ ] The log line appears in the notebook for each job, and method, rotation angle and consensus fraction are recorded in this ticket under `## Comments`.
- [ ] The recorded angles confirm or refute the hypothesis that DA3's world frame is not level.
- [ ] At least one Capture contains portrait photos from a phone, and the result is recorded, as input to ticket 05.
- [ ] Peak VRAM of the Aligning stage is recorded.
- [ ] The new step shows in the frontend's job progress (left open from ticket 01).
- [ ] Bounding boxes of Matches are checked by eye against the before Scene.
- [ ] If the thresholds or the focal prior need changing, the new defaults and the reason are recorded, or a follow-up ticket is opened.
