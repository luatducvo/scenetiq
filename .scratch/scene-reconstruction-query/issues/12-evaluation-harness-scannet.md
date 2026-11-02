# 12: Evaluation harness on ScanNet200 and threshold calibration

**What to build:** A developer runs one command on Colab against 5–10 ScanNet200 scenes. It reports class-agnostic instance AP and recommends an F1-optimal Query threshold, which becomes the configured default. Re-running the command is the end-to-end regression test (Seam B).

Spec: `.scratch/scene-reconstruction-query/spec.md` (Evaluation harness, Testing Decisions)

**Blocked by:** 10 (superpoints), 11 (real embedding)

**Status:** ready-for-agent

- [ ] For each ScanNet scene, the harness:
  - [ ] samples RGB frames evenly down to at most 300 Views, discarding the ScanNet depth and poses
  - [ ] runs the real `Pipeline.run`
  - [ ] aligns the Reconstruction to the ground truth with a similarity transform, estimated between the DA3 cameras and the ScanNet poses of the sampled frames
  - [ ] transfers the predicted Instances to the ground-truth points by nearest neighbour
- [ ] It computes class-agnostic AP, AP50 and AP25.
- [ ] It runs one Query per ground-truth class name and sweeps the threshold to find the F1-optimal value across all scenes.
- [ ] It writes a report to a chosen output location, with per-scene and aggregate metrics and the recommended threshold.
- [ ] The recommended threshold becomes the configured default, and AGENTS.md records how it was produced.
- [ ] The regression check compares AP with the last recorded run and fails when the drop exceeds a documented tolerance.
- [ ] The pure parts (frame sampling, similarity alignment, AP computation, threshold sweep) have CPU unit tests on synthetic data.
- [ ] ScanNet data is never committed or redistributed. AGENTS.md says where to put it, locally or on Drive.
- [ ] The Definition of Done in AGENTS.md is met.
