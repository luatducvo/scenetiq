# 08: Real Reconstruction: DA3 depth fused by TSDF into a metric point cloud

**What to build:** On Colab, uploading real photos now produces a real Reconstruction: the user sees an actual metric-scale point cloud of their Scene. There are no Instances yet, so every point has instance id -1.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Pipeline stages 1–2). ADRs 0001, 0002.

**Blocked by:** 03 (point cloud viewer), 07 (Colab named tunnel)

**Status:** ready-for-agent

- [ ] The real Pipeline runs two stages, each reported through `on_stage`:
  - [ ] **Reconstructing**: DA3 `DA3NESTED-GIANT-LARGE`, called through its Python API, gives the metric depth, confidence, extrinsics and intrinsics of every View.
  - [ ] **Fusing**: confidence-filtered TSDF fusion builds a mesh. The mesh vertices (about 1–2 cm voxel) become the Reconstruction's point cloud.
- [ ] The reconstruction model sits behind its own stage interface, so the weights can be swapped (ADR 0001).
- [ ] Each View's camera (camera-to-world 4×4 plus intrinsics) and the fused mesh are kept in the Scene's storage for later stages. The mesh is never served to the frontend.
- [ ] Pure, unit-tested adapters convert:
  - [ ] DA3 world-to-camera 3×4 to camera-to-world 4×4
  - [ ] depth to float metres
  - [ ] colour to the depth resolution
- [ ] The Colab configuration defaults to the real Pipeline. CPU test runs still use the fake.
- [ ] A `gpu`-marked smoke test runs the real stages on a small Capture and checks that the point cloud is non-empty and within plausible metric bounds.
- [ ] The Definition of Done in AGENTS.md is met. AGENTS.md lists the model dependencies and how to install them on Colab.
