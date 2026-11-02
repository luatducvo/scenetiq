# 09: Real Instances: SAM2 Masks merged by MV3DIS

**What to build:** On Colab, a processed Scene now contains real class-agnostic Instances, and the user sees them in the viewer. The instance id array and the Instance list are real, with centroids and bounding boxes. Embeddings are still fake.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Pipeline stages 4–5). ADRs 0002, 0003.

**Blocked by:** 08 (real Reconstruction)

**Status:** ready-for-agent

- [ ] The **Masking** stage runs SAM2 automatic mask generation, with no prompts, on every View (ADR 0003).
- [ ] MV3DIS is vendored into the location reserved in ticket 01, with its upstream URL, pinned commit and MIT licence recorded.
- [ ] MV3DIS is refactored to take in-memory inputs (points, depth, poses, intrinsics, Masks) instead of the ScanNet on-disk layout. Its algorithmic behaviour does not change.
- [ ] The **Merging** stage runs MV3DIS with `point_level=True` (the fallback from ADR 0002; superpoints arrive in ticket 10). It produces Instances with point indices, a centroid and a bounding box.
- [ ] The first-party wrapper around the vendored MV3DIS is linted, type-checked and documented. Only the vendored directory is excluded from checks.
- [ ] A `gpu`-marked smoke test checks that a small real Capture yields at least one Instance, with consistent instance ids, centroids and bounding boxes.
- [ ] The Definition of Done in AGENTS.md is met. AGENTS.md documents the vendoring convention: location, provenance record, and exclusion from checks.
