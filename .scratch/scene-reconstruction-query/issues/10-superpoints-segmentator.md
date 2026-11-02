# 10: Superpoints from the ScanNet Segmentator on the fused mesh

**What to build:** Instances get cleaner and Merging gets faster. MV3DIS now groups points by superpoints computed on the TSDF mesh, as in its original pipeline, instead of treating every point on its own.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Pipeline stage 3). ADR 0002.

**Blocked by:** 09 (real masking and merging)

**Status:** ready-for-agent

- [ ] The ScanNet Segmentator runs as the **Superpoints** stage on the fused mesh. It is either vendored, with provenance and licence recorded, or installed as a pinned dependency if a package exists.
- [ ] Superpoint ids are aligned with the Reconstruction's point order and fed to MV3DIS through its custom-superpoint path.
- [ ] `point_level=True` stays available as a configuration fallback for debugging.
- [ ] A `gpu`-marked smoke test checks that superpoints cover every point and that Merging still yields Instances.
- [ ] The Definition of Done in AGENTS.md is met.
