# 01: Show the Aligning stage in job progress

**What to build:** A user creating a Scene sees an `aligning` step in job progress, between `fusing` and `superpoints`, making seven stages. This holds on both the fake and the real backend. At this point the real Pipeline only reports the stage and rotates nothing; later tickets give it behaviour. Parent spec: `.scratch/gravity-alignment/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] The stage enumeration has `aligning` between `fusing` and `superpoints`.
- [x] The real Pipeline reports `aligning` in that position, and its output is unchanged by the stage.
- [x] The frontend's job-progress stage list shows `aligning` in that position.
- [x] A Seam A test covers the job-stage progression with `aligning` in order and seven stages in total.
- [x] Any test that fixes the stage list or count, including the GPU smoke test, is updated.
- [x] The stage list in the v1 spec includes Aligning.
- [x] Running the backend with the fake Pipeline and creating a Scene shows the new step in the frontend.
- [x] Ruff, ty, pytest and the frontend checks pass.
