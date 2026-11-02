# 03: View the Reconstruction as a point cloud

**What to build:** When a job succeeds, the user opens the Scene and sees its Reconstruction as a coloured point cloud in the browser. They can orbit, pan and zoom it.

Spec: `.scratch/scene-reconstruction-query/spec.md` (HTTP API, Frontend)

**Blocked by:** 02 (walking skeleton)

**Status:** ready-for-agent

- [ ] The get-point-cloud endpoint returns binary PLY (xyz in metres, rgb).
- [ ] Requesting an unknown Scene, or a Scene whose job has not succeeded, returns a 4xx with a clear reason.
- [ ] The frontend Scene viewer (react-three-fiber / three.js) renders the PLY with orbit, pan and zoom controls.
- [ ] The viewer opens automatically when a job succeeds.
- [ ] Seam A tests check that the PLY for a fake Reconstruction has the expected point count, positions and colours, and they cover the error cases.
- [ ] The Definition of Done in AGENTS.md is met.
