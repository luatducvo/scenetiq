# 06: Query a Scene and see Matches highlighted

**What to build:** In the Scene viewer, the user types a Query and sees each Match highlighted on the point cloud with a 3D bounding box, plus a list of Matches ranked by score. If no Instance clears the threshold, the closest Instance is shown, clearly flagged as a Low-confidence Match. The Instance embeddings and the text encoder are still fakes in this ticket.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Query module, HTTP API, Frontend)

**Blocked by:** 03 (point cloud viewer)

**Status:** ready-for-agent

- [ ] A text-encoder interface is injected into the app alongside the Pipeline. A fake implementation maps known strings to known vectors.
- [ ] The Query module:
  - [ ] scores Instances by cosine similarity
  - [ ] returns the Matches at or above the configured threshold, sorted by descending score
  - [ ] when there are none, returns exactly one Low-confidence Match, flagged as such
- [ ] Endpoints:
  - [ ] Per-point `instance_id`: compact binary, in the same order as the PLY, -1 for points that belong to no Instance.
  - [ ] Instances: id, centroid, bbox_min and bbox_max, in metres.
  - [ ] Query: returns `{matches: [{instance_id, score}], low_confidence}`.
- [ ] Viewer:
  - [ ] a Query box
  - [ ] the points of each Match highlighted, using the instance id array
  - [ ] a bounding box for each Match
  - [ ] a ranked list of Matches with their scores, with the low-confidence flag visible
- [ ] Tests:
  - [ ] unit tests for computing centroids and bounding boxes
  - [ ] Seam A tests for ranking, the threshold boundary, the Low-confidence fallback, the threshold coming from config, the instance id array matching the PLY order, and the Instance metadata
- [ ] The Definition of Done in AGENTS.md is met.
