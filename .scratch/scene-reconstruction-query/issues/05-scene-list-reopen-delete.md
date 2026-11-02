# 05: List, reopen and delete Scenes

**What to build:** The user sees a list of their Scenes with name, creation time and status. They can open any succeeded Scene in the viewer and delete Scenes they no longer need.

Spec: `.scratch/scene-reconstruction-query/spec.md` (HTTP API, Storage, Frontend)

**Blocked by:** 03 (point cloud viewer)

**Status:** ready-for-agent

- [ ] The list-Scenes endpoint returns `[{scene_id, name, created_at, status}]`, newest first.
- [ ] The delete-Scene endpoint removes the Scene's directory and metadata.
  - [ ] It refuses to delete the Scene of a job that is still running.
  - [ ] It returns 404 for an unknown Scene.
- [ ] The frontend opens on the Scene list. Each succeeded Scene can be opened, and each Scene can be deleted after a confirmation.
- [ ] Seam A tests cover:
  - [ ] listing
  - [ ] deleting: the artifacts are gone, the Scene is no longer listed, and its point cloud returns 404
  - [ ] refusing to delete during a running job
- [ ] The Definition of Done in AGENTS.md is met.
