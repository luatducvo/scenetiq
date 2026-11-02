# 04: Recover cleanly after a backend restart

**What to build:** If the backend restarts while a job is running (for example, when Colab disconnects), the user later sees that job as failed because it was interrupted by the restart, not stuck as running. Scenes that had already finished stay available.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Storage)

**Blocked by:** 02 (walking skeleton)

**Status:** ready-for-agent

- [ ] At app startup, every job still marked queued or running in SQLite is marked failed, with the reason "interrupted by restart". Jobs are not resumed.
- [ ] Scenes that had succeeded on the same storage root are still served after the restart.
- [ ] After recovery, a new upload is accepted: no stale single-job lock is left behind.
- [ ] Seam A test, in order:
  - [ ] create the app and leave a job in the running state
  - [ ] create a new app on the same storage root
  - [ ] check the job reads as failed with the interrupted reason
  - [ ] check a new upload is accepted
- [ ] The Definition of Done in AGENTS.md is met.
