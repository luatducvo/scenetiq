# 05: Stage timings and job notes

**What to build:**

- The Processing page shows how long each stage took, a live elapsed time for the current stage, and the job's total time.
- Below the stages, it lists the job notes as they are recorded:
  - how many Views were reconstructed;
  - that the Scene could not be aligned to gravity, so vertical Relations may be wrong;
  - how many Instances were kept and dropped;
  - how many got a Caption.
- Both stay visible after the job ends.
- A database from before this change still opens.

Spec: *Processing page* user stories 43–51 and 61–62; *HTTP API* (`GET /jobs/{id}`); *Job reporting* (notes); *Storage*; *Frontend* (Processing page).

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] The store records `started_at` when a job starts running and `finished_at` when it succeeds or fails. It adds a job events table for stage starts and notes, and deletes a Scene's events with the Scene.
- [ ] Opening a store whose scenes table has the old schema adds the missing columns and creates the events table, keeping its Scenes.
- [ ] A job interrupted by a restart keeps a null `finished_at`.
- [ ] The job runner stores each `stage()` and `note()` through the store. A note is attributed to the running stage.
- [ ] `GET /jobs/{id}` adds `started_at`, `finished_at`, `stages` and `notes`:
  - `stages` has one entry per started stage, with `stage`, `started_at` and `finished_at`. A stage finishes when the next starts or the job ends, and the running stage's `finished_at` is null;
  - `notes` has `at`, `stage` and `message`.

  All existing fields are unchanged.
- [ ] The fake Pipeline records a fixed note after its first stage.
- [ ] The real Pipeline records the four notes of the spec. The gravity note is recorded only when Aligning returned its inputs unchanged.
- [ ] Seam A tests:
  - after a successful job, `stages` lists every Stage in order;
  - each `finished_at` equals the next `started_at`, and the last equals the job's `finished_at`;
  - the fake note appears with its stage;
  - a failed job ends with a `finished_at`;
  - an old-schema database opens through the app and lists its Scene.
- [ ] The Processing page shows each stage's duration, a ticking time for the current stage, the total time, and the notes in order. It keeps showing them after a terminal status, and stops polling on one.
- [ ] AGENTS.md's "only the job log says a Scene was left unaligned" constraint is updated: the job notes now say it too. The README describes stage times and notes.
- [ ] Ruff, ty and pytest are green, and the frontend passes lint, typecheck and build.
- [ ] Manual check against the local fake backend: the Processing page shows durations for all eight stages and the fake note.
