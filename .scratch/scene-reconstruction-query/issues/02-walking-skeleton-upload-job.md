# 02: Walking skeleton: upload a Capture, run a job, see stage progress

**What to build:** In the local React app, the user enters the backend URL and the shared token, uploads a named Capture, and watches its job move through the pipeline stages until it succeeds or fails. The real models are not connected yet. The backend runs an injected fake Pipeline that reports stages and returns a fixed Reconstruction.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Pipeline module, Storage, HTTP API, Frontend)

**Blocked by:** 01 (tooling and conventions)

**Status:** ready-for-agent

- [ ] Pipeline
  - [ ] A `Pipeline` interface exists, `run(capture, on_stage) -> Reconstruction`, with the Capture and Reconstruction types described in the spec.
  - [ ] A fake implementation exists, and the app gets its Pipeline by injection.
- [ ] FastAPI app
  - [ ] Configuration covers the storage root, the token and the threshold.
  - [ ] Any request without the correct token gets 401.
  - [ ] CORS allows the localhost frontend origin.
- [ ] Create-Scene endpoint
  - [ ] Takes a name and images, and returns `{scene_id, job_id}`.
  - [ ] Rejects fewer than 10 or more than 300 images, and files that are not JPG/PNG, with a 4xx and the reason.
  - [ ] Returns a conflict error while another job is running.
- [ ] Jobs and job status
  - [ ] Jobs run one at a time in a background worker.
  - [ ] Job and Scene metadata persist in SQLite under the storage root.
  - [ ] The job-status endpoint reports the status (queued, running, succeeded, failed), the current stage, and the error message on failure.
  - [ ] On success, the Reconstruction artifacts are written to the Scene's directory.
- [ ] Frontend (Vite + React + TypeScript)
  - [ ] A settings panel for the backend URL and token, saved in browser storage.
  - [ ] An upload form that checks the 10–300 image count on the client.
  - [ ] A progress view that polls job status and shows the stage and any error.
  - [ ] ESLint and `tsc --noEmit` are configured. Their commands are in AGENTS.md and in the Definition of Done for frontend work.
- [ ] Seam A tests cover: token enforcement, upload validation, the conflict on a concurrent upload, status transitions with stage reporting, and a failed job when the fake Pipeline raises.
- [ ] The Definition of Done in AGENTS.md is met. AGENTS.md explains how to run the backend and the frontend locally.
