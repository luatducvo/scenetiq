# 07: Cancel a running job

**What to build:** A Cancel button, with confirmation, on the Processing page and on running Scenes in the Scene list. It asks the running job to stop at its next checkpoint: a stage boundary, the next View in Masking, or the next Instance in Captioning. The page says the job is stopping until it has.

The Scene is then marked `cancelled`:

- its photos and partial results are deleted;
- its record stays, with its stage times and notes, until the user deletes it;
- the job slot is free for the next Capture.

The Colab progress line says `cancelled at <stage>`.

Spec: *Cancellation* user stories 52–60 and 40–42; *HTTP API* (`POST /jobs/{id}/cancel`, status `cancelled`); *Job reporting and cancellation*; *Frontend* (Scene list, Processing page); *Out of Scope* (no pause or resume, no interrupting a model call).

**Blocked by:** 01, 05, 06

**Status:** ready-for-agent

- [ ] Job status gains a terminal `cancelled`. A cancelled Scene can be deleted, and its Reconstruction endpoints answer 409.
- [ ] `POST /jobs/{id}/cancel` answers 202 for a queued or running job, including a repeated request while it stops. It answers 404 for an unknown job and 409 for a succeeded, failed or cancelled one.
- [ ] The job runner keeps a cancellation flag for the active job. `checkpoint()` and `stage()` raise the new `JobCancelledError` once it is set. The runner also checks the flag after the Pipeline returns and before saving.
- [ ] On cancellation the runner:
  - sets status `cancelled` and `finished_at`;
  - deletes the Scene's directory and keeps its record and events;
  - reports `cancelled` progress;
  - logs at INFO;
  - frees the slot.
- [ ] The real Pipeline's Masking stage checks before each View, and its Captioning stage before each Instance's generation, through an optional checkpoint callable.
- [ ] The fake Pipeline honours cancellation through the reporter, and its gate still holds a job.
- [ ] The Colab progress line renders `cancelled at <stage>`, with a unit test next to the existing progress bar tests.
- [ ] Seam A tests, with a job held by the fake Pipeline's gate:
  - cancel answers 202;
  - the job ends `cancelled`;
  - the Scene's directory is gone;
  - `/health` shows no running job;
  - a new Capture can start;
  - the cancelled Scene can be deleted and its points endpoint answers 409.

  Also: cancelling a succeeded job answers 409, and an unknown job 404.
- [ ] The Processing page shows Cancel with a confirmation while the job is queued or running, then "stopping" until the status is `cancelled`, then that it was cancelled. Its stage times and notes stay visible.
- [ ] The Scene list shows a `cancelled` status style, Cancel with a confirmation on running Scenes, and Delete on cancelled ones.
- [ ] AGENTS.md adds the constraint that cancellation is cooperative, through the reporter's checkpoints, never by killing the thread. The README describes Cancel.
- [ ] Ruff, ty and pytest are green, and the frontend passes lint, typecheck and build.
- [ ] Manual check against the local backend. The fake Pipeline finishes instantly without a gate, so either temporarily gate it or check the flow through the Seam A tests. The Cancel buttons should appear, confirm and update the status.
