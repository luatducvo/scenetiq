# 01: Prefactor: the Pipeline reports through a job reporter

**What to build:** Replace the Pipeline's bare stage callback with a reporter, so that later tickets can record job notes and cancel a job without changing the Pipeline interface again. Nothing changes for the user: jobs report the same stages, in the same order, and end the same way.

The reporter is a new protocol in the domain module, with three operations (spec: *Job reporting and cancellation*):

- `stage(stage)` reports that a stage starts.
- `checkpoint()` does nothing yet. Ticket 07 makes it raise when cancellation was requested.
- `note(message)` accepts a job note. For now the job runner only logs it. Ticket 05 stores it.

`Pipeline.run(capture, on_stage)` becomes `run(capture, reporter)`. The job runner builds the reporter. The real Pipeline, the fake Pipeline and the GPU smoke test use it.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] The domain module defines the reporter protocol with `stage`, `checkpoint` and `note`, each with a Google-style docstring. The bare stage callback type is gone.
- [x] The real Pipeline and the fake Pipeline take a reporter and call `stage()` wherever they called the callback before.
- [x] The job runner implements the reporter: `stage()` updates the job's stage and the Colab progress line as before, `checkpoint()` does nothing, and `note()` logs at INFO.
- [x] The GPU smoke test records stages through a small reporter. It is not run here, but it type-checks.
- [x] All existing Seam A tests pass unchanged. Ruff, ty and pytest are green.
