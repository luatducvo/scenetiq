# Backend on a GPU runtime behind a Cloudflare named tunnel, one View per upload request

The Pipeline needs a large GPU, so the FastAPI backend runs on a GPU runtime, started by the notebook in `notebooks/`. The React frontend runs on the developer's machine and reaches it through a Cloudflare named tunnel, whose hostname stays the same across runtime restarts. Every request carries a shared token (`SCENETIQ_TOKEN`); there are no accounts.

Cloudflare refuses large request bodies (a 105 MB POST got a 413 at the edge), so a Capture is uploaded in three steps: `POST /captures` with its View count, `POST /captures/{id}/views` once per View, in parallel, then `POST /scenes` to commit it and start the job.

## Considered Options

- Mounted network storage for Scenes and logs: survives the runtime, but slows uploads and pipeline I/O.
- One multipart request per Capture: capped at roughly 70 phone photos by the edge limit.

## Consequences

- Scenes and job state (SQLite) live on the runtime's local disk and are lost with the runtime. Logs reach only the notebook's output. After a server restart on the same runtime, finished Scenes reload and in-flight jobs are marked failed.
- One job runs at a time (`JobBusyError`), and one Capture is staged at a time: starting a new one discards an unfinished one.
- A failed View is retried alone, and sending the same View index again replaces it. The index travels in the form, not the path, so the browser caches one CORS preflight.
