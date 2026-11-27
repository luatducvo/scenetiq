# Upload a Capture one View per request

The backend is reached through Cloudflare (ADR 0004), which refuses large request bodies: a 105 MB POST to our hostname got a 413 at the edge, before reaching the backend. A Capture sent as one multipart request therefore could not exceed roughly 70 photos of 1.5 MB, far below the Capture sizes we accept.

So a Capture takes three steps: `POST /captures` with its View count, `POST /captures/{id}/views` once per View, in parallel, then `POST /scenes` to commit it and start the job.

## Consequences

- A failed View is retried alone, and an invalid file or a busy backend is reported before the rest of the Capture is sent.
- Sending the same View index again replaces it, so retries are safe.
- Every View uses the same URL with the index in the form, because browsers cache a CORS preflight per URL. An index in the path would add one round trip per View.
- Only one Capture is staged at a time, under `<storage_root>/captures/`. Starting a new one discards an unfinished one, so abandoned Captures need no other cleanup. A second browser tab starting a Capture makes the first one fail with a 404.
- Parallel requests gained about 3x on HTTP/1.1 uplink tests to Cloudflare, but browsers may multiplex them over one HTTP/2 connection, where that gain was not measured. The upload form shows live MB/s for checking this on a real run.
