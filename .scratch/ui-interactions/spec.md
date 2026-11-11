# Spec: Upload preview, heatmap Matches, Names view, job timings and cancellation

Status: ready-for-agent

Vocabulary follows `GLOSSARY.md`. What users call an Instance "label" is its Caption's Name, so this spec says **Name** and calls the new viewer mode the **Names view** (GLOSSARY lists "label" under *Avoid* for Caption). Respect `docs/adr/`, in particular 0006 (one View per request), 0008 (RRF top-K, no threshold) and 0009 (spatial Queries). This work adds ADR 0010 (photos are shrunk and stripped in the browser). ADR 0008 is unchanged: `rrf_k` stays 60 and the ranking does not change; only the colour scale is stretched.

## Problem Statement

A user uploads phone photos, waits for the job, then explores the Scene with Queries. Today each of those steps is harder than it needs to be:

- **Upload.** The user cannot see which photos they picked before sending them. Phone photos are large (often 3–8 MB each) and carry metadata the Pipeline never reads: GPS position, camera maker notes, embedded preview pictures. All of it travels through the Cloudflare tunnel. A portrait photo stored as landscape plus an EXIF orientation flag reaches every stage sideways, because nothing applies the flag.
- **Queries.** A Query returns one Match by default. With rrf_k = 60, the scores of the top Matches differ only in the second decimal, so they say nothing visible. Each Match is drawn with a bounding box, a floating Name label and, for spatial Queries, extra Anchor boxes. Everything outside the Matches is dimmed. The result is cluttered and hides the Scene's own colours. The Match list shows raw scores the user cannot interpret.
- **Viewing.** The point size is fixed. It is too small for a sparse Scene and too large for a dense one.
- **Inspecting Instances.** The Instances view colours each Instance but never says what the Captioning stage called it. The user cannot tell at a glance which Instances got no Caption.
- **Jobs.** The Scene list shows neither how many Instances a Scene has nor how long it took to process. The Processing page shows only the current stage: no time spent per stage, and no summary of what happened. In particular, a Scene the Aligning stage could not level is reported only in the Colab log, although it makes vertical Relations unreliable. A job started by mistake (wrong photos, wrong View stride) cannot be stopped. It holds the single job slot and the GPU until it finishes, which can take tens of minutes.

## Solution

- **Upload preview.** After choosing photos, the user sees a grid of thumbnails before anything is sent. The browser prepares every photo:
  - it applies the EXIF orientation to the pixels;
  - it shrinks the photo so that its long side is at most 2048 px;
  - it re-encodes it as JPEG at quality 0.9, which drops all metadata.

  The grid shows the size before and after, for each photo and in total. The user can remove single photos, and a warning appears when the count is outside the backend's View limits. Only the prepared photos are uploaded.
- **Heatmap Matches.**
  - A Query returns 5 Matches by default, and a slider in the viewer sets K from 1 up to a configured maximum.
  - Each Match's points are coloured by a heatmap of its score, stretched over the scores of the Matches shown: the best shown Match gets the hottest colour, the lowest shown Match the coolest, and the others in proportion to their score. There is no colour legend. Every other point keeps its original colour, undimmed.
  - Boxes, floating labels and Anchor highlights are gone.
  - The Match list shows only Names, best first, each with a swatch of its heatmap colour.
  - The RRF constant stays 60, so the ranking is exactly as today. Its scores of the top Matches lie close together (often 0.93–1.0), which is why the colour scale is stretched over the Matches shown instead of fixed at 0–1.
- **Point size.** A slider scales the point size relative to the automatic size, and the browser remembers the choice.
- **Names view.**
  - The Instances toggle becomes a Names view: every Instance keeps its colour and shows its Caption's Name at its centroid.
  - An Instance without a Caption is drawn black and has no Name.
  - Points that belong to no Instance stay grey, as today.
- **Scene list.** Two new columns: the number of Instances and the processing time.
- **Processing page.** Each stage shows how long it took, and the current stage counts live. Below the stages, a short list of job notes appears as they happen:
  - how many Views were reconstructed;
  - that the Scene could not be aligned to gravity, so above, below, on, under, tallest and lowest may be wrong;
  - how many Instances were found and how many were dropped as unseen;
  - how many Instances got a Caption.
- **Cancel.** A Cancel button on the Processing page and in the Scene list stops the running job at its next checkpoint: a stage boundary, the next View in Masking, or the next Instance in Captioning. The Scene is then marked cancelled, its photos and partial results are deleted, its record stays until the user deletes it, and the job slot and GPU are free for the next Capture.

## User Stories

### Upload preview and photo preparation

1. As a user, I want to see thumbnails of the photos I selected before uploading, so that I can check I picked the right Capture.
2. As a user, I want to remove a single photo from the selection, so that I can drop a blurred or unrelated shot without re-selecting everything.
3. As a user, I want to see how many photos are selected and whether that count is within the backend's View limits, so that I do not start an upload the backend will refuse.
4. As a user, I want the upload button disabled while the count is outside the limits or photos are still being prepared, so that I cannot send an incomplete or invalid Capture.
5. As a user, I want the browser to shrink each photo so that its long side is at most 2048 px, so that the upload takes a fraction of the time on a slow connection.
6. As a user, I want every photo re-encoded without metadata, so that my GPS position and camera details never leave my machine.
7. As a user, I want a portrait photo turned upright according to its EXIF orientation before upload, so that every stage of the Pipeline sees it the right way up.
8. As a user, I want to see each photo's size before and after preparation, and the total for the Capture, so that I know how much I saved.
9. As a user, I want a progress indicator while photos are being prepared, so that I know the page has not frozen on a large selection.
10. As a user, I want a photo the browser cannot decode reported by name and left out of the selection, so that one bad file does not block the rest.
11. As a user, I want a photo already smaller than 2048 px kept at its size, so that preparation never enlarges a photo.
12. As a user, I want photos uploaded in the order shown in the grid, so that the View order and the View stride stay predictable.
13. As a Colab owner, I want the maximum side and the JPEG quality set in the notebook's `CONFIG`, so that I can trade upload size for detail without rebuilding the frontend.

### Queries and heatmap

14. As a user, I want a Query to return 5 Matches by default, so that I can see more than one candidate when the first is wrong.
15. As a user, I want a slider in the viewer that sets how many Matches a Query returns, so that I can widen or narrow the candidates.
16. As a user, I want the slider capped at a maximum the backend allows, so that I cannot ask for an unbounded number of Matches.
17. As a user, I want changing the slider to re-run the current Query, so that I see the new number of Matches without retyping.
18. As a user, I want each Match's points coloured by a heatmap of its score, stretched over the Matches shown, so that Matches with close scores still get clearly different colours.
19. As a user, I want no colour legend or score shown, so that the view stays uncluttered and I judge Matches by colour order and Name.
20. As a user, I want every point outside the Matches kept in its original colour, so that I can still recognise the Scene around the Matches.
21. As a user, I want no bounding boxes or floating labels drawn for Matches, so that the view stays uncluttered.
22. As a user, I want no Anchor highlighted, boxed or listed for a spatial Query, so that the only highlighted points are the Matches.
23. As a user, I want spatial Queries such as "the box on the shelf" to keep working, so that removing the Anchor display does not remove spatial reasoning.
24. As a user, I want the Match list to show only each Match's Name, best first, so that I can read it without interpreting numbers.
25. As a user, I want each Name in the list preceded by a swatch of that Match's heatmap colour, so that I can find it in the Scene.
26. As a user, I want a Match without a Caption listed as an unnamed Instance with its id, so that it is still identifiable.
27. As a user, I want the line that says which part of a Query was not taken into account to stay, so that I still know when "left of" was ignored or the Query could not be read.
28. As a user, I want the ranking of Matches unchanged by the heatmap, so that a colour change never changes which Instance comes first.

### Point size

29. As a user, I want a slider that makes points larger or smaller, so that both sparse and dense Scenes are readable.
30. As a user, I want the point size relative to the Scene's automatic size, so that the same setting suits Scenes of different extents.
31. As a user, I want the browser to remember my point size between Scenes and visits, so that I do not set it again each time.

### Names view

32. As a user, I want a Names view that shows every Instance's Name at its centroid, so that I can see what the Captioning stage called each part of the Scene.
33. As a user, I want each Instance in the Names view drawn in its own colour, so that I can see where one Instance ends and the next begins.
34. As a user, I want an Instance without a Caption drawn black and without a Name, so that I can see which Instances got no Caption.
35. As a user, I want points that belong to no Instance kept grey, so that they stay distinct from uncaptioned Instances.
36. As a user, I want the Names view unavailable while a Query result is shown, with a hint to clear the Query, so that the two colourings never mix.

### Scene list

37. As a user, I want each Scene in the list to show its number of Instances, so that I can spot a Scene that was split badly.
38. As a user, I want each finished Scene to show its processing time, so that I can estimate how long the next Capture will take.
39. As a user, I want the number of Instances and the processing time left blank for a Scene still processing, failed, cancelled or interrupted, so that no misleading value is shown.
40. As a user, I want a cancelled Scene shown with its own status, so that I can tell it from a failed one.
41. As a user, I want to delete a cancelled Scene, so that I can clear it from the list.
42. As a user, I want a Cancel button on a running Scene in the list, so that I can stop it without opening the Processing page.

### Processing page

43. As a user, I want each finished stage to show how long it took, so that I know where the time goes.
44. As a user, I want the current stage to show a live elapsed time, so that I can tell a slow stage from a stuck page.
45. As a user, I want the total time of the job, so that I can compare Captures.
46. As a user, I want job notes listed under the stages as they are recorded, so that I learn what happened without the Colab log.
47. As a user, I want a note saying how many Views were reconstructed, so that I can check the View stride did what I expected.
48. As a user, I want a note when the Scene could not be aligned to gravity, saying that vertical Relations may be wrong, so that I do not trust "above" or "on" on that Scene.
49. As a user, I want a note with how many Instances were found and how many were dropped as unseen, so that I can judge the segmentation.
50. As a user, I want a note with how many Instances got a Caption, so that I know how much of the Scene can be found by Name.
51. As a user, I want the stage times and notes still visible after the job ends, so that I can read them once it has finished, failed or been cancelled.

### Cancellation

52. As a user, I want a Cancel button on the Processing page while the job is queued or running, so that I can stop a job I started by mistake.
53. As a user, I want to confirm before cancelling, so that one misclick does not throw away a long job.
54. As a user, I want the page to say the job is stopping until it reaches its next checkpoint, so that I understand why it does not stop at once.
55. As a user, I want a cancelled job to stop at the next stage boundary, the next View in Masking or the next Instance in Captioning, so that it stops within minutes rather than at the end.
56. As a user, I want a cancelled Scene's photos and partial results deleted, so that they do not fill the Colab disk.
57. As a user, I want the job slot free as soon as the cancelled job stops, so that I can upload the next Capture.
58. As a Colab owner, I want the models a cancelled job loaded for its stage released as on a failure, so that the next job has the same GPU memory as after a failed one.
59. As a user, I want cancelling a job that has already finished refused with a clear message, so that a finished Scene is never deleted by a late click.
60. As a Colab owner, I want the notebook's progress line to say a job was cancelled and at which stage, so that the log tells a cancellation from a failure.

### Compatibility

61. As a Colab owner, I want a database from before this change to open and show its Scenes, so that a local storage root keeps working after an update.
62. As an API client, I want the existing fields of every response unchanged, so that older clients keep working.

## Implementation Decisions

### Configuration (`scenetiq.config`)

- `query_top_k` default goes from 1 to 5. It is the K used when a Query does not ask for one.
- New `query_max_top_k`, default 20: the largest K a Query may ask for. It must not be below `query_top_k`.
- `rrf_k` stays 60.
- New `upload_max_side`, default 2048, at least 1: the longest side, in pixels, the frontend shrinks photos to.
- New `upload_jpeg_quality`, default 0.9, in (0, 1]: the JPEG quality the frontend re-encodes photos at.
- All new settings are overridable through the notebook's `CONFIG`, like every other setting, and are documented in the configuration docs.

### HTTP API

- **New `GET /options`** (token required, like every route) returns what the frontend needs to configure itself: `min_views`, `max_views`, `upload_max_side`, `upload_jpeg_quality`, `query_top_k` and `query_max_top_k`.
- **`POST /scenes/{id}/query`** body gains an optional `top_k`.
  - When it is missing, `query_top_k` applies.
  - It must be from 1 to `query_max_top_k`; otherwise the answer is 422.
  - It applies to plain and spatial Queries alike, everywhere `query_top_k` applies today, including the shortlist of the wanted object of a spatial Query.
  - The response is unchanged. Anchors are still returned; the frontend simply ignores them.
- **`GET /jobs/{id}`** keeps its fields and adds:
  - `started_at` and `finished_at`: ISO 8601 UTC, null until the job starts or ends.
  - `stages`: one entry per stage that has started, in order, each with `stage`, `started_at` and `finished_at`. A stage finishes when the next one starts or the job ends; `finished_at` is null for the running stage.
  - `notes`: entries in the order recorded, each with `at`, `stage` (the stage that was running, or null) and `message`.
- **`GET /scenes`** keeps its fields and adds `started_at`, `finished_at` and `instance_count`. `instance_count` is null unless the job succeeded. The frontend computes the processing time from the two timestamps.
- **New `POST /jobs/{id}/cancel`** asks the running job to stop.
  - It answers 202 when the job is queued or running and the request was registered. Asking again while the job is stopping is also 202.
  - It answers 404 for an unknown job and 409 for a job that has already succeeded, failed or been cancelled.
  - It returns before the job stops. The client follows `GET /jobs/{id}` until the status is `cancelled`.
- **Job status gains `cancelled`.**
  - It is terminal, like `succeeded` and `failed`.
  - A cancelled Scene can be deleted, and its Reconstruction endpoints answer 409 like any Scene that is not ready.
  - `GET /health` reports `job_running` false once the cancelled job has stopped.
- `/health` is unchanged.

### Job reporting and cancellation (domain, job runner, Pipelines)

- **The `Pipeline` protocol changes from `run(capture, on_stage)` to `run(capture, reporter)`.** The reporter is a new protocol in the domain module with three operations:
  - `stage(stage)` reports that a stage starts. Before recording it, it checks for cancellation.
  - `checkpoint()` raises the new `JobCancelledError` if cancellation was requested, and otherwise does nothing.
  - `note(message)` records one job note, attributed to the stage that is running.
- **The bare stage callback type goes away.** The real Pipeline, the fake Pipeline and the GPU smoke test are updated.
- **The job runner owns the reporter.**
  - It keeps one cancellation flag for the job that holds the slot and exposes `cancel(job_id)`. That returns whether the job is the active one and still unfinished.
  - The runner also checks the flag once after the Pipeline returns and before the Reconstruction is saved, so a cancellation that arrives during the last stage still wins. Once saving has started, the job finishes normally.
- **On `JobCancelledError` the runner:**
  - marks the job `cancelled` and sets `finished_at`;
  - deletes the Scene's directory (Views and any partial Reconstruction), keeping its record;
  - reports progress with status `cancelled`;
  - frees the slot.

  It logs the cancellation at INFO, not as an exception.
- **The real Pipeline calls `checkpoint()`:**
  - at every stage boundary, through `stage()`;
  - before each View in the Masking stage's mask generation;
  - before each Instance's generation in the Captioning stage.

  The Masking and Captioning functions take an optional checkpoint callable, so their signatures stay usable without one.
- **Some work cannot be interrupted:** a single model call (one DA3 forward pass over every View, one SAM2 generation, one MV3DIS run, one Qwen3-VL generation). The page says so while the job is stopping.
- **GPU memory after a cancellation** follows the existing failure path: DA3 and SAM2 leave the GPU in their own cleanup when the exception passes through, and the next job frees anything left at its start. SigLIP2, Qwen3-Embedding and Qwen3-VL stay resident, as after any job.
- **The real Pipeline records these notes:**
  - after Reconstructing: the number of Views reconstructed;
  - after Aligning, only when the Views gave no consensus: that the Scene was left in DA3's frame, so above, below, on, under, tallest and lowest may be wrong. The Aligning stage already returns its inputs unchanged in that case;
  - after Embedding: the number of Instances kept and the number dropped because no View sees them well enough;
  - after Captioning: how many of those Instances got a Caption.
- **The fake Pipeline records a fixed note** after its first stage, and supports cancellation through the same reporter. Its existing gate keeps a job running so that a test can cancel it.
- **The Colab progress line** gets a `cancelled at <stage>` state alongside `failed at <stage>`.

### Storage

- The scenes table gains `started_at`, `finished_at` and `instance_count`.
  - `started_at` is set when the job starts running.
  - `finished_at` is set when it succeeds, fails or is cancelled.
  - `instance_count` is set from the Reconstruction's Instances when it is saved.
- A new job events table holds one row per event: job id, sequence number, timestamp, kind (`stage` or `note`), stage and message. Stage rows replace nothing: the scenes table still holds the current stage.
- **Migration.** On opening, the store adds the missing columns to an existing scenes table and creates the events table if needed, so a database from before this change opens with its Scenes intact, with null timestamps and counts.
- Deleting a Scene also deletes its events.
- A job interrupted by a restart is marked failed as today. Its `finished_at` stays null, so no processing time is shown for it.

### Frontend

- **Configuration.** The upload form and the viewer read `GET /options` once when they mount. If it fails, the upload form shows the error and does not allow uploading.
- **Photo preparation** is a pure module of the frontend. For each selected file it:
  - decodes the file with the EXIF orientation applied;
  - scales it so that its long side is at most `upload_max_side`, never enlarging it;
  - draws it on a canvas and exports a JPEG at `upload_jpeg_quality`;
  - makes a small thumbnail (about 160 px) for the grid.

  It returns the prepared file (renamed to `.jpg`), the thumbnail URL, the original and prepared byte sizes, and the pixel size.
  - It runs a few files at a time and reports progress.
  - It always re-encodes, even when the result is larger, so that the orientation is applied and the metadata dropped for every photo.
  - A file that cannot be decoded is reported by name and left out.
  - Thumbnail URLs are revoked when a photo is removed or the form unmounts.
- **Upload form.**
  - The thumbnail grid keeps the selection order and has a remove button on each photo.
  - Totals show the size before and after.
  - A warning appears while the count is outside `min_views`–`max_views`.
  - The upload button is disabled while preparing, while the count is out of range, or while there are no photos.
  - The existing upload, its progress and its retries are unchanged, but they now send the prepared files.
- **Viewer.**
  - Bounding boxes, floating labels and every Anchor colour, label and list entry are removed.
  - Match points get a heatmap colour of their score. The colour position of a Match is (score − lowest shown score) / (highest shown score − lowest shown score). When every shown Match has the same score, including K = 1, all get the hottest colour. Every other point keeps its original colour.
  - No colour legend is shown. The stretch is computed in the frontend only; the API's scores are unchanged.
  - The Match list shows a heatmap swatch and the Name, or "Instance <id>" without a Caption, best first, with no score.
  - The notice line about unsupported parts of the Query stays.
- **K slider.** It goes from 1 to `query_max_top_k` and starts at `query_top_k`. It is sent as `top_k` with each Query. Changing it re-runs the current Query when a result is shown.
- **Point size slider.** It is a multiplier on the automatic point size, roughly 0.25× to 4×. The canvas takes it as a prop, and the viewer remembers it in local storage, guarded for browsers where storage throws.
- **Names view.**
  - It replaces the Instances toggle and is disabled while a result is shown, as today.
  - Every Instance with a Name keeps its Instance colour and gets its Name at its centroid.
  - Every Instance without a Name is black.
  - Points of no Instance are grey.
- **Scene list.**
  - New columns show the Instance count and the processing time, formatted as minutes and seconds, or blank when missing.
  - `cancelled` gets its own status style.
  - Running Scenes get a Cancel button with a confirmation.
  - Cancelled Scenes get Delete.
- **Processing page.**
  - Each stage shows its duration from `stages`, and the current stage ticks every second.
  - The total time is shown.
  - Notes are listed in order.
  - A Cancel button with a confirmation shows while the job is queued or running. After a cancel request, the page shows that the job is stopping until the status becomes `cancelled`, then says it was cancelled.
  - Polling stops on any terminal status.

### Documents

- **New ADR 0010:** photos are oriented, shrunk to `upload_max_side` and re-encoded without metadata in the browser before upload.
  - Why: the Pipeline reads no metadata, and DA3 works at 504 px. SigLIP2 and Qwen3-VL crop from the uploaded photo, so 2048 px keeps small Instances legible. Shrinking cuts the upload several times over.
  - Consequence: an API client other than the frontend still sends raw photos, and the backend still applies no orientation.
- **AGENTS.md:**
  - The EXIF constraint changes: the frontend applies orientation, the backend does not.
  - New constraint: cancellation is cooperative, through the reporter's checkpoints, never by killing the thread.
- **README:** the usage section describes the upload preview, the heatmap, the K and point size sliders, the Names view, the stage times and notes, and Cancel. It no longer mentions boxes or Anchor colours.
- **Configuration docs and pipeline docs** get the new and changed defaults.

## Testing Decisions

- **What a good test looks like.** Tests go through the HTTP API and assert on what a client sees: status codes, response fields, the order of entries, and files existing or not under the storage root. They never assert on private store methods or SQL. Expected values come from hand-worked numbers, never from re-running the code's formula.
- **Seam A (`tests/api/`, FakePipeline, FakeTextEncoder, FakeCaptionEncoder)** covers almost everything:
  - **`GET /options`** returns the configured values, including overrides.
  - **`top_k` in a Query:**
    - it returns that many Matches, best first;
    - a missing `top_k` falls back to `query_top_k`;
    - 0 and values above `query_max_top_k` answer 422;
    - a spatial Query honours `top_k` for the wanted object's shortlist.

    Prior art: the existing tests that build Settings with an explicit `query_top_k`.
  - **Job details:**
    - after a successful job, `stages` lists every Stage in order;
    - each stage's `finished_at` equals the next stage's `started_at`, and the last equals the job's `finished_at`;
    - `notes` contains the fake Pipeline's note with its stage.

    Prior art: the upload and job tests.
  - **The Scene list** carries `instance_count` (3 for the fake Reconstruction) and both timestamps for a succeeded Scene. It carries nulls for a failed one.
  - **Cancellation:**
    - with the fake Pipeline's gate holding a job, `POST /jobs/{id}/cancel` answers 202, and the job ends `cancelled`;
    - the Scene's directory is gone;
    - `GET /health` shows no running job;
    - a new Capture can start;
    - the cancelled Scene can be deleted;
    - its Reconstruction endpoints answer 409;
    - cancelling a succeeded job answers 409, and an unknown job 404.

    Prior art: the tests that hold a job with the gate to check the busy slot.
  - **Migration:** a storage root holding a database with the old scenes schema and one Scene opens through the app, and `GET /scenes` lists that Scene with null timestamps and count. Prior art: the restart recovery tests, which also prepare a store before starting the app.
- **Unit (`tests/unit/`):** the Colab progress line for a cancelled job reads `cancelled at <stage>`. Prior art: the existing progress bar tests. The new Settings validation (the K maximum below K, the upload side, the JPEG quality out of range) goes with the existing config tests.
- **Seam B (GPU):** the smoke test is updated to the reporter protocol, recording stages and notes. It is not run here. The checkpoints inside Masking and Captioning sit inside model-loading code, so per the test seam rules they are covered only by a Colab run.
- **Frontend:** there is no frontend test seam, and none is added. The frontend must pass lint, typecheck and build. Each frontend ticket carries a manual checklist run against the local backend with the fake Pipeline.

## Out of Scope

- Pausing and resuming a job, or keeping the partial results of a cancelled one.
- Interrupting a single model call (a DA3 forward pass, an SAM2 generation, an MV3DIS run, a Qwen3-VL generation).
- Cancelling an upload in progress. The existing behaviour stays: an abandoned Capture is discarded when the next one starts.
- Server-side shrinking, re-encoding or orientation of photos sent by clients other than the frontend.
- HEIC input. The upload still accepts only JPG and PNG, although iOS Safari converts HEIC to JPEG when picking photos.
- Absolute similarity scores, or any threshold on Matches (ADR 0008).
- Removing spatial reasoning or the Anchor fields from the API. Only their display goes.
- Clicking points or Instances in the 3D view, camera fly-to, Query history, and showing Descriptions in the viewer.
- Changing `rrf_k` or the ranking, and showing raw cosine similarities as the heatmap: the image and Caption cosines are on different scales, and the heatmap shows the fused score that decides the order.
- Validating the new `query_top_k`, `upload_max_side` or `upload_jpeg_quality` defaults against ground truth.

## Further Notes

- **Why the scale is stretched.** With rrf_k = 60 and an Instance ranked r in both rankings, its score is 61 / (60 + r): 1.0, about 0.984, 0.968, 0.953 and 0.938 for ranks 1 to 5. On a fixed 0–1 scale these would look the same. Stretched over the five, they span the whole ramp. The cost: the colours show order and relative gaps among the Matches shown, not how good a Match is, and with no legend or score shown, nothing tells the user how close the scores really are. This is accepted as requested.
- **Spatial Query scores** are assignment scores in [0, 1] (ADR 0009), not RRF scores. The same stretched heatmap applies to them.
- **Points with no Instance stay grey in the Names view.** The request only asked for uncaptioned Instances to be black, and grey keeps the two distinguishable.
- **Black Instances in the Names view** are hard to see on the viewer's dark background. This is accepted as requested, and is easy to revisit.
