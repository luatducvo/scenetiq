# Spec: Scene reconstruction and open-vocabulary query (v1)

Status: ready-for-agent

Vocabulary follows `GLOSSARY.md`. Respect ADRs 0001–0004 in `docs/adr/`.

## Problem Statement

A user has a set of ordinary 2D photos of a room or other static space. They want to explore it in 3D and find objects in it by describing them in free text ("red chair", "the potted plant"). Today they would need depth sensors or known camera poses, and they would be limited to a fixed list of object classes. Free text does not work. The user has no GPU of their own, but they can use Google Colab with an 80GB GPU.

## Solution

The user opens a local web app and uploads a Capture of 10–300 photos of a Scene. The photos need no camera poses or intrinsics. The backend runs on Colab and reconstructs the Scene: it builds a metric-scale point cloud and splits it into class-agnostic Instances, each with a semantic embedding. The web app shows the job's progress stage by stage, then displays the point cloud. The user types a Query and sees the matching Instances highlighted with 3D bounding boxes. If nothing clears the relevance threshold, the app shows the single closest Instance flagged as a Low-confidence Match. Processed Scenes persist across Colab restarts and can be reopened or deleted.

## User Stories

1. As a user, I want to upload a set of photos of a static Scene, so that I can get a 3D Reconstruction without special hardware.
2. As a user, I want to upload photos without camera poses or intrinsics, so that I can use any ordinary camera or phone.
3. As a user, I want to name a Scene when I upload it, so that I can recognise it later in the list.
4. As a user, I want an upload with fewer than 10 photos rejected immediately with a clear message, so that I don't wait for a job that cannot succeed.
5. As a user, I want an upload with more than 300 photos rejected immediately with a clear message, so that I don't trigger a job that would run out of GPU memory.
6. As a user, I want files that are not JPG/PNG images rejected with a clear message, so that I know which file is the problem.
7. As a user, I want to receive a job identifier right after uploading, so that I can track processing without keeping the request open.
8. As a user, I want to see which stage the job is in (e.g. reconstructing, segmenting, embedding), so that I know it is progressing.
9. As a user, I want to see when a job has failed and why, so that I can retry or change my photos.
10. As a user, I want only one job to run at a time, and a new upload to be refused or queued clearly while one is running, so that the GPU is not overloaded.
11. As a user, I want to see the Reconstruction as a coloured point cloud in my browser, so that I can explore the Scene in 3D.
12. As a user, I want to orbit, pan and zoom the point cloud, so that I can inspect it from any angle.
13. As a user, I want the Reconstruction to be in real-world metric units, so that positions and sizes are meaningful.
14. As a user, I want to type a free-text Query against an open Scene, so that I can find objects without a fixed list of classes.
15. As a user, I want to find objects that are not in any standard dataset's class list, so that the search is truly open-vocabulary.
16. As a user, I want the points of every Matched Instance highlighted, so that I can see exactly which object was found.
17. As a user, I want a 3D bounding box drawn around each Match, so that I can locate it at a glance.
18. As a user, I want Matches ranked by relevance score, with the score shown, so that I can judge how confident the system is.
19. As a user, I want to see when a Query has no confident Match, so that I am not misled.
20. As a user, I want the closest Instance shown as a clearly flagged Low-confidence Match when nothing clears the threshold, so that I still get a best lead.
21. As a user, I want to run many Queries on the same Scene without reprocessing, so that each Query answers quickly.
22. As a user, I want to see a list of my processed Scenes, so that I can go back to earlier work.
23. As a user, I want to reopen a processed Scene from the list, so that I can query it again.
24. As a user, I want to delete a Scene, so that I can free storage and clean up the list.
25. As a user, I want my processed Scenes to survive a Colab restart, so that I don't lose hours of GPU work.
26. As a user, I want a job that was interrupted by a Colab restart marked as failed, not left stuck as running, so that I know to re-upload.
27. As a user, I want the backend reachable at a stable URL across Colab restarts, so that I don't have to reconfigure the frontend every time.
28. As a user, I want to set the backend URL and access token in the frontend, so that I can point it at my Colab instance.
29. As the owner, I want every API request to need a shared token, so that strangers who find the tunnel URL cannot use my GPU or see my Scenes.
30. As the owner, I want requests with a missing or wrong token rejected, so that the backend is never open by accident.
31. As the owner, I want to start the whole backend from one Colab notebook (mount Drive, install, start the server, open the tunnel), so that bringing the system up is a single action.
32. As a developer, I want each Instance to keep its centroid and 3D bounding box in metres, so that spatial reasoning can be added later without reprocessing.
33. As a developer, I want the relevance threshold to be a configuration value, so that it can be re-tuned without code changes.
34. As a developer, I want an evaluation harness that runs the real pipeline on 5–10 ScanNet200 scenes using RGB only, so that I can measure instance AP and tune the threshold.
35. As a developer, I want the harness to sample ScanNet frames evenly down to at most 300 Views, so that evaluation respects the same Capture limits as users.
36. As a developer, I want the harness to choose the F1-optimal threshold, using ground-truth class names as Queries, so that the threshold is grounded in data.
37. As a developer, I want the harness to act as an end-to-end regression test, so that a change to any stage that hurts quality is caught.
38. As a developer, I want the pipeline to sit behind one interface that the API depends on, so that the API can be tested on a CPU with a fake pipeline.
39. As a developer, I want the reconstruction model swappable behind its stage interface, so that switching to Apache-2.0 DA3 weights stays possible (ADR 0001).
40. As a developer, I want MV3DIS and the ScanNet Segmentator vendored with their source and licence recorded and the upstream commit pinned, so that I can trace and diff against upstream.

## Implementation Decisions

**Runtime and packaging**
- The whole project targets Python 3.12 (`.python-version` 3.12, `requires-python >=3.12`). DA3 needs ≤3.13, and Colab runs 3.12.
- Every ticket must meet the Definition of Done in AGENTS.md: ruff lint and format pass, ty passes, docstrings (Google style) on all public first-party code, tests at the right seam in the pytest test tree, and AGENTS.md updated when conventions change.
- It is a monorepo. The Python package holds the pipeline library and the FastAPI app. MV3DIS and the ScanNet Segmentator are vendored inside the package. A separate React frontend (Vite + react-three-fiber/three.js) runs on localhost. A Colab notebook only bootstraps the backend; all logic lives in the package.
- DA3 is a pip dependency, called through its Python API (`DepthAnything3.from_pretrained(...).inference(...)`).

**Pipeline module (the deep module)**
- One interface: `Pipeline.run(capture, on_stage) -> Reconstruction`. It takes a Capture (ordered list of View images) and a stage-progress callback. It returns a Reconstruction.
- A Reconstruction contains:
  - point cloud: xyz in metres, rgb
  - per-point `instance_id` (-1 for points that belong to no Instance)
  - the camera of every View
  - for every Instance: an id, its point indices, a centroid, an axis-aligned 3D bounding box (metres), and an L2-normalised embedding
- The stages run in this order, and stage names are reported through `on_stage`:
  1. **Reconstructing.** DA3 `DA3NESTED-GIANT-LARGE` gives metric depth, confidence, extrinsics and intrinsics for every View (ADR 0001).
  2. **Fusing.** TSDF fusion of the DA3 depth (with confidence filtering) produces a mesh. The mesh vertices are the Reconstruction's point cloud, about 1–2 cm voxel (ADR 0002).
  3. **Superpoints.** The vendored ScanNet Segmentator runs on the fused mesh.
  4. **Masking.** SAM2 automatic mask generation, with no prompts, runs on every View (ADR 0003).
  5. **Merging.** The vendored MV3DIS, refactored to take in-memory inputs (points, superpoints, depth, poses, intrinsics, Masks), uses 3D-guided matching to produce class-agnostic Instances. Its depth-consistency visibility check runs against depth rendered from or consistent with the fused geometry.
  6. **Embedding.** SigLIP2 (giant variant, `google/siglip2-giant-opt-patch16-384`, the largest available; chosen over "large" since 80GB VRAM allows it) encodes multi-view crops of each Instance from the Views where it is visible. These are aggregated into one embedding per Instance.
- Coordinate and unit conversions sit in small pure adapter functions:
  - DA3 world-to-camera 3×4 → camera-to-world 4×4
  - float metres for depth
  - colour resized to the depth resolution
- The vendored MV3DIS is refactored so it no longer reads the ScanNet on-disk layout (millimetre PNG depth, `pose/*.txt`, `*_vh_clean_2.ply`). Its algorithmic behaviour stays the same.

**Query module**
- `query(reconstruction, text, threshold) -> QueryResult`. It encodes the text with the SigLIP2 text tower, then scores every Instance by cosine similarity.
- QueryResult holds either:
  - a list of Matches (instance id plus score) at or above the threshold, sorted by descending score, or
  - when there are none, exactly one Low-confidence Match (the top-scoring Instance), flagged as such.
- Queries keep no state and no history is stored.
- The threshold comes from configuration. Its default is the value the evaluation harness produces.

**Storage**
- Everything persists under a configurable root, which is a Google Drive folder on Colab. Each Scene has its own directory holding its Reconstruction artifacts:
  - the point cloud as binary PLY
  - the per-point instance ids
  - Instance metadata and embeddings
- Job and Scene metadata (id, name, created time, status, current stage, error message) live in a SQLite database under the same root.
- At startup, any job left in a running state is marked `failed` with an "interrupted by restart" reason. Jobs are not resumed.
- Deleting a Scene removes its directory and its metadata row.

**HTTP API (FastAPI)**
- Every endpoint needs a shared token in a request header. The token is set by configuration (a Colab secret). A missing or wrong token gets 401.
- Endpoints (names are indicative):
  - create Scene: multipart upload of name and images → `{scene_id, job_id}`. Validates 10–300 images and JPG/PNG type, and returns 4xx with a reason otherwise. Returns a conflict error while another job is running.
  - get job status → `{status: queued|running|succeeded|failed, stage, error}`.
  - list Scenes → `[{scene_id, name, created_at, status}]`.
  - get Scene point cloud → binary PLY (xyz + rgb).
  - get Scene instance ids → compact binary array of per-point `instance_id`, in the same order as the PLY.
  - get Scene Instances → `[{instance_id, centroid, bbox_min, bbox_max}]`.
  - query Scene → `{matches: [{instance_id, score}], low_confidence: bool}`.
  - delete Scene.
- The Pipeline is injected into the app, which never imports model code directly. Jobs run one at a time in a background worker inside the server process.
- CORS allows the localhost frontend origin.

**Frontend**
- A settings panel takes the backend URL and token, stored in browser storage.
- Screens:
  - Scene list (open, delete)
  - upload form (name and images, with client-side 10–300 count validation)
  - job progress view that polls job status and shows the stage
  - Scene viewer
- The Scene viewer:
  - renders the PLY point cloud and lets the user orbit, pan and zoom
  - has a Query box
  - highlights the points of the Matched Instances (using the instance id array) and draws their bounding boxes
  - shows a ranked list of Matches with scores, and a visible flag on a Low-confidence Match
- The mesh is never sent to the frontend.

**Deployment**
- The Colab notebook:
  1. mounts Drive
  2. installs the package and its model dependencies
  3. reads the token from Colab secrets
  4. starts uvicorn
  5. starts a Cloudflare **named** tunnel, so the URL stays stable (ADR 0004)

**Evaluation harness**
- A CLI or module entry point takes a list of ScanNet200 scene ids and their local data paths.
- For each scene it:
  1. samples RGB frames evenly to at most 300, discarding the ScanNet depth and poses
  2. runs `Pipeline.run`
  3. transfers predicted Instances to ScanNet ground-truth points by nearest neighbour, because the Reconstruction is in its own frame and scale; it aligns first using the ScanNet poses of the sampled frames against the DA3 cameras, with a similarity transform
  4. computes class-agnostic instance AP / AP50 / AP25
  5. runs a Query for each ground-truth class name
- Across all scenes it sweeps the threshold to find the F1-optimal value.
- It writes a report and the recommended threshold.

## Testing Decisions

- Good tests check only external behaviour through a seam: requests in, responses and persisted effects out. They never touch internal stage functions, private helpers or intermediate files.
- **Seam A: HTTP API with a fake Pipeline (CPU, runs anywhere).**
  - The FastAPI app runs against a temporary storage root, with a fake Pipeline that returns a fixed, hand-built Reconstruction. That Reconstruction has a few Instances with known embeddings, and a fake text encoder maps known Query strings to known vectors.
  - It covers:
    - token enforcement
    - upload validation (fewer than 10 or more than 300 images, wrong file types)
    - single-job conflict
    - job status transitions and stage reporting
    - failed job with an error message, when the fake raises
    - Scene list, delete, and point cloud / instance id / Instance retrieval, with the PLY and instance id array kept consistent
    - Query thresholding, ranking and the Low-confidence Match fallback
    - restart recovery: a job left in a running state is marked failed when the app starts on the same storage root
  - This is the main test suite and must pass on CI without a GPU.
- **Seam B: Real Pipeline through the evaluation harness (GPU, Colab only).**
  - This is the end-to-end regression for all pipeline stages. Stages have no separate unit tests.
  - It records AP and the tuned threshold. A drop in AP against the last recorded run counts as a regression.
- The pure coordinate and unit adapters (pose convention, depth units, bounding box and centroid computation) may have small unit tests, because they are easy to get silently wrong.
- There is no prior art in the codebase: the repo is a fresh skeleton. Use pytest with FastAPI's TestClient (httpx).

## Out of Scope

- Spatial reasoning Queries (relations between Instances). v1 only stores the centroid and bounding box needed later.
- Video upload, frame extraction, and Captures with known poses or intrinsics.
- More than 300 Views per Capture, and DA3-Streaming or chunked reconstruction.
- More than one Capture per Scene, adding photos to an existing Scene, and renaming Scenes.
- Resuming interrupted jobs.
- Multiple users, accounts and per-user permissions.
- Sending or viewing the mesh in the frontend.
- Query history.
- Commercial use (ADR 0001: the DA3 Nested weights are CC BY-NC).
- Class labels on Instances.

## Further Notes

- The default 2D Mask source for MV3DIS (Grounded-SAM2 prompted with ScanNet200 classes) is deliberately not used (ADR 0003). Expect many more Masks. The merging stage has to cope with over-segmentation, and runtime per Capture will go up.
- MV3DIS has a `--points_clipfeature_path` argument, but its loader is unimplemented upstream. Do not rely on it. The Embedding stage is new code.
- MV3DIS's visibility check uses a relative 5% depth tolerance, so depth and points must share one scale. That is one reason the point cloud comes from the TSDF fusion of the same DA3 depth.
- ScanNet's own data licence applies to the evaluation scenes. They are used for evaluation only and are not redistributed.
