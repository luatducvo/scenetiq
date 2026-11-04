# Spec: Gravity alignment of the Reconstruction

Status: ready-for-agent

Vocabulary follows `GLOSSARY.md`. Respect the ADRs in `docs/adr/`, in particular 0002 (mesh vertices are the point cloud) and 0005 (why DA3 is vendored).

## Problem Statement

A user opens a finished Scene and the point cloud looks tilted: the floor is not level and the Scene sits at an angle in the viewer. Nothing in the system knows which way is physically up. TSDF fusion integrates depth in whatever world frame the Reconstructing stage's cameras use, and that frame is not tied to gravity. The viewer nevertheless assumes a fixed up direction.

The tilt also damages data, not just the picture. An Instance's bounding box is axis-aligned in the world frame, so in a tilted Scene every box is larger than the object it surrounds, and any later spatial reasoning would inherit a wrong notion of "above" and "below".

## Solution

Every new Reconstruction is Gravity-aligned. The Pipeline gains an Aligning stage after Fusing. It estimates the Scene's up direction from the photos with GeoCalib, refines it from the fused geometry with RANSAC plane detection when a suitable horizontal plane exists, and rotates the points and cameras so that physical up matches the conventional up direction.

The user sees a level Scene and tighter bounding boxes, and one extra step in job progress. When the evidence is too weak to trust, the Scene is left as it was and the job still succeeds, with the reason in the backend log.

## User Stories

1. As a user, I want the floor of my Scene to appear level in the viewer, so that the Scene looks like the place I photographed.
2. As a user, I want the viewer to orbit around a sensible vertical axis, so that I can explore the Scene without fighting a tilted horizon.
3. As a user, I want Gravity alignment to happen without any action or setting on my part, so that every new Scene benefits from it.
4. As a user, I want to see an aligning step in job progress, so that I know what the job is doing during that time.
5. As a user, I want the job to still succeed when the Scene cannot be aligned confidently, so that I do not lose a long run over something that does not affect Queries.
6. As a user, I want a Match's bounding box to fit the object more tightly, so that I can locate it at a glance.
7. As a user, I want highlighted Instance points to stay exactly on their objects after alignment, so that Matches remain trustworthy.
8. As a user, I want Query results to be unaffected by alignment, so that finding objects works as before.
9. As a user with mixed portrait and landscape photos, I want alignment to work on my Capture, so that I do not have to take photos in one orientation.
10. As a user with a Scene that has no large flat surface, I want it to be aligned from the photos alone, so that alignment does not depend on having a visible floor.
11. As a user whose photos show mostly a table or floor from above, I want a poor estimate to be rejected instead of applied, so that the Scene is not rotated into a worse position.
12. As a user who photographed a Scene with a tabletop but no visible floor, I want the tabletop to serve as the level reference, so that I get the benefit of plane refinement.
13. As a user with an existing Scene, I want it to keep working unchanged, so that this feature does not break what I already have.
14. As a user with an existing tilted Scene, I want to be able to get an aligned version by creating a new Scene from the same photos, so that I have a path to the improvement.
15. As the owner, I want one log line per job saying which method was used, the rotation angle and how many Views agreed, so that I can judge whether alignment worked.
16. As the owner, I want that log line visible in the Colab notebook, so that I do not need any other tool to see it.
17. As the owner, I want the log to state why a Scene was not rotated, so that I can tell weak evidence from a defect.
18. As the owner, I want a technical failure in this stage to fail the job with an error, so that a broken install cannot silently leave every Scene tilted.
19. As the owner, I want the peak VRAM of the Aligning stage logged like the other stages, so that I can see its cost.
20. As the owner, I want GeoCalib to leave the GPU when the stage ends, also on failure, so that later stages have that memory.
21. As the owner, I want GeoCalib installed by the notebook's existing install step at a pinned commit, so that deployment stays a single action and is reproducible.
22. As the owner, I want every alignment threshold to be overridable from the notebook's `CONFIG` cell, so that I can tune it without changing code.
23. As a developer, I want a single documented convention for the up direction, so that spatial reasoning can rely on it later.
24. As a developer, I want a Reconstruction's points and cameras to always be in the same frame, so that later stages and the viewer stay consistent.
25. As a developer, I want the point-order contract preserved, so that the PLY, `instance_ids.bin` and the MV3DIS input still line up.
26. As a developer, I want alignment to be a rigid rotation, so that distances remain in metres and DA3's metric scale is untouched.
27. As a developer, I want the alignment decisions as pure functions with no model, so that they can be tested on CPU with hand-worked values.
28. As a developer, I want the model behind an interface of its own, so that the up-direction source can be swapped like the other stage models.
29. As a developer, I want the CPU suite to run without GeoCalib or torch, so that tests remain runnable on machines with no GPU.
30. As a developer, I want a GPU smoke test to catch a sign error in the up direction, so that an upside-down Scene cannot ship unnoticed.
31. As a developer, I want the fake Pipeline to report the new stage, so that the API and frontend are exercised against the real stage list.
32. As a developer, I want the decision and its rejected alternatives recorded in an ADR, so that nobody re-adds a post-hoc viewer rotation or changes the convention by accident.
33. As a developer, I want the v1 spec's stage list to match the Pipeline, so that the documentation stays true.

## Implementation Decisions

**Stage**
- A new stage named `aligning` runs between `fusing` and `superpoints`, making seven stages. It always runs. There is no setting that turns it off.
- It runs after Fusing because RANSAC needs the denoised geometry, and a rotation keeps the vertex order, so the point-order contract holds.
- It replaces the fused vertices with rotated ones and every View's camera-to-world with the same rotation applied. Triangles, colours, images and depth maps stay as they are; depth is in the camera's frame, so it remains valid.
- Later stages receive only aligned data. Instance centroids and bounding boxes are therefore computed in the aligned frame without any change to their code.

**Convention**
- Up is **−Y** in every Reconstruction. This matches the viewer's existing camera setting and the OpenCV camera convention, so the viewer's camera does not change.
- The rotation is the smallest one that takes the estimated up direction to −Y, and it must handle the case where the estimate is exactly opposite. It has no translation or scale: the origin stays where DA3 put it, and the floor is not moved to any particular height.

**Estimating up from photos (GeoCalib)**
- Use at most 12 Views, spaced evenly through the Views the View stride selected, or all of them when there are fewer.
- Estimate each View independently from the images held at depth-map resolution. GeoCalib resizes its input to 320 px itself, so full-resolution photos add little.
- Pass the Reconstructing stage's focal length, the mean of fx and fy, as a prior by default. A config option turns the prior off, because a wrong focal length could bias pitch and this has not been measured.
- Transform each per-View up vector into the world frame with that View's camera rotation.
- Consensus: take the estimate with the most estimates within 10° of it (itself included), breaking ties by View order. It succeeds when that group is more than 50% of the estimates. The result is the normalised mean of the group.
- A Capture with one View trivially reaches consensus. Two Views more than 10° apart do not.

**Refining up from geometry (RANSAC)**
- Run Open3D's plane segmentation repeatedly, removing each plane's inliers, for at most 5 planes, with a 3 cm inlier distance. Stop early when too few points remain.
- A plane qualifies when it holds at least 5% of the fused points and its normal is within 10° of the GeoCalib direction, ignoring the normal's sign.
- Use the normal of the largest qualifying plane, with its sign flipped to agree with the GeoCalib direction. Floor, tabletop and ceiling are all acceptable; no attempt is made to tell them apart.
- The search works on a copy of the points and never changes the Reconstruction's point set. It may subsample for speed.

**Outcomes**
- Consensus and a qualifying plane: rotate using the plane normal (method `plane`).
- Consensus and no qualifying plane: rotate using the GeoCalib direction (method `geocalib`).
- No consensus: do not rotate, and the job continues (method `none`).
- Technical errors (import failure, weights download failure, out of memory) are not a low-confidence outcome. They propagate and fail the job like any other stage's errors.

**Module shape**
- A new alignment module holds the pure decisions: consensus, plane selection, the rotation between two directions, and applying a rotation to points and cameras. None of these need a model or Open3D.
- A thin function wraps Open3D's plane search and returns candidate planes as a normal and an inlier fraction. Open3D is imported inside it.
- The GeoCalib wrapper lives behind a small interface, in the style of the Reconstructor interface, and returns per-View up vectors in camera frames as plain arrays. Its output holds no GPU tensors. `geocalib` and `torch` are imported only inside functions.
- The real Pipeline receives the aligner through its constructor with a GeoCalib default, and the stage runs inside the existing per-stage memory tracking. The model is loaded for the stage and released on exit, also on failure.

**Configuration**
- A new `AlignmentOptions` settings group holds: maximum Views, whether to use the focal prior, consensus angle, consensus fraction, plane inlier distance, maximum planes, minimum plane fraction and maximum plane angle. Defaults are 12, on, 10°, 0.5, 0.03 m, 5, 0.05 and 10°.
- It is added to `Settings` and passed to the stage in the server's component assembly, like every other stage's options.

**Dependencies**
- Install GeoCalib from `github.com/cvg/GeoCalib` at commit `1dedb29f94fc11b896c5f572005978aeebccd67f` through the GPU requirements file, not `pyproject.toml`. It is not vendored: unlike DA3, its requirements carry no version pins, so nothing conflicts with Colab's CUDA torch. This brings in `kornia` and `matplotlib`.
- Add `geocalib` to the ty whitelist of lazily imported third-party modules.
- Weights (CC BY 4.0) download through `torch.hub` at run time and are never committed. The code is Apache-2.0.

**Logging**
- Each job logs one line with: method, rotation angle in degrees, Views used, the consensus fraction and, when a plane was used, its inlier fraction.
- Log through a logger declared in the Colab logging configuration. A logger not declared there never reaches the notebook.
- Nothing about alignment is added to the HTTP API, the stored Scene metadata or the viewer. When a Scene is not rotated, the log is the only explanation.

**Frontend**
- The job-progress stage list gains `aligning` in its position. The viewer's camera is unchanged.

**Documents to update as part of this work**
- New ADR 0007, recording: GeoCalib plus RANSAC rather than either alone; rotation stored in the Reconstruction rather than applied by the viewer; after Fusing rather than before; no yaw alignment; and the −Y convention.
- The stage list in the v1 spec (`.scratch/scene-reconstruction-query/spec.md`).
- `AGENTS.md`, for what an agent cannot read from the code: the plane-search test needs Open3D and is skipped by the CPU suite, the same way as the fusion test, and how to run it; and the −Y up contract.
- `GLOSSARY.md` already defines Gravity alignment. Do not rename it; avoid the terms it lists, in particular "calibration" for this concept.

## Testing Decisions

A good test checks external behaviour with expectations worked out by hand, never by recomputing the code's own formula. Do not unit-test or mock anything that loads a model.

- **Seam A, the HTTP API with `FakePipeline`.** The job-stage progression includes `aligning` between `fusing` and `superpoints` and counts seven stages. Prior art: the job-status test in the API suite. `FakePipeline` walks the stage enumeration, so it reports the new stage without changes.
- **Unit, pure functions on CPU.** Prior art: the existing unit tests of adapters, Mask selection and the Embedding stage's helpers.
  - Rotation between directions: up `(0, 0, 1)` to `(0, -1, 0)` maps the point `(1, 2, 3)` to `(1, -3, 2)`. The opposite case, up `(0, 1, 0)`, maps up to `(0, -1, 0)` and preserves lengths and determinant 1.
  - Applying a rotation: point order and count are unchanged, and a point projected through a rotated camera lands on the same pixel as before rotation.
  - Consensus: one estimate succeeds; two estimates 40° apart fail; two of three within 10° succeed with their normalised mean; exactly half does not succeed.
  - Plane selection: picks the largest qualifying plane rather than the largest overall (a large wall against a smaller floor); rejects a plane under the minimum fraction; rejects a plane over the angle limit; flips a normal that points against the GeoCalib direction; returns nothing for an empty candidate list.
  - The outcome ladder gives `plane`, `geocalib` and `none` for the three situations, and `none` leaves points and cameras unchanged.
  - Settings overrides for the new group are accepted, and unknown keys are refused, as with the existing settings tests.
- **Open3D-dependent.** A synthetic floor and wall yield candidates with the expected normals. It needs Open3D, so the CPU suite skips it, like the fusion test.
- **Seam B, real Pipeline on GPU, `@pytest.mark.gpu`.** Prior art: the real Pipeline smoke test.
  - The stage list includes `aligning`.
  - After a run, on a Capture with photos held roughly upright, the mean camera up direction is near −Y. This guards the sign of GeoCalib's vector, because an inverted sign would put it near 180°.
  - GeoCalib is off the GPU after a run: the existing assertion that only SigLIP2 stays there still holds.
- **Acceptance.** There is no ground truth and no angular measurement. The feature is accepted by reading the log line and looking at two or three real Captures in the viewer, comparing a Scene created before the change with one created after it from the same photos.

## Out of Scope

- Yaw alignment: walls are not made parallel to the horizontal axes, and bounding boxes of objects standing diagonally stay larger than their object.
- Moving the origin or placing the floor at a fixed height.
- Migrating existing Scenes. They keep their frame; a new Scene from the same photos gets the new one.
- A setting to disable the stage.
- Showing alignment status in the API or the frontend.
- GeoCalib's `camera_R_rig` joint estimation. It is designed for rigid rigs with accurate rotations; consider it only if per-View estimation proves too inaccurate.
- Weighting estimates by GeoCalib's reported uncertainty.
- Reading gravity from IMU data or EXIF.
- A tool or dataset that measures alignment error.
- Correcting warped geometry. A rigid rotation cannot straighten a floor DA3 reconstructed as curved.

## Further Notes

- **GeoCalib's vector points up.** Despite the name, `Gravity.vec3d` is `(0, -1, 0)` for a level camera according to `Gravity.from_rp`, which is up in the OpenCV camera frame. This was read from the code at the pinned commit, not run. The Seam B test confirms it.
- **Thresholds are unvalidated starting values**, like `DEFAULT_MATCH_THRESHOLD`. Expect to tune them on real Captures.
- **Accepted risk:** when the largest plane within 10° of the GeoCalib direction is really a slope, such as an 8° ramp, the Scene ends up tilted by that amount.
- **Accepted consequence of having no off switch:** if GeoCalib cannot run, for example because its weights cannot be downloaded, every job fails at this stage until it is fixed.
- **Likely failure case:** photos taken straight down at a surface show little horizon, so GeoCalib estimates may scatter. The expected result is method `none`, not a wrong rotation, but this has not been tried.
- **Hypothesis behind this work:** the tilt comes from DA3's world frame not being level. This follows from the code, but it has not been measured on a real Capture. The first Colab run's logged rotation angle will show how large the correction really is.
