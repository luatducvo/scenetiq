# 03: Refine up with a RANSAC plane

**What to build:** When a Scene has a suitable horizontal surface, such as a floor, tabletop or ceiling, its up direction comes from that plane's normal, which gives a more exact level than the photos alone. When none qualifies, the GeoCalib direction is kept as before. The log line says which method was used. See "Refining up from geometry" and "Outcomes" in `.scratch/gravity-alignment/spec.md`.

**Blocked by:** 02: Gravity-align new Scenes with GeoCalib

**Status:** ready-for-agent

- [x] After consensus, Open3D plane segmentation finds up to 5 planes with a 3 cm inlier distance, removing each plane's inliers, and stops early when too few points remain.
- [x] A plane qualifies with at least 5% of the fused points and a normal within 10° of the GeoCalib direction, ignoring sign.
- [x] The largest qualifying plane's normal is used, flipped to agree with the GeoCalib direction, and the job logs method `plane` with its inlier fraction.
- [x] With no qualifying plane the job uses the GeoCalib direction and logs method `geocalib`.
- [x] Without consensus no plane search changes the outcome: the result stays `none` with nothing rotated.
- [x] The search works on a copy, may subsample, and never changes the Reconstruction's point set or order.
- [x] Open3D is imported only inside functions, and the selection logic is pure and needs no Open3D.
- [x] `AlignmentOptions` gains the inlier distance, maximum planes, minimum plane fraction and maximum plane angle with the spec's defaults.
- [x] Unit tests with hand-worked values cover selection (the largest qualifying plane beats a larger wall; too small rejected; over the angle limit rejected; an opposite normal is flipped; empty list gives nothing) and all three outcomes.
- [x] A test with a synthetic floor and wall checks the candidate normals through Open3D, and the CPU suite skips it like the fusion test.
- [x] `AGENTS.md` says how to run that test.
- [x] ADR 0007 still matches what was built.
- [x] Ruff, ty, the CPU pytest suite and the Open3D test pass, and every public symbol has a Google-style docstring.
