# 02: Gravity-align new Scenes with GeoCalib

**What to build:** Every new Scene is Gravity-aligned from its photos: physical up becomes −Y, the floor looks close to level in the viewer, and Instance centroids and bounding boxes are computed in the aligned frame. When the Views do not agree on up, the Scene is left unrotated and the job still succeeds. A technical failure fails the job. Each job writes one log line, visible in the Colab notebook, saying what happened. See "Stage", "Convention", "Estimating up from photos", "Outcomes", "Module shape", "Configuration", "Dependencies" and "Logging" in `.scratch/gravity-alignment/spec.md`.

**Blocked by:** 01: Show the Aligning stage in job progress

**Status:** ready-for-agent

- [ ] The stage estimates up per View with GeoCalib, using at most 12 evenly spaced Views, or all of them when there are fewer.
- [ ] The Reconstructing stage's focal length is passed as a prior by default, and a config option turns it off.
- [ ] Up vectors are moved to the world frame and consensus follows the spec: more than 50% of estimates within 10°, with ties broken by View order.
- [ ] With consensus, points and every View's camera get the same smallest rotation taking up to −Y, including when the estimate is exactly opposite. There is no translation or scale.
- [ ] Without consensus, points and cameras are unchanged and the job succeeds with method `none`.
- [ ] Import, weights-download and out-of-memory errors fail the job.
- [ ] Point order and count are unchanged, and triangles, colours, images and depth maps are not modified.
- [ ] GeoCalib sits behind its own interface, is imported only inside functions, returns no GPU tensors, and leaves the GPU when the stage ends, also on failure.
- [ ] The stage runs inside the per-stage VRAM tracking.
- [ ] `AlignmentOptions` holds this ticket's settings with the spec's defaults, can be overridden through the config, and has no switch to disable the stage.
- [ ] GeoCalib is installed at the spec's pinned commit through the GPU requirements, not `pyproject.toml`, and is whitelisted for ty.
- [ ] One log line per job gives method, rotation angle, Views used and consensus fraction, through a logger declared in the Colab logging configuration.
- [ ] Unit tests with hand-worked values cover the rotation (including `(0, 0, 1)` to `(0, -1, 0)` mapping `(1, 2, 3)` to `(1, -3, 2)`, and the opposite case), applying it (a point projects to the same pixel before and after), consensus (one estimate; two 40° apart; two of three within 10°; exactly half), and the settings overrides.
- [ ] A Seam B test asserts that the mean camera up is near −Y after a run on a Capture with photos held roughly upright. It is written here and run in ticket 04, because there is no GPU locally.
- [ ] ADR 0007 records GeoCalib plus RANSAC rather than either alone, rotation stored in the Reconstruction rather than applied by the viewer, after Fusing, no yaw alignment, and the −Y convention.
- [ ] `AGENTS.md` states the −Y up contract.
- [ ] Ruff, ty and the CPU pytest suite pass, and every public symbol has a Google-style docstring.
