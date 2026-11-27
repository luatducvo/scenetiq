# 01: Colab acceptance of the new Match ranking and the Instance masks

**What to build:** On a real Colab runtime, confirm that the changes of ADR 0011 work on real Scenes and judge their unvalidated defaults, with `python -m scenetiq.evaluation` wherever a number can replace an impression. Nothing in ADR 0011 has run on a GPU: it was written from the code after a Query for "cái ghế" coloured the floor.

**Status:** ready-for-human

- [ ] `uv run pytest -m gpu` passes.
- [ ] A Capture is reconstructed again after the change, so its embeddings and Captions come from masked crops.
- [ ] The floor, a wall and a table top each have a Name that names the surface, not what stands on it. Note any that do not, with the crop if possible.
- [ ] The Instance masks look right on a few crops: no dotted holes on close objects, no mask far wider than the object. Note the `embedding.max_splat_radius` that fixes it if the default does not.
- [ ] The time of the Embedding and Captioning stages is noted and compared with a job before the change.
- [ ] The parser fills `aspects` on its first parse with no xgrammar error, and names sensible Aspects for "chair", "red chair", "wooden thing" and "something to sit on", in English and in Vietnamese.
- [ ] "cái ghế" returns the chairs and not the floor. The number of Matches is noted for a Query with one, with several and with no fitting Instance.
- [ ] At least 20 Queries on at least 2 Scenes are labelled by hand (`--instances SCENE_ID` lists the ids) and scored. The precision, recall and share of right best Matches are recorded here.
- [ ] `query_score_ratio` is tried at 0.3, 0.5 and 0.7 on those labels, and `embedding.masked_weight` at 0, 0.5 and 0.8 (each needs a new job); the best values are recorded and made the defaults.
- [ ] Any surprise, and any default that had to change, is recorded under `## Comments` in this file.
