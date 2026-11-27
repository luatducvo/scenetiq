# Pipeline stages

## Point order

The PLY, `instance_ids.bin` and the MV3DIS input all use the TSDF mesh vertex order. Points keep that order and count from fusion onward: never reorder or filter them.

## MV3DIS

- Always pass primitive ids: `point_level=True` is broken upstream and would allocate N×N matrices.
- The mask label map is `uint8`, so the Masking stage caps Masks at 255 per View.

## Instance masks (ADR 0011)

`plan_views` gives every visible point a splat radius from the Fusing stage's voxel size, and the Embedding and Captioning stages draw the Instance's mask from those squares. Without the point spacing a plan has no radii, and both stages silently fall back to plain bounding-box crops, which is how a floor takes the embedding and the Caption of what stands on it. A stage that changes the point spacing after fusion must pass the new spacing to `plan_views`.

## Captioning (ADR 0008)

Replies are schema-forced by xgrammar's transformers logits processor, the same way as the [Query parser](query.md#query-parser-adr-0009).

- A processor keeps matcher state, so build a new one for every `generate` call; reuse the compiled grammar.
- Its vocabulary is sized by the model's output layer, not by the tokenizer, which is smaller.
- Keep `parse_caption` as a second check: a reply can still be cut off at `max_new_tokens`.
- **Aspect order is a storage contract:** `caption_embeddings.npy` keeps one row per Aspect in `Aspect` order, and the Captioning prompt, the parser's schema and its `aspects` field use the Aspect values. Keep `Aspect` members in their current order and names.

## Orientation

- **Up is −Y in every Reconstruction** (ADR 0007), so "above" means a smaller y. A Scene whose Views gave no consensus stays in DA3's frame, and its vertical Relations (above, below, on, under, tallest, lowest) still use DA3's Y axis: they can be wrong, with no warning at Query time. Only the job log and the job's notes (Processing page) say a Scene was left unaligned.
- **GeoCalib's `gravity.vec3d` points up**, despite its name: it is `(0, -1, 0)` for a level camera. It follows a photo's roll only to about ±45°; beyond that it still answers, but wrongly (one photo: 90° gave −15.5°, 180° gave −0.1°). Consensus misses this when every View is rolled the same way.
- **Only the frontend applies EXIF orientation** (ADR 0010). DA3, the Embedding stage and the Captioning stage read raw pixels with `Image.open`, so a portrait photo sent by another client as landscape plus an EXIF flag reaches every stage sideways.

## Cancelling a job

Cancelling is cooperative. The job runner sets a flag; the Pipeline stops only where it calls the reporter's `stage()` or `checkpoint()`, which raise `JobCancelledError`. The job thread is never killed or interrupted. A long loop added to a stage takes a checkpoint callable, as Masking (per View) and Captioning (per Instance) do; a single model call cannot be interrupted. A cancelled job's Scene directory is deleted, and its record and events stay.
