# Spec: Captioning stage and Caption-aware Query ranking

Status: ready-for-agent

Vocabulary follows `GLOSSARY.md` (Caption, Match and Instance were redefined for this work). Respect the ADRs in `docs/adr/`, in particular 0008 (Captions, separate text embedding model, RRF top-K) and 0007 (up is −Y).

## Problem Statement

A user opens a Scene and types a Query. Today the only thing that answers is SigLIP2's image embedding of each Instance, compared with the Query text and cut off by a threshold. Queries phrased the way people actually talk often fail: a synonym ("couch" for a sofa), a material ("the wooden thing"), or a use ("something to sit on"). When nothing clears the threshold the user gets a single Low-confidence Match, which is often the wrong object. The user also has no idea what the system thinks each object is: a Match is only a highlighted set of points with a score.

## Solution

Every new Scene gets a Captioning stage after Embedding. A vision-language model running on the Colab GPU looks at each Instance in the Views that see it best, with the Instance boxed in each crop, and reads a Caption: its Name, up to three Synonyms, its Attributes (colour, material…) and its Use. The Caption is turned into one text document and embedded by a dedicated text embedding model.

A Query is then ranked twice: against the Instances' image embeddings, as before, and against their Caption embeddings. The two rankings are fused with Reciprocal Rank Fusion, and the top K Instances are the Matches. There is no threshold and no Low-confidence Match: a Query on a Scene with Instances always returns Matches, ranked, each with a normalised fused score. In the viewer, each Match shows its Name next to the score in the Match list and as a label on its bounding box. Scenes created before this change keep working, ranked by image similarity alone.

## User Stories

1. As a user, I want a Query using a synonym ("couch") to find an object the system would call something else ("sofa"), so that I do not have to guess the exact word.
2. As a user, I want a Query naming a material ("wooden table", "metal shelf") to favour objects of that material, so that I can narrow down similar objects.
3. As a user, I want a Query naming a colour to favour objects of that colour, so that "the red chair" finds the red one.
4. As a user, I want a Query describing a use ("something to sit on", "where I put my laptop") to find objects with that use, so that I can search by purpose instead of name.
5. As a user, I want to type Queries in Vietnamese as well as English, so that I can search in my own language even though Captions are written in English.
6. As a user, I want every Query on a Scene with Instances to return Matches, so that I always get something to look at.
7. As a user, I want Matches ranked from most to least relevant, so that the best candidate comes first.
8. As a user, I want each Match to show a score between 0 and 1, so that I can compare Matches within one Query.
9. As a user, I want to see each Match's Name next to its score in the Match list, so that I know what the system thinks the object is.
10. As a user, I want each Match's Name drawn as a label on its 3D bounding box, so that I can read what was found without looking away from the Scene.
11. As a user, I want a Match with no Caption to still appear, without a Name, so that an object the model could not describe is not hidden.
12. As a user, I want only the Name shown, not attributes or uses, so that the viewer stays uncluttered.
13. As a user, I want to see a captioning step in job progress after embedding, so that I know what the job is doing during that time.
14. As a user, I want Captioning to happen without any action or setting on my part, so that every new Scene benefits from it.
15. As a user, I want the job to still succeed when the model returns something unusable for one object, so that I do not lose a long run over one Caption.
16. As a user, I want the job to fail with an error when the captioning model cannot load or runs out of memory, so that I know to fix the setup rather than receive Scenes silently without Captions.
17. As a user with existing Scenes, I want them to keep working unchanged, ranked by image similarity, so that this feature breaks nothing I already have.
18. As a user with an existing Scene, I want to get Captions by creating a new Scene from the same photos, so that I have a path to the improvement.
19. As a user, I want Queries to stay fast after this change, so that searching still feels interactive.
20. As a user, I want the no-Match and Low-confidence flags gone from the viewer, so that the interface matches how Matches now work.
21. As the owner, I want the number of Matches per Query (K) to be a configuration value, defaulting to 1, so that I can show more candidates without code changes.
22. As the owner, I want the RRF constant to be a configuration value, defaulting to 60, so that I can tune the fusion.
23. As the owner, I want the captioning model, the number of Views per Instance, the crop context, the image resolution cap and the generation length overridable from the notebook's `CONFIG` cell, so that I can tune or swap them without code changes.
24. As the owner, I want the text embedding model and its query instruction overridable from the notebook's `CONFIG` cell, so that I can try another model.
25. As the owner, I want the old Match threshold setting removed, so that no setting pretends to do something it no longer does.
26. As the owner, I want the captioning model to leave the GPU when the stage ends, also on failure, so that the next job and the resident models have that memory.
27. As the owner, I want the text embedding model to stay on the GPU between jobs, like SigLIP2, so that Queries do not wait for a model load.
28. As the owner, I want the peak VRAM of the Captioning stage logged like the other stages, so that I can see its cost.
29. As the owner, I want one log line per job saying how many Instances got a Caption and how many did not, so that I can see whether the model is behaving.
30. As the owner, I want those log lines visible in the Colab notebook, so that I need no other tool.
31. As the owner, I want both new models installed by the notebook's existing install step, with no new install step, so that deployment stays a single action.
32. As the owner, I want model weights downloaded at run time and never committed, so that the repo stays small and licence-clean.
33. As a developer, I want the API's Instance listing to include each Instance's Name (or null), and nothing else from the Caption, so that the frontend can label Matches.
34. As a developer, I want each Match in the Query response to carry its fused score plus its image score and Caption score, so that I can see why it ranked where it did.
35. As a developer, I want the Captioning stage to reuse the Embedding stage's choice of Views, so that a Caption and an embedding describe the same evidence and the selection is computed once.
36. As a developer, I want only Instances that survive the Embedding stage to be captioned, so that no work is spent on Instances that are dropped.
37. As a developer, I want the captioning model behind an interface of its own, so that it can be swapped like the other stage models.
38. As a developer, I want the text embedding model behind an interface of its own, used by both the Pipeline and the API, so that one loaded instance serves both.
39. As a developer, I want parsing the model's reply, building the Caption document and fusing rankings to be pure functions, so that they can be tested on CPU with hand-worked values.
40. As a developer, I want the CPU suite to run without either new model, torch or transformers, so that tests stay runnable on machines with no GPU.
41. As a developer, I want the fake Pipeline to report the new stage and produce Instances with and without Captions, so that the API and frontend are exercised against the real behaviour.
42. As a developer, I want a fake Caption encoder that maps known strings to known vectors, so that Query ranking can be tested with hand-worked values.
43. As a developer, I want the point-order contract untouched by this stage, so that the PLY, the instance id array and the Instance list still line up.
44. As a developer, I want stored Scenes without Caption data to load as Instances without Captions, so that no migration is needed.
45. As a developer, I want the decision and its rejected alternatives recorded in an ADR, so that nobody re-adds a threshold or a weighted score by accident.
46. As a developer, I want the v1 spec, the configuration and pipeline docs and AGENTS.md to match the new behaviour, so that the documentation stays true.

## Implementation Decisions

**Stage**
- A new stage named `captioning` runs after `embedding`, making it the last stage. It always runs; there is no setting that turns it off.
- It runs after the Embedding stage has dropped the Instances no View sees well enough, and captions only the survivors.
- The Embedding stage's per-Instance View planning (which Views see the Instance best, and its visible pixels in each) is extracted into one function. The Pipeline computes the plan once and hands it to both stages. Captioning uses the first N Views of each plan (N configurable, default 3).

**Caption**
- A Caption has a Name (required, non-empty), up to three Synonyms, a list of Attributes and a Use, all English free text. It is not drawn from any fixed class list.
- An Instance gains an optional Caption and an optional Caption embedding.

**Captioning model**
- Default `Qwen/Qwen3-VL-8B-Instruct` in bf16, run locally on the Colab GPU through transformers, behind a captioner interface that takes a list of images and returns the model's raw text.
- One prompt per Instance, holding its chosen Views. Each image is a context crop around the Instance (the tight box grown by a configurable fraction, default 0.6) with the tight box drawn on it, from the original full-resolution photo. The prompt asks for JSON with `name`, `synonyms`, `attributes` and `use` describing the boxed object.
- Batch size 1, greedy decoding, a configurable cap on new tokens (default 128) and on image pixels.
- The model loads on first use within the stage and is released when the stage ends, also on failure, following the existing load/release pattern of the Reconstructing and Masking stages. The stage's output holds no GPU tensors.
- The model's reply is parsed tolerantly: JSON inside code fences or with surrounding text is accepted, Synonyms beyond three are dropped, and a reply without a usable Name yields no Caption.

**Text embedding model**
- Default `Qwen/Qwen3-Embedding-0.6B`, behind a Caption encoder interface with two operations: encode a batch of documents and encode one Query (with the model's query instruction prefix). Both return L2-normalised vectors.
- One instance serves the Pipeline (Caption documents) and the API (Query text). It stays on the GPU between jobs and loads under a lock, like SigLIP2, because Queries and the job run on different threads.
- Each Caption becomes one document combining Name, Synonyms, Attributes and Use (for example "sofa; also: couch, settee; grey, fabric; used for sitting").

**Query**
- A Query is encoded twice: by SigLIP2's text tower and by the Caption encoder.
- Two rankings by cosine similarity: every Instance against its image embedding, and every Instance that has a Caption against its Caption embedding.
- Reciprocal Rank Fusion: an Instance's fused value is the sum over the rankings it appears in of `1 / (rrf_k + rank)`, with ranks from 1. An Instance without a Caption gets only its image term.
- Matches are the top K by fused value, ties broken by image similarity then Instance id. A Scene with fewer than K Instances returns all of them; a Scene with no Instances returns none.
- The reported score is the fused value divided by its maximum possible value `channels / (rrf_k + 1)`, where `channels` is the number of rankings that exist for the Reconstruction (2, or 1 when no Instance has a Caption), so it lies in [0, 1].
- The threshold, its default, the Low-confidence Match and the text encoder's score calibration (`to_score`) are removed.

**Configuration**
- New settings groups for Captioning (model id, Views per Instance, context fraction, pixel cap, max new tokens) and for the Caption encoder (model id, query instruction, batch size).
- New top-level settings `query_top_k` (default 1) and `rrf_k` (default 60). The `match_threshold` setting is removed.
- All are added in the one configuration module and passed to their stage in the server's component assembly, which now returns the Pipeline, the SigLIP2 text encoder and the Caption encoder.

**Storage**
- The Instance metadata file gains a `caption` object per Instance (null when there is none).
- A new Caption embeddings file holds one row per Instance, aligned with the Instance metadata, with a zero row where the Caption is null. It is written only when at least one Instance has a Caption.
- Scenes stored before this change have neither and load with no Captions. No backfill.

**HTTP API**
- Instance listing: each Instance gains `name` (string or null). Synonyms, Attributes and Use are not exposed.
- Query: `{matches: [{instance_id, score, image_score, caption_score}]}`, with `caption_score` null for an Instance without a Caption. The `low_confidence` field is removed.

**Frontend**
- The job-progress stage list gains `captioning` after `embedding`.
- The Match list shows each Match's Name (when present) next to its score. Each Match's bounding box carries its Name as a label. The Low-confidence flag and its styling are removed.

**Dependencies**
- Both models are installed through the existing GPU requirements via transformers; the requirements must pin a transformers version that supports Qwen3-VL. Nothing new goes into the CPU project dependencies. `transformers` and `torch` stay imported only inside functions.

**Logging**
- One line per job with the number of Instances captioned and the number without a Caption, through a logger the Colab logging configuration lets through. Peak VRAM is logged through the existing per-stage tracking.

**Documents to update as part of this work**
- `GLOSSARY.md` and ADR 0008 are already written.
- The v1 spec: stage list, Query module, user stories about the threshold and the Low-confidence Match, and the Out of Scope line about class labels on Instances.
- AGENTS.md: replace the note on scores and the unvalidated threshold; add that the captioning model leaves the GPU after its stage while the Caption encoder stays resident under a lock; the transformers minimum version if one is needed.
- The configuration and pipeline docs.

## Testing Decisions

A good test checks external behaviour through a seam, with expectations worked out by hand, never by recomputing the code's own formula. Never unit-test or mock anything that loads a model: both new models are covered only by Seam B.

- **Seam A, the HTTP API with fakes (main suite, CPU).** The fake Pipeline returns a hand-built Reconstruction in which some Instances have Captions and known Caption embeddings and one does not; a new fake Caption encoder maps known Query strings to known vectors, alongside the existing fake text encoder. Prior art: the existing Query, job-status and Instance-listing API tests.
  - A Query whose image ranking puts an Instance low but whose Caption ranking puts it first returns that Instance first, with a hand-worked normalised score.
  - An Instance without a Caption appears with a null Caption score and only its image term.
  - A Scene with no Captions is ranked by image similarity alone and normalised over one channel.
  - `query_top_k` cuts the Matches; a Scene with fewer Instances returns all; an empty Scene returns none.
  - The response has no `low_confidence` field.
  - The Instance listing returns `name`, or null.
  - Job progress includes `captioning` after `embedding`.
  - A Scene directory written in the old format (no Caption data) still lists Instances and answers Queries.
- **Unit, pure functions on CPU.** Prior art: the Embedding helper and alignment unit tests.
  - Parsing the model reply: plain JSON, JSON in code fences, JSON with trailing text, a missing or empty Name (no Caption), more than three Synonyms (truncated), malformed text (no Caption).
  - Building the Caption document from a Caption.
  - RRF fusion and normalisation on small hand-worked rankings, including ties and an Instance absent from one ranking.
  - The extracted View planning returns the same Views as before the refactor on a hand-built case.
  - The boxed context crop: crop bounds and drawn box position on a hand-built case.
  - Settings: new groups and keys accepted, unknown keys refused, `match_threshold` no longer accepted; component assembly for the fake Pipeline returns the three components.
- **Seam B, real Pipeline on GPU, `@pytest.mark.gpu`.** Prior art: the real Pipeline smoke test.
  - The stage list includes `captioning`.
  - At least one Instance has a Caption with a non-empty Name.
  - A Query returns `query_top_k` Matches.
  - After a run, only SigLIP2 and the Caption encoder remain on the GPU.
- **Acceptance.** There is no ground truth and no retrieval metric. The feature is accepted by reading the captioning log line and trying synonym, material, colour and use Queries on two or three real Captures, comparing with a Scene created before the change from the same photos.

## Out of Scope

- Showing Synonyms, Attributes or Use anywhere in the API or the frontend.
- Backfilling Captions for existing Scenes.
- A relevance threshold, a Low-confidence Match, or any absolute notion of "no good Match".
- Choosing K per Query in the UI or the API.
- External VLM or embedding APIs.
- Larger captioning models by default (32B was rejected for VRAM headroom; the model id stays configurable).
- Captions in languages other than English.
- Spatial reasoning Queries.
- A tool or dataset that measures retrieval quality.
- Editing or correcting Captions by hand.

## Further Notes

- **Scores are relative.** A fused score depends on ranks, so it cannot be compared across Queries or Scenes, and a top Match may still be irrelevant when nothing in the Scene fits.
- **Instances without a Caption are disadvantaged** in the fused ranking, because they get one term instead of two. Accepted; the log line shows how often it happens.
- **VRAM:** the 8B model in bf16 takes about 17 GB while SigLIP2 (~3.5 GB) and the Caption encoder (~1.2 GB) stay resident. The Captioning stage is the only new transient load.
- **Runtime:** one generation per surviving Instance, with up to three images each, is added to every job, plus one model load. Not measured yet.
- **EXIF orientation is still not applied**, so a portrait photo stored sideways reaches the captioning model sideways, as it already reaches every other stage.
- **Defaults are unvalidated starting values** (K, the RRF constant, Views per Instance, the context fraction). Expect to tune them on real Captures.
