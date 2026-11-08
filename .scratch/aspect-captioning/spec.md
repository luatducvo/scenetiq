# Spec: Multi-Aspect Sentence-Level Captioning

Status: ready-for-agent

Vocabulary follows `GLOSSARY.md`; this work adds **Aspect** and **Description** and redefines **Caption**. Respect the ADRs in `docs/adr/`, in particular 0008 (Captions, separate text embedding model, RRF top-K), which this work amends in place. Builds on `.scratch/instance-captioning/spec.md`, which describes the Captioning stage as it stands.

## Problem Statement

A user types a Query to find an object in a Scene. The Captioning stage already reads each Instance's Name, Synonyms, Attributes and Use, but it squeezes all of them into one keyword-style document ("sofa; also: couch, settee; grey, fabric; used for sitting") and embeds that as a single vector. A Query that describes one side of an object in a full sentence, such as its look ("a long grey thing with wide armrests"), what it is made of ("something upholstered with wooden legs") or what it is for ("where I can lie down and watch TV"), is compared with a vector that blends every side at once. Whatever matches gets diluted by everything that does not. The Captioning model is also only asked for JSON, so a reply can come back malformed and the Instance silently loses its Caption.

## Solution

The vision-language model writes one short English sentence, a Description, for each of four fixed Aspects of an Instance:

- **Identity**: what it is and what else people call it.
- **Appearance**: colour, shape and relative size.
- **Material**: what it is made of and its parts.
- **Use**: what it is used for.

Each Description is embedded on its own. When a Query runs, an Instance's Caption score is that of its best-matching Description. A question about colour is therefore answered by the Appearance sentence, and a question about purpose by the Use sentence.

The model's reply is forced, token by token, to follow a fixed JSON schema, so the only remaining way to lose a Caption is a reply cut off by the token limit. The user still sees only the Name, next to the Match and on its box.

The old one-document Caption is removed.

## User Stories

1. As a user, I want a Query that describes an object's look in a full sentence to find that object, so that I can search the way I would describe it to a person.
2. As a user, I want a Query about what an object is made of to find it, so that "the thing with metal legs" works.
3. As a user, I want a Query about what an object is for to find it, so that "somewhere to put my coffee" works.
4. As a user, I want a Query that names an object or one of its Synonyms to keep working as well as before, so that the change costs me nothing for short Queries.
5. As a user, I want a Query that matches one Aspect strongly not to be dragged down by Aspects it says nothing about, so that the right object wins.
6. As a user, I want to keep seeing the Name next to each Match and on its box, so that the display does not change.
7. As a user, I want Instances whose reply could not be read to still be findable by image similarity, so that a Captioning failure never hides an object.
8. As a user, I want almost every Instance to get a Caption, so that the Caption channel actually helps.
9. As a user, I want Captioning failures to be rare and explained in the log, so that I can tell a cut-off reply from a model problem when I judge results by hand.
10. As a user, I want the job to keep going when one Instance's reply is unusable, so that one bad reply does not cost me the Scene.
11. As a user judging results by hand, I want each Match to say which Aspect of the Caption matched best, so that I can see why it was chosen.
12. As a user judging results by hand, I want the raw Caption cosine of the best Description beside the fused score, so that I can compare channels.
13. As a user, I want the Descriptions to talk only about the boxed object, so that a neighbouring object's colour or material does not leak into its Caption.
14. As a user, I want Descriptions in plain English short sentences, so that the Caption encoder compares like with like.
15. As a user, I want the Captioning stage not to become much slower, so that a Scene still finishes in a reasonable time.
16. As a user, I want the job progress to still show the Captioning stage, so that I can follow the job as before.
17. As a user, I want a Scene where the model left an Aspect blank to still use the Aspects it did fill, so that a partial Caption still helps.
18. As a user, I want an Instance whose Identity sentence came back blank to still be findable by its Name and Synonyms, so that the most important Aspect is never missing.
19. As a user, I want Matches to still be the top K by Reciprocal Rank Fusion with no threshold, so that Query behaviour stays as ADR 0008 describes it.
20. As a user, I want a Scene where no Instance got a Caption to be ranked by image similarity alone, as before.
21. As a developer, I want Caption to be Name, Synonyms and one Description per Aspect, with Attributes and Use removed, so that the model has one shape.
22. As a developer, I want the four Aspects to be a fixed, ordered set defined once in the domain, so that prompt, schema, storage and scoring cannot drift apart.
23. As a developer, I want the JSON schema the model must follow to be one constant in the Captioning stage, not a setting, so that the parser and the schema cannot disagree.
24. As a developer, I want the schema to require every key, cap Synonyms at three and bound every string's length, so that replies stay short and complete.
25. As a developer, I want the schema to allow an empty Description, so that the model is not forced to invent an Aspect it cannot see.
26. As a developer, I want constrained decoding to come from a logits processor plugged into the existing generate call, so that the chat template and greedy decoding stay ours.
27. As a developer, I want the schema compiled once per model load and a fresh logits processor built for every generate call, so that the processor's one-use limit is respected.
28. As a developer, I want the reply parser kept as a second check, so that a reply cut off by the token limit yields no Caption instead of a crash.
29. As a developer, I want the default token limit raised to 320, so that four sentences plus Name and Synonyms fit.
30. As a developer, I want an Identity Description generated from Name and Synonyms when the model leaves it empty, so that every Caption has at least one Description.
31. As a developer, I want each non-empty Description embedded separately as a Caption-encoder document, so that each Aspect has its own vector.
32. As a developer, I want the Caption ranking to order Instances by their best Description's cosine, so that there is still exactly one Caption ranking and RRF normalisation is unchanged.
33. As a developer, I want Matches to carry the Aspect whose Description scored best, or none, so that the API can expose it.
34. As a developer, I want `instances.json` to store a Caption as Name, Synonyms and the four Descriptions, so that stored Scenes round-trip.
35. As a developer, I want Caption embeddings stored as one array of Instances × Aspects × dimensions in Aspect order, with zero rows for empty Descriptions and Instances without a Caption, so that storage stays aligned with `instances.json`.
36. As a developer, I want the one-document Caption, its document builder and its storage format deleted, so that only one method exists.
37. As a developer, I want the fake Pipeline and fake Caption encoder to produce Captions with known Descriptions and known vectors, so that Seam A can test per-Aspect ranking.
38. As a developer, I want xgrammar pinned in the GPU requirements without replacing Colab's CUDA torch, so that installation stays safe.
39. As a developer, I want transformers pinned to 5.22.2, so that the GPU stack is on the latest patch release.
40. As a developer, I want ADR 0008, GLOSSARY, the pipeline and configuration docs and AGENTS.md updated in the same change, so that the docs match the code.
41. As a maintainer, I want AGENTS.md to record the one-processor-per-generate rule and the xgrammar pin, so that a future change does not break them.
42. As a maintainer, I want the Colab acceptance ticket to check the vocabulary size passed to xgrammar, whether it enforces the three-Synonym cap, the extra time per Instance, and that torch is unchanged after install, so that the unverified parts get checked.

## Implementation Decisions

**Domain**
- A new ordered `Aspect` enumeration with the values identity, appearance, material and use, in that order. It is the single source for prompt wording, schema keys, storage order and the `caption_aspect` values.
- `Caption` becomes Name, Synonyms (at most three) and one Description per Aspect. A Description is a string and may be empty. Attributes and Use are removed.
- An Instance's Caption embeddings become one L2-normalised vector per non-empty Description, keyed by Aspect, replacing the single Caption embedding. The invariant stays: no Caption embeddings without a Caption.

**Captioning stage**
- The prompt asks for the object inside the red box only, gives one plain-English sentence of at most about 25 words per Aspect, and describes each field in words. It has no few-shot example.
- The required schema (constant, no setting):

  ```json
  {
    "type": "object",
    "properties": {
      "name":       {"type": "string", "minLength": 1, "maxLength": 40},
      "synonyms":   {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 3},
      "identity":   {"type": "string", "maxLength": 200},
      "appearance": {"type": "string", "maxLength": 200},
      "material":   {"type": "string", "maxLength": 200},
      "use":        {"type": "string", "maxLength": 200}
    },
    "required": ["name", "synonyms", "identity", "appearance", "material", "use"],
    "additionalProperties": false
  }
  ```

- The reply parser keeps its tolerance and stays the second check. It returns no Caption when the Name is missing or empty, or when the JSON is unreadable (for example, cut off by the token limit). It logs why at the Captioning logger. When Identity is empty it builds one from Name and Synonyms, for example "A sofa, also called a couch or settee.".
- The view selection, boxed crops, the three Views per Instance and context expansion are unchanged.

**VLM wrapper**
- Constrained decoding uses xgrammar's transformers logits processor, passed to the existing `generate` call. Greedy decoding and our chat template stay.
- The schema is compiled once when the model loads, using the tokenizer and the model's padded vocabulary size. A new processor is built for every Instance's `generate` call.
- Releasing the model frees the compiled grammar along with it.

**Embedding of Descriptions**
- Every non-empty Description of every captioned Instance is one document for the Caption encoder. They are encoded in batches; the encoder interface is unchanged.

**Query**
- Each Instance with a Caption gets a Caption similarity: the maximum cosine over its Description embeddings. The Aspect of that maximum is recorded.
- The Caption ranking orders captioned Instances by that maximum. RRF, the two-channel normalisation, `rrf_k`, `query_top_k` and the tie-breaks are unchanged.
- `Match` gains `caption_aspect` (an Aspect or none). `caption_score` is the maximum cosine.

**Storage**
- In `instances.json`, `caption` is an object with name, synonyms, identity, appearance, material and use, or null.
- `caption_embeddings.npy` has shape (Instances, 4, D) in Aspect order. A zero row means no Description. It is written only when some Instance has a Caption.
- No old-format handling: storage starts empty with each runtime.

**API and frontend**
- The Instances endpoint still returns only `name`.
- Query Matches add `caption_aspect`: `"identity"`, `"appearance"`, `"material"`, `"use"` or null.
- The frontend types gain the field. The display is unchanged.

**Config**
- The `max_new_tokens` default for Captioning goes from 128 to 320. No other setting changes, and the schema is not configurable.

**Dependencies**
- Add `xgrammar==0.2.8` to the GPU requirements. It requires torch with no upper bound, so it leaves Colab's torch in place.
- Pin `transformers==5.22.2`.

**Docs**
- Amend ADR 0008 in place: per-Aspect Descriptions, separate embedding per Description, max per Instance for the Caption channel, and schema-forced replies. Record the rejected options from this decision: mean or per-Aspect RRF channels, keeping the one-document Caption, a Context Aspect, and few-shot prompting.
- GLOSSARY: add Aspect and Description, and redefine Caption. Avoid "sentence" and "facet".
- Update the pipeline and configuration docs and AGENTS.md: the xgrammar pin, one processor per generate, the new storage shape and the Match field.

## Testing Decisions

- Tests check behaviour through public interfaces, with expected values worked by hand, never recomputed with the code's own formula. Nothing that loads a model is mocked or unit-tested; that includes the VLM, constrained decoding and the Caption encoder.
- **Seam A** (HTTP API with `FakePipeline`, `FakeTextEncoder` and `FakeCaptionEncoder`) is the main seam:
  - A Query whose Caption vector matches one Instance's Appearance Description makes that Instance win, with `caption_aspect = "appearance"`, `caption_score` equal to the hand-worked cosine, and a hand-worked fused score.
  - A Query matching a different Aspect of another Instance flips the result.
  - An Instance with some empty Descriptions is scored over the ones it has.
  - An Instance without a Caption gets `caption_aspect` null and only its image rank.
  - A Scene with no Captions is normalised over one ranking.
  - Instances still return `name` only.
  - Job progress still includes Captioning.
  - Prior art: the existing RRF tests in the Query API test module.
- **Unit**:
  - The reply parser, with valid, fenced and truncated replies, a missing or empty Name, empty Descriptions, more than three Synonyms and extra keys.
  - Identity built from Name and Synonyms.
  - Choosing the best Description per Instance, ties included.
  - Storage round-trip of the new Caption and embedding shape.
  - The `max_new_tokens` default in the config tests.
  - Prior art: the existing Captioning and Query fusion unit tests.
- **Seam B** (GPU, Colab): extend the real-Pipeline smoke test. At least one Instance has a Caption with a non-empty Name and at least one non-empty Description, and a Query returns `query_top_k` Matches with `caption_aspect` set.

## Out of Scope

- A Context Aspect, or any spatial reasoning between Instances.
- An evaluation set or any tool that measures Query quality; the user judges results by hand.
- Changing the VLM or the Caption encoder model.
- Vietnamese or bilingual Captions, and Query translation.
- Exposing Descriptions or Synonyms through the API or the UI.
- Mean or per-Aspect fusion, or new RRF channels.
- Backfilling or reading Scenes stored in the old format.
- Tuning `rrf_k`, `query_top_k` or the Captioning defaults beyond `max_new_tokens`.

## Further Notes

- Not yet verified, to be checked in the Colab acceptance ticket:
  - the padded vocabulary size xgrammar needs for Qwen3-VL (probably the text config's vocabulary size);
  - whether xgrammar enforces `maxItems`;
  - the time each Instance's generation gains;
  - that torch is unchanged after installing xgrammar;
  - that the Pipeline runs on transformers 5.22.2.
- xgrammar's processor does not support beam search. The stage uses greedy decoding, which it supports.
- The test seams were agreed while designing this change: Seam A, Unit and Seam B, with no new seam.
