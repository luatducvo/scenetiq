# Queries

## Match scoring (ADR 0011, ADR 0012)

Matches are cut relative to the best one. Each channel's similarities are standardized over the Scene's Instances and averaged; a Match's score is that fused value divided by the best one, so the best scores 1. A Query without a Relation keeps the Instances scoring at least `query_score_ratio`.

- **Semantic Labels** (Name, Synonyms, Descriptions) first decide which Instances are candidates. `negative_query_threshold` is the one absolute similarity in ranking, and it only admits a candidate that shares no word stems with the phrase; it never cuts scored Matches.
- **A negative Query**, one whose object has no candidate in the Scene, returns 0 Matches and colours nothing in the viewer.
- **Without a `top_k` from the caller**, a Query returns every qualifying Match, up to the `query_max_top_k` ceiling. `api.py` still falls back to `query_top_k` here, which contradicts ADR 0012.
- **The Caption channel** scores an Instance by its best-matching Description (max, never a mean) among the Aspects the parsed object speaks about, falling back to all its Descriptions when it has none of them. An Instance without a Caption takes 0 on that channel, the captioned average. A Scene where no Instance got a Caption uses the image channel alone, with no Semantic Label filtering.
- **Stored Scenes keep the embeddings of the models that built them.** Changing `siglip.model_id` or `caption_embedding.model_id` makes older Scenes rank wrongly, without an error.
- **SigLIP2 Query text is truncated explicitly** with `truncation=True, max_length=64`: the checkpoint loads `SiglipProcessor`, which does not truncate, and a longer Query raises `ValueError`.

## Query parser (ADR 0009)

Every Query waits for one Qwen3-VL generation, the parser's, including a Query without a Relation; the first Query after a restart also pays the model's load, and during a job's Captioning stage a Query waits for the current Instance's generation.

- The parser gets the Query whole; the 64-token limit is SigLIP2's alone.
- Replies are schema-forced like [Captioning's](pipeline.md#captioning-adr-0008), and `read_parsed_query` stays as the second check.
- A parser raises `QueryParseError` only for an unusable reply, which the endpoint answers as a plain Query. A model or CUDA failure propagates and fails the request.

## Spatial Queries (ADR 0009)

Spatial Queries are solved on stored geometry. Route only on the parsed Query, never on keywords: a Query without a constraint returns exactly the plain Matches.

- A plain Query is ranked by its parsed object's English phrase and Aspects when it names one object or its Relations found no assignment, and by its original text when the parse failed or it lists several unrelated objects.
- Every Relation returns a score in [0, 1], never a yes/no, and measures distances against the Scene extent, never in absolute metres. Relations are not a third fused channel.
- The candidate margins (`spatial.caption_margin`, `spatial.image_margin`) only limit which shortlisted Instances may stand in for an object of a Query with a Relation, relative to the best of that object's shortlist, and always leave at least one candidate. Apply them only to Queries with a Relation, keep `query_score_ratio` off phrase rankings of a spatial Query, and keep both relative.
