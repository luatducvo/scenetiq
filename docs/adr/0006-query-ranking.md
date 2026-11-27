# Query ranking: two standardized channels, Semantic Label filtering and a relative cut

A Query is ranked on two channels. The image channel compares its SigLIP2 text embedding with each Instance's SigLIP2 image embedding. The Caption channel compares its Qwen3-Embedding vector with the embedding of each Description of the Instance's Caption, written by Qwen3-VL-8B in the Captioning stage under an xgrammar-forced JSON schema, and keeps the best one (max, never mean) among the Aspects the parsed Query speaks about, or all Aspects when it names none. `run_query` in `query.py`:

1. **Fusion.** Each channel is standardized over the Scene's Instances; the fused value is their mean. An Instance without a Caption takes 0 on the Caption channel; a Scene with no Captions uses the image channel alone.
2. **Semantic Label filtering.** When the Scene has Captions, only Instances whose Name, Synonyms or relevant Descriptions share a word stem with the parsed phrase, or whose Caption similarity reaches `negative_query_threshold` (0.35), are candidates. No candidate means 0 Matches: a negative Query colours nothing.
3. **Cut.** A Match's score is its fused value over the best candidate's, so the best scores 1. A Query without a Relation keeps candidates scoring at least `query_score_ratio` (0.5), at most `top_k` of them; `top_k` defaults to `query_top_k` (5) and the caller may ask up to `query_max_top_k` (20).

A plain Query is ranked by the parser's English phrase and Aspects when it names one object, and by its original text when the parse failed or it lists unrelated objects. Instance crops are masked: `plan_views` gives every visible point a splat radius from the fusion voxel size, the Embedding stage adds a grey-filled masked crop (`embedding.masked_weight`), and the Captioning stage dims the background (`captioning.background_dim`), so a floor no longer takes the embedding of what stands on it.

## Considered Options

- Reciprocal Rank Fusion: keeps only ranks, so a real chair and the floor were one rank apart and there was no gap to cut at.
- An absolute similarity threshold on scores: similarities are uncalibrated per channel and per Scene. The one absolute value left only admits candidates that share no word with the phrase.
- One keyword document per Caption, or the mean of Description similarities: a Query about one Aspect is diluted by the others.
- An external VLM API, or Qwen3-VL-32B: a key and per-call cost and photos leaving the runtime; or too little VRAM left beside the resident models.

## Consequences

- Scores are relative to one Query and one Scene; with two or three Instances standardization is coarse.
- A Query waits for one Qwen3-VL generation (the parser), and the three Query models stay resident.
- Stored Scenes keep the embeddings of the models that built them.
- `query_score_ratio`, `negative_query_threshold`, `masked_weight`, `background_dim` and the splat radius are unvalidated guesses; `python -m scenetiq.evaluation` measures them against hand-labelled Queries.
