# Unverified defaults

Nothing here has been checked against ground truth or run on Colab yet. Treat these values as provisional, and remove a line once its acceptance issue is resolved.

| Item | Tracking issue |
|---|---|
| `query_top_k`, `query_score_ratio`, the Captioning defaults | `.scratch/query-relevance/issues/01-colab-acceptance.md` (measure with `python -m scenetiq.evaluation` against a running backend; the repo holds no labels) |
| `spatial.candidates_per_object`, both candidate margins, `spatial.falloff`, the Relation formulas | none |
| xgrammar vocabulary sized by the output layer | `.scratch/aspect-captioning/issues/03-colab-acceptance.md` |
| Text-only generation through Qwen3-VL's chat template, its xgrammar vocabulary size, the `aspects` the parser names | `.scratch/spatial-reasoning/issues/08-colab-acceptance.md`, `.scratch/query-relevance/issues/01-colab-acceptance.md` |
| `embedding.masked_weight`, `captioning.background_dim`, the Captioning prompt's wording about darkened parts | none |
| DA3 on numpy 2 without xformers (ADR 0005) | `pytest -m gpu` on Colab |
