# 02: Return top-K Matches ranked by RRF

**What to build:** A Query returns the top `query_top_k` Instances (default 1) ranked by Reciprocal Rank Fusion, with a score normalised to [0, 1], instead of every Instance above a threshold. For now only the image ranking exists, so the order matches today's, but the threshold and the Low-confidence Match are gone. Each Match carries `score`, `image_score` and `caption_score` (always null until ticket 03). The viewer drops the Low-confidence flag. See "Query", "Configuration" and "HTTP API" in `.scratch/instance-captioning/spec.md` and ADR 0008.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] RRF fusion is a pure function: an Instance's fused value is the sum over the rankings it appears in of `1 / (rrf_k + rank)`, ranks from 1; ties are broken by image similarity, then Instance id.
- [x] The reported score is the fused value divided by `channels / (rrf_k + 1)`, where `channels` is the number of rankings that exist (1 here).
- [x] `query_top_k` (default 1) and `rrf_k` (default 60) are settings, overridable through the config. `match_threshold` and its default constant are removed, and an override naming it is refused.
- [x] The text encoder interface loses its score calibration; the SigLIP2 and fake encoders follow.
- [x] The Query response is `{matches: [{instance_id, score, image_score, caption_score}]}` with no `low_confidence`. A Scene with fewer than K Instances returns all of them, an empty Scene none.
- [x] Seam A tests use hand-worked values for the ranking, the normalised score, the K cut and the empty Scene; unit tests cover the fusion function, ties included.
- [x] The frontend types and viewer no longer reference the Low-confidence flag; lint, typecheck and build pass.
- [x] The v1 spec's Query module and user stories 19–20, the AGENTS.md note on scores and the threshold, and the configuration docs describe top-K RRF.
