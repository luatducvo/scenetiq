# 05: Candidate margin

**What to build:** An Instance that fits a phrase much worse than the best one can no longer stand in for that object, so that "the largest chair" in a Scene with one chair does not return a sofa. After the first M Instances of a phrase's fused ranking are shortlisted, a relative margin drops the poor ones. This applies only to the objects of a Query that has a Relation; a Query without one still returns `query_top_k` Matches with no threshold, as ADR 0008 says. See "Candidates" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] On each channel, a shortlisted Instance passes when its similarity is no more than the margin below the highest similarity on that channel among the shortlisted Instances. The channels are the cosine of the best-matching Description and the image similarity.
- [x] An Instance stays a candidate when it passes on at least one channel. An Instance without a Caption is judged on the image channel alone.
- [x] Every object keeps at least one candidate. There is no absolute similarity threshold anywhere.
- [x] Each channel has its own margin setting in `scenetiq.config`, validated: default 0.1 for the Caption channel and 0.05 for the image channel.
- [x] A spatial Query returns as many Matches as the wanted object has candidates, at most `query_top_k` and at least one.
- [x] Unit: an Instance exactly at the margin, just below it, passing on one channel only, and without a Caption.
- [x] Seam A: a shortlisted Instance more than the margin below the best on both channels does not become the Match although it would satisfy the Relation best.
- [x] Seam A: an Instance that passes on one channel only stays a candidate, and so does one without a Caption that passes on the image channel.
- [x] Seam A: an object whose shortlist is cut to one Instance still yields a Match.
- [x] Seam A: with `query_top_k` larger than M, a Query without a Relation still returns `query_top_k` Matches.
- [x] ADR 0009 states that the candidate margin limits which Instances may stand in for an object of a spatial Query and is not the relevance threshold on Matches that ADR 0008 removed. It lists an absolute similarity threshold as a rejected option.
- [x] AGENTS.md records that both margins are not validated and that the margin is not a threshold on Matches. The configuration docs list the two settings.

## Comments

- "As many Matches as the wanted object has candidates" holds when every candidate of the wanted object has a valid assignment. A candidate whose only partners would reuse it (for example, it is also the sole candidate of an Anchor) has no assignment and yields no Match; when no candidate has one, the Query falls back to the plain path (ticket 06 reports that). The margin itself always leaves at least one candidate per object.
- Margins are compared exactly (`similarity >= best - margin`), with no tolerance. Similarities come from float32 embeddings, so an Instance that looks exactly at the margin in printed values may fall on either side by about 1e-8.
