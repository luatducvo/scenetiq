# 04: Relations that order candidates: nearest to, farthest from, largest, smallest, tallest, lowest

**What to build:** Queries such as "the chair nearest the window", "the plant farthest from the sofa", "the largest table", "the smallest box", "the tallest shelf" and "the lowest shelf" find the right Instance. Unlike the Relations of tickets 02 and 03, these score a candidate against the other candidates of the same object, so the solver must know the whole candidate set of the subject. The size and height Relations take no Anchor. See "Relations" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] The subject's candidates are ordered by the Relation's measure; the first scores 1, the last 0, the rest evenly spaced. A single candidate scores 1. Tied candidates share a score.
- [x] `nearest to` orders by the gap between the candidate's box and the Anchor's box, smallest first; `farthest from` is the reverse. The order is taken for the Anchor of the assignment being scored.
- [x] `largest` and `smallest` order by bounding-box volume.
- [x] `tallest` orders by vertical extent. `lowest` orders by the height of the box's bottom, with up as −Y.
- [x] A Match of a Query with only a size or height Relation has no Anchors and reports that Relations were applied.
- [x] These Relations combine with others in one Query by the same product of scores.
- [x] Unit: each ordering with one, two and several candidates, ties included, with hand-worked scores.
- [x] Seam A: "largest" returns the candidate with the biggest box although it does not rank first for the phrase.
- [x] Seam A: "nearest to" returns the closest candidate, with its Anchor.

## Comments

- Ties: candidates tie when their measure is exactly equal. The distinct values are ordered and spaced evenly, so a tie shares one score and takes one step (27, 8, 8, 1 largest first: 1, 1/2, 1/2, 0); tied first candidates all score 1.
- The order is taken over the subject's whole candidate list, Anchor included when it is one of them (two objects with one phrase). Excluding it would let an Anchor drawn from the subject's own candidates thin out the field and make a superlative trivially true; kept, it ranks first for nearest to (gap 0) and penalises that assignment. Covered at Seam A by `test_a_chair_standing_in_for_the_anchor_still_competes_for_nearest`.
