# 04: Relations that order candidates: nearest to, farthest from, largest, smallest, tallest, lowest

**What to build:** Queries such as "the chair nearest the window", "the plant farthest from the sofa", "the largest table", "the smallest box", "the tallest shelf" and "the lowest shelf" find the right Instance. Unlike the Relations of tickets 02 and 03, these score a candidate against the other candidates of the same object, so the solver must know the whole candidate set of the subject. The size and height Relations take no Anchor. See "Relations" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] The subject's candidates are ordered by the Relation's measure; the first scores 1, the last 0, the rest evenly spaced. A single candidate scores 1. Tied candidates share a score.
- [ ] `nearest to` orders by the gap between the candidate's box and the Anchor's box, smallest first; `farthest from` is the reverse. The order is taken for the Anchor of the assignment being scored.
- [ ] `largest` and `smallest` order by bounding-box volume.
- [ ] `tallest` orders by vertical extent. `lowest` orders by the height of the box's bottom, with up as −Y.
- [ ] A Match of a Query with only a size or height Relation has no Anchors and reports that Relations were applied.
- [ ] These Relations combine with others in one Query by the same product of scores.
- [ ] Unit: each ordering with one, two and several candidates, ties included, with hand-worked scores.
- [ ] Seam A: "largest" returns the candidate with the biggest box although it does not rank first for the phrase.
- [ ] Seam A: "nearest to" returns the closest candidate, with its Anchor.
