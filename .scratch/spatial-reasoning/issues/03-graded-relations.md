# 03: Relations scored from geometry: above, below, on, under, far from, between

**What to build:** Queries such as "the lamp above the desk", "the box on top of the shelf", "the bag under the table", "the chair far from the door" and "the lamp between the sofa and the door" find the right Instance. A Query with several Relations, such as "the box on the shelf near the TV", honours all of them: an assignment's score is the product of its constraints' scores. `between` is the first Relation with two Anchors, and both are returned with the Match. See "Relations" and "Solver" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] Each Relation is a pure function of bounding boxes and centroids that returns a score between 0 and 1, never a yes/no, and uses no absolute distance in metres.
- [x] Up is −Y (ADR 0007): higher means a smaller y, the top of a box is its minimum y and the bottom its maximum y. The footprint of a box is its X–Z rectangle.
- [x] `far from` is one minus `near`.
- [x] `above` is zero unless the subject's centroid is higher than the Anchor's; otherwise it is the falloff of the horizontal gap between the two footprints. `below` is the mirror.
- [x] `on` is `above`, times the share of the smaller footprint that the two footprints share, times the falloff of the vertical distance between the subject's bottom and the Anchor's top. `under` is the mirror.
- [x] `between` is the falloff of the distance from the subject's centroid to the segment joining the two Anchors' centroids, scaled by that segment's length.
- [x] Unit: each Relation with boxes placed by hand (stacked, side by side, touching, overlapping, apart), including one case that would pass with up as +Y and must fail. Expected values are worked by hand, never recomputed with the code's formula.
- [x] Seam A: a Query with two Relations returns the candidate that satisfies both over one that satisfies only one, with the hand-worked product as its score.
- [x] Seam A: a `between` Query returns a Match with two Anchors.
- [x] AGENTS.md records that a Scene without Gravity alignment answers vertical Relations on DA3's Y axis and can be wrong, with no warning.
