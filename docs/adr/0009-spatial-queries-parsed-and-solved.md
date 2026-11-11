# Spatial Queries parsed into objects and Relations, solved on stored geometry

A Query such as "the chair near the window" was ranked as one text against every Instance, so the words about the window only blurred the ranking, and the centroids and bounding boxes every Reconstruction stores (in metres, up is −Y by ADR 0007) were never used.

Every Query now goes through a parser that returns a parsed Query: up to four objects, each a short English phrase; the object the user wants; up to four constraints, each a Relation from a closed set, a subject and its Anchors; and the parts of the Query the set cannot express. The parser will be a schema-forced generation on the resident vision-language model (ADR 0008); until it exists, the real backend reads every Query as one without a Relation, and the tests use a fake.

A Query without a constraint is answered exactly as before, with its original text. Otherwise each object's phrase is ranked with the plain Query ranking, and its first M Instances (5 by default) are its candidates; the wanted object takes `query_top_k` when that is larger. A deterministic solver scores every assignment of one candidate to each object, with no Instance used twice, as the product of its constraints' scores, and orders the wanted object's candidates by their best assignment. Ties go to the earlier position in the phrase's ranking, then the lower Instance id. When no assignment is valid, the Query is answered as a plain one.

Each Relation is a pure function of axis-aligned boxes and centroids that returns a score in [0, 1]. Distances are gaps between boxes, not between centroids, and are measured against the Scene extent (the diagonal of the box around every Instance) through `exp(−d / (s · extent))` (`between` measures against the segment joining its two Anchors instead), so no Relation uses an absolute distance in metres. `near` is the falloff of the gap, 1 when the boxes touch or overlap. The ordering Relations (nearest to, farthest from, largest, smallest, tallest, lowest) score a candidate by its place among all the candidates of its object, by gap to the assignment's Anchor, box volume, vertical extent or the height of the box's bottom: the first distinct value scores 1, the last 0, evenly spaced, so tied candidates share a score and a tie takes one step. An Anchor that is also a candidate of the subject (two objects with one phrase) stays in that order, so standing a subject candidate in for the Anchor cannot thin out the candidates it is compared with. The other Relations follow the spec; a Relation without a scoring function fails loudly.

## Considered Options

- Geometry code written by a language model per Query (LaSP, ZSVG3D): flexible, but runs generated code, and a wrong program fails silently or not at all.
- The full QVR3D pipeline, where a vision-language model scores attributes and Relations per candidate and View: many generations per Query, and its own ablation credits most of its gain to choosing Views by Query, which is a later slice.
- Models trained on ScanNet-style data (Open3DSG, Chat-Scene, LLaVA-3D, 3D-VisTA): trained on sensor scans with fixed classes, not on Reconstructions from photos, and each adds a model to the GPU stack.
- A precomputed scene graph: fixes the Relations and their thresholds at job time, so older Scenes would need reprocessing for every change.
- A keyword test that decides whether a Query is spatial: misses paraphrases and other languages; every Query goes through the parser instead.
- Boolean Relations with distance thresholds: noisy boxes would remove the right candidate instead of lowering its score, and one threshold cannot fit a desk top and a hall.
- Relations as a third Reciprocal Rank Fusion channel: an Instance that fits the phrase well but breaks the Relation could still win.
- An absolute similarity threshold on candidates: similarities sit on different, uncalibrated scales per channel and per Scene, and a threshold could leave an object without candidates.

## Consequences

Matches of a spatial Query are ordered by the Relation score; the phrase ranking only picks the candidates and breaks ties, so a candidate that is not what the phrase names can win ("the largest chair" in a Scene with one chair could return a sofa). A candidate margin, added later, limits which shortlisted Instances may stand in for an object of a spatial Query. It is not the relevance threshold on Matches that ADR 0008 removed: a Query without a Relation still returns `query_top_k` Matches with no threshold.

A Match's score for a spatial Query is its assignment score, comparable only within one Query; its similarities are those of the wanted object's phrase, and it carries the Anchors of its assignment, which the viewer marks in their own colour. M, the falloff fraction and the Relation formulas are starting guesses, not validated. Bounding boxes are axis-aligned and yaw is arbitrary, so gaps between diagonal objects are underestimated. Every Query will wait for one generation of the parser, including Queries without a Relation.
