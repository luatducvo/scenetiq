# Spec: Spatial Reasoning Queries, first slice

Status: ready-for-agent

Vocabulary follows `GLOSSARY.md`; this work adds **Relation** and **Anchor** and brings **Spatial reasoning** into scope. Respect the ADRs in `docs/adr/`, in particular 0007 (up is −Y) and 0008 (Captions, RRF top-K, no threshold). This work adds ADR 0009 and amends ADR 0008 where it says the vision-language model leaves the GPU after Captioning. Builds on `.scratch/aspect-captioning/spec.md`, which describes Captioning and the Query as they stand.

## Problem Statement

A user types a Query to find an object in a Scene. Open-vocabulary Queries work: "red chair" or "something to sit on" finds the Instance. A Query that picks an object by where it is does not: "the box on top of the shelf", "the chair nearest the window", "the lamp between the sofa and the door", "the largest table". Each Instance is ranked on its own against the whole Query text, so the words about other objects only blur the ranking, and the positions the Reconstruction already stores (centroid and bounding box, in metres, with a known up direction) are never used.

## Solution

Every Query is first read by a language model that splits it into the objects it mentions and the Relations between them. One object is the one the user wants; the others are Anchors. Each object is looked up with the existing open-vocabulary ranking, and a deterministic solver then picks the combination of Instances that best satisfies the Relations, using only the stored centroids and bounding boxes.

The Match is the wanted object from the best combination. The viewer also marks the Anchors that combination used, in a different colour, so the user can see why the Match was chosen and notice when an Anchor was misread.

This first slice covers Relations that do not depend on where the user stands: above, below, on, under, near, far from, between, nearest to, farthest from, largest, smallest, tallest and lowest. When a Query asks for something else (left, right, in front of, behind, negation, counting, ordinals), or the language model's reply cannot be read, the Query is still answered the way it is today and the user is told which part was not taken into account.

A Query without a Relation behaves exactly as before.

## User Stories

1. As a user, I want "the box on top of the shelf" to find the box that rests on the shelf, so that I can pick an object by what it sits on.
2. As a user, I want "the bag under the table" to find the bag below the table, so that I can pick an object by what is over it.
3. As a user, I want "the lamp above the desk" to find the lamp higher than the desk and over it, so that vertical Relations work without contact.
4. As a user, I want "the chair near the window" to prefer chairs close to the window, so that I can pick an object by its neighbour.
5. As a user, I want "the chair far from the door" to prefer chairs away from the door, so that the opposite of near works.
6. As a user, I want "the chair nearest the window" to find the closest of the candidate chairs, so that superlatives against an Anchor work.
7. As a user, I want "the plant farthest from the sofa" to find the most distant candidate, so that the opposite superlative works.
8. As a user, I want "the lamp between the sofa and the door" to find the lamp lying between the two, so that a Relation with two Anchors works.
9. As a user, I want "the largest table", "the smallest box", "the tallest shelf" and "the lowest shelf" to compare the candidates of one object, so that size and height superlatives work without an Anchor.
10. As a user, I want a Query with several Relations, such as "the box on the shelf near the TV", to honour all of them together, so that I can narrow down step by step.
11. As a user, I want a Query without any Relation to return exactly the Matches it returns today, so that the change costs me nothing for plain Queries.
12. As a user, I want to write the Query in Vietnamese or English, so that I can describe a place in my own words.
13. As a user, I want the Anchors of a Match marked in the viewer in a different colour from the Match, so that I can see what the Match was measured against.
14. As a user, I want to notice when an Anchor was misread, so that I can rephrase the Query.
15. As a user, I want a wrongly recognised Anchor not to ruin the Query when a better candidate exists, so that one bad top result does not decide everything.
16. As a user, I want to be told when part of my Query was not taken into account, such as "left of", so that I do not trust a Match for something it did not check.
17. As a user, I want a Query to return Matches even when the language model's reply is unusable, so that the feature never makes a Query fail.
18. As a user, I want to be told when the spatial part of my Query was dropped because the reply was unusable, so that I can tell it apart from a plain Query.
19. As a user, I want a long Query to be read in full, so that the Relation at the end of a sentence is not cut off.
20. As a user, I want to keep seeing the Name next to each Match and on its box, so that the display stays familiar.
21. As a user, I want a Query sent while a job is in its Captioning stage to wait only briefly, so that I can keep using older Scenes during a job.
22. As a user, I want the first Query after a restart to work even though it is slower, so that I never have to warm anything up by hand.
23. As a user, I want Scenes that were stored before this change to answer spatial Queries without being reprocessed, so that my existing Scenes gain the feature.
24. As a user judging results by hand, I want each Match of a spatial Query to carry how well its combination satisfied the Relations, so that I can compare candidates.
25. As a user judging results by hand, I want the image and Caption similarities of the Match beside that score, so that I can tell a Relation problem from a recognition problem.
26. As a user, I want two objects of the same kind in one Query ("the chair next to another chair") never to resolve to the same Instance, so that an object is not its own Anchor.
27. As a user, I want objects that touch, such as a chair pushed against a table, to count as near even when both are large, so that near follows the gap I see.
28. As a user, I want vertical Relations to follow physical up, so that "above" means what it means in the room.
29. As a developer, I want the language model's reply forced to a fixed JSON schema, so that a reply can be lost only by being cut off.
30. As a developer, I want the reply checked a second time (object ids exist, the wanted object is one of them, each Relation has the right number of Anchors), so that a schema-valid but inconsistent reply falls back instead of crashing.
31. As a developer, I want the Relations to be a closed, named set defined once in the domain, so that the schema, the prompt and the solver cannot drift apart.
32. As a developer, I want each Relation to be one pure function of bounding boxes and centroids, so that it is tested on CPU with hand-worked values.
33. As a developer, I want every Relation to return a score between 0 and 1 and never a yes/no, so that noisy boxes degrade a result instead of removing it.
34. As a developer, I want no Relation to use an absolute distance in metres, so that the same code works for a desk top and a hall.
35. As a developer, I want the solver to be a pure function from candidates and Relations to ranked combinations, so that it is tested without any model.
36. As a developer, I want the candidates of each object to come from the existing Query ranking, so that open-vocabulary matching stays in one place.
37. As a developer, I want the language model to give each object as a short English phrase, so that it is compared with English Descriptions.
38. As a developer, I want the parser behind a protocol with a fake, so that the Query path is tested through the HTTP API without a model.
39. As a developer, I want the parser to reuse the vision-language model that Captioning already loads, so that no new model is added to the GPU stack.
40. As a developer, I want that model to stay on the GPU after Captioning and to load on first use after a restart, so that a Query does not pay a load each time.
41. As a developer, I want its lock taken around each single generation, so that a Query waits for at most one Instance's Caption.
42. As a developer, I want a new logits processor for every generation, with one compiled grammar per schema, so that the existing constrained-decoding rule is kept.
43. As a developer, I want the parser's token limit to be its own setting, set generously, so that a cut-off reply is rare and Captioning's limit is untouched.
44. As a developer, I want the number of candidates per object, the candidate margins and the falloff constants in `scenetiq.config`, so that every default stays in one place.
45. As a developer, I want the Query endpoint to keep its path and its existing fields, so that an older client still works.
46. As a developer, I want ADR 0009 written, ADR 0008 amended, and GLOSSARY, the pipeline and configuration docs and AGENTS.md updated in the same change, so that the docs match the code.
47. As a maintainer, I want AGENTS.md to record that the candidate count, the candidate margins and the Relation formulas are not validated, that the candidate margin is not a threshold on Matches, that the vision-language model now stays resident, and that a Scene without Gravity alignment answers vertical Relations wrongly, so that a future change does not assume otherwise.
48. As a maintainer, I want a Colab acceptance ticket that measures the parse time per Query, the first-Query load time, the peak VRAM of a job that starts with the vision-language model already resident, and whether the parser reads Vietnamese Queries, so that the unverified parts get checked.
49. As a user, I want an Instance that fits a phrase much worse than the best one not to stand in for that object, so that "the largest chair" in a Scene with one chair does not return a sofa.
50. As a user, I want every object of my Query to keep at least its best candidate, so that the candidate margin never leaves a spatial Query without a Match.
51. As a user, I want an Instance without a Caption to still be able to stand in for an object when its image fits, so that a Captioning failure does not hide it from spatial Queries.

## Implementation Decisions

**Domain**
- A new closed `Relation` enumeration: above, below, on, under, near, far from, between, nearest to, farthest from, largest, smallest, tallest, lowest. Each has a fixed number of Anchors: none for largest, smallest, tallest and lowest; two for between; one for the rest.
- A parsed Query is: up to four objects, each with an id and a short English phrase; the id of the wanted object; up to four constraints, each a Relation, a subject object and its Anchor objects; and a list of the parts of the Query that the Relation set cannot express, quoted from the Query.
- A Match gains the Anchors of its combination: for each, the Instance and the phrase it stood for. It is empty for a Query without a Relation.

**Query parser**
- Every Query goes through the parser. There is no keyword test that decides whether a Query is spatial.
- The parser receives the Query text whole, with no truncation. The 64-token limit applies only to the SigLIP2 text encoder, as today.
- The prompt is text only. It lists the Relations with a one-line meaning each, asks for English phrases whatever the Query's language, and tells the model to put anything it cannot express (left, right, in front of, behind, negation, counting, ordinals) into the unsupported list instead of forcing it into a Relation. It has no few-shot example, for the reason ADR 0008 gives.
- The reply is forced to this schema (constant, no setting):

  ```json
  {
    "type": "object",
    "properties": {
      "objects": {
        "type": "array", "minItems": 1, "maxItems": 4,
        "items": {
          "type": "object",
          "properties": {
            "id": {"enum": ["a", "b", "c", "d"]},
            "phrase": {"type": "string", "minLength": 1, "maxLength": 120}
          },
          "required": ["id", "phrase"], "additionalProperties": false
        }
      },
      "target": {"enum": ["a", "b", "c", "d"]},
      "constraints": {
        "type": "array", "maxItems": 4,
        "items": {
          "type": "object",
          "properties": {
            "relation": {"enum": ["above", "below", "on", "under", "near", "far_from", "between",
                                  "nearest_to", "farthest_from", "largest", "smallest", "tallest", "lowest"]},
            "subject": {"enum": ["a", "b", "c", "d"]},
            "anchors": {"type": "array", "maxItems": 2, "items": {"enum": ["a", "b", "c", "d"]}}
          },
          "required": ["relation", "subject", "anchors"], "additionalProperties": false
        }
      },
      "unsupported": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 80}}
    },
    "required": ["objects", "target", "constraints", "unsupported"],
    "additionalProperties": false
  }
  ```

- A second check rejects a reply that is cut off or unreadable, repeats an object id, names a target or an object that is not listed, gives a Relation the wrong number of Anchors, or uses the subject as its own Anchor. A rejected reply is a parse failure. It is logged with its reason.
- The parser has its own options in the configuration: the token limit for a reply, default 1024. The schema ends the reply when the JSON closes, so a generous limit costs nothing on a normal reply.

**Routing**
- No constraints in the parsed Query: the Query runs as today, with the original text, and returns the same Matches. The unsupported list is still reported.
- Parse failure: the same, and the response says the parse failed.
- One or more constraints: the spatial path below. Unsupported parts are reported and otherwise ignored.

**Candidates**
- Each object is ranked with the existing Query ranking, using its English phrase for both the image and the Caption channel. The first M Instances of that fused ranking are shortlisted. M is a setting, default 5. The wanted object shortlists `query_top_k` Instances when that is larger than M.
- A candidate margin then drops shortlisted Instances that fit the phrase much worse than the best one. On each channel, an Instance passes when its similarity is no more than a margin below the highest similarity on that channel among the shortlisted Instances. The channels are the cosine of the best-matching Description and the image similarity. An Instance stays a candidate when it passes on at least one channel; an Instance without a Caption is judged on the image channel alone.
- The margin is relative, so the best shortlisted Instance of each channel always stays and every object has at least one candidate. There is no absolute similarity threshold.
- Each channel has its own margin setting, because the two similarities are on different scales: default 0.1 for the Caption channel and 0.05 for the image channel. Neither is validated.
- The margin applies only to the candidates of a Query that has a Relation. A Query without one still returns `query_top_k` Matches with no threshold, as ADR 0008 says. A spatial Query can therefore return fewer than `query_top_k` Matches, but always at least one.
- When no assignment is valid (a Scene with fewer Instances than objects in the Query, or objects whose only candidates are the same Instance), the Query runs as today and the response says the Relations were not applied.

**Relations**
- All are pure functions of axis-aligned bounding boxes and centroids, in the Reconstruction's frame. Up is −Y: higher means a smaller y, the top of a box is its minimum y and the bottom its maximum y. The horizontal footprint of a box is its X–Z rectangle.
- Distances are divided by the Scene extent, the diagonal of the box that encloses every Instance's bounding box, and passed through an exponential falloff `exp(−d / (s · extent))`. The fraction `s` is a setting, default 0.1.
- Starting definitions (not validated):
  - near: the falloff of the gap between the two boxes (zero when they touch or overlap). far from: one minus near.
  - above: zero unless the subject's centroid is higher than the Anchor's; otherwise the falloff of the horizontal gap between the two footprints. below: the mirror.
  - on: above, times the share of the smaller footprint that the two footprints share, times the falloff of the vertical distance between the subject's bottom and the Anchor's top. under: the mirror.
  - between: the falloff of the distance from the subject's centroid to the segment joining the two Anchors' centroids, scaled by that segment's length instead of the Scene extent.
  - nearest to, farthest from: the subject's candidates are ordered by their gap to the Anchor; the first scores 1, the last 0, evenly spaced; a single candidate scores 1.
  - largest, smallest: the same ordering by bounding-box volume. tallest: by vertical extent. lowest: by the height of the box's bottom.
- These read the gap between boxes, not between centroids, because two large objects that touch have distant centroids.

**Solver**
- It scores every assignment of one candidate to each object, with no Instance used twice. With four objects and five candidates this is at most 625 assignments, so it enumerates them.
- An assignment's score is the product of its constraints' scores.
- Each candidate of the wanted object keeps its best assignment. Matches are those candidates ordered by that score, ties broken by the candidate's position in the fused ranking of its phrase, then by Instance id. The first `query_top_k` are returned.
- For a spatial Query a Match's `score` is the assignment score. Its image and Caption similarities and Aspect are those of the wanted object's phrase. Like the fused score, it is comparable only within one Query.
- Relations are not a third Reciprocal Rank Fusion channel: an Instance that fits the phrase well and breaks the Relation could then still win.

**Vision-language model**
- The model Captioning uses also serves the parser, through a text-only generation with its own compiled grammar.
- It no longer leaves the GPU when Captioning ends, on success or on failure. It loads on first use, by a job or by a Query, behind a lock, like SigLIP2 and the Caption encoder. One instance is shared by the Pipeline and the API and is built once where the other shared components are built.
- The lock is held around each single generation, so a Query runs between two Instances of a Captioning stage.
- DA3 and SAM2 still leave the GPU when their stage ends.

**API and frontend**
- The Query endpoint keeps its path and request. Its response adds: whether Relations were applied; the unsupported parts of the Query as text; and whether the parse failed. Each Match adds its Anchors (Instance id and phrase).
- The viewer highlights Anchors in a colour different from the Match and shows their phrase. It shows one line of notice when parts of the Query were not taken into account or the parse failed.

**Fakes and local backend**
- A fake parser returns a prepared parsed Query for a known Query text and "no Relation" for anything else, so the local backend with the fake Pipeline behaves as it does today unless a test prepares it.

**Docs**
- ADR 0009: Queries are parsed into objects and Relations by a schema-forced language model and solved deterministically on stored geometry. Record the rejected options: language-model-generated geometry code (LaSP, ZSVG3D), the full QVR3D pipeline (attribute and relation scoring by a vision-language model per candidate and View), models trained on ScanNet-style data (Open3DSG, Chat-Scene, LLaVA-3D, 3D-VisTA), a precomputed scene graph, a keyword test for routing, boolean Relations with distance thresholds, Relations as a Reciprocal Rank Fusion channel, and an absolute similarity threshold on candidates. State that the candidate margin limits which Instances may stand in for an object of a spatial Query and is not the relevance threshold on Matches that ADR 0008 removed.
- Amend ADR 0008: the vision-language model stays on the GPU; three models are now resident between jobs.
- GLOSSARY: add Relation (a spatial relationship between Instances that a Query states) and Anchor (an Instance a Query uses as a reference to single out the Match). Spatial reasoning loses "Out of scope for v1" and names what the first slice covers.
- Update the pipeline and configuration docs and AGENTS.md (see user story 47).

## Testing Decisions

- Tests check behaviour through public interfaces, with expected values worked by hand, never recomputed with the code's own formula. Nothing that loads a model is mocked or unit-tested; that includes the parser's model and constrained decoding.
- **Seam A** (HTTP API with the fake Pipeline, the fake encoders and the new fake parser) is the main seam:
  - A Query the fake parser reports as having no Relation returns the same response body as before, plus the new fields in their empty state.
  - A Query with one Relation returns the candidate that satisfies it, although another candidate ranks first for the phrase, with the hand-worked score and the right Anchor.
  - A wrong first candidate for the Anchor does not win when a later candidate gives a better combination.
  - A Query with two Relations returns the candidate that satisfies both over one that satisfies only one.
  - A between Query returns two Anchors.
  - Two objects with the same phrase never resolve to one Instance.
  - Unsupported parts appear in the response and the Matches equal those of the same Query without them.
  - A parse failure returns the plain Matches and the failure flag.
  - A Scene with fewer Instances than objects returns the plain Matches and reports that Relations were not applied.
  - A shortlisted Instance more than the margin below the best on both channels is not a candidate: it does not become the Match although it would satisfy the Relation best.
  - An Instance that passes on one channel only stays a candidate, and so does an Instance without a Caption that passes on the image channel.
  - An object whose shortlist is cut to one Instance still yields a Match.
  - With `query_top_k` larger than M, a spatial Query returns as many Matches as the wanted object has candidates, at most `query_top_k`; a Query without a Relation still returns `query_top_k`.
  - Prior art: the existing RRF tests in the Query API test module.
- **Unit**:
  - Each Relation, with boxes placed by hand: touching, overlapping, apart, stacked, side by side; including that above uses a smaller y.
  - The ordering Relations with one, two and several candidates, ties included.
  - The solver: product of scores, no Instance used twice, tie-breaks, no valid assignment.
  - The candidate margin: an Instance exactly at the margin, just below it, passing on one channel, and without a Caption.
  - The reply's second check: a cut-off reply, a repeated id, a missing target, a wrong number of Anchors, a subject as its own Anchor.
  - The new configuration defaults and their validation.
  - Prior art: the existing Query fusion, geometry and Captioning parser unit tests.
- **Seam B** (GPU, Colab): extend the real-Pipeline smoke test. A Query with a Relation returns between one and `query_top_k` Matches with at least one Anchor, and a Query without one returns no Anchors. It asserts shape, not correctness.
- The frontend is covered by its lint, type check and build, as today.

## Out of Scope

- Left, right, in front of and behind, and the choice of the viewpoint they are measured from. That is the second slice.
- A vision-language model that looks at Views to judge a Relation or an attribute (the QVR3D-style slice), including choosing Views by Query.
- Negation, counting and ordinals.
- Answers in text, such as a count or a distance.
- An evaluation set or any tool that measures Query quality. The user judges results by hand; a numeric benchmark comes later.
- Tuning M, the candidate margins, the falloff fraction, `rrf_k` or `query_top_k`.
- A relevance threshold on Matches, a threshold on Queries without a Relation, and any absolute similarity threshold. ADR 0008 stands.
- Storing whether a Scene was Gravity-aligned, and warning the user when it was not.
- Oriented bounding boxes or yaw alignment.
- A separate language model for parsing, or an external API.
- Caching parsed Queries.

## Further Notes

- **Known weakness, reduced but not removed.** Matches are ordered by the Relation score and the phrase ranking only breaks ties, so a candidate that is not what the phrase names can win: "the largest chair" in a Scene with one chair could return a sofa. The candidate margin is there to drop such Instances. Its defaults are guesses: too wide and the sofa stays, too narrow and the second real chair is dropped. The image channel is the likelier leak, because an Instance passes on either channel and image similarities sit in a narrow range. Check this first when judging by hand, and adjust the two margins before anything else.
- A Scene with nothing that fits a phrase still gives that object a candidate, the best of a bad shortlist. The margin is relative and cannot tell that the best is poor.
- A Scene whose Views gave no consensus on up stays in DA3's frame. Vertical Relations are computed on its Y axis anyway and can be wrong, with no warning. Only the job log says a Scene was left unaligned.
- Bounding boxes are axis-aligned and yaw is arbitrary, so an object standing diagonally has a larger box. Gaps are then underestimated and footprints overestimated.
- Every Query now waits for one language-model generation, including Queries without a Relation.
- A job that starts with the vision-language model resident has less free GPU memory for DA3 and SAM2 than before. Nothing checks the View count against GPU memory.
- Not yet verified, to be checked in the Colab acceptance ticket: the parse time per Query; the load time paid by the first Query after a restart; the peak VRAM of a job with the model resident; that the model reads Vietnamese Queries and writes English phrases; that the text-only generation works with the same xgrammar vocabulary size as Captioning.
- Research behind the design: CSVG (BMVC 2025, arXiv 2411.14594) for objects as variables and Relations as constraints solved together; QVR3D (arXiv 2611.01084) for the later vision slice, where its own ablation shows most of the gain comes from choosing Views by Query and little from its probabilistic scoring. Benchmark figures in those papers are for ScanNet sensor scans and say nothing about Reconstructions made from photos.
- The earlier Captioning specs list Query translation as out of scope. The parser now writes English phrases for the objects of a Query that has a Relation; a Query without one is still passed on in its original language.
- The test seams were agreed while designing this change: Seam A, Unit and Seam B, with no new seam.
