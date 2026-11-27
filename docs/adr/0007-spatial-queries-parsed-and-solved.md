# Spatial Queries parsed into objects and Relations, solved on stored geometry

Every Query goes through a parser: one text-only generation on the resident Qwen3-VL (the Captioning model, shared), forced to a JSON schema built from the domain's limits. It returns up to four objects, each a short English phrase with its Aspects, the wanted object, and up to four constraints, each a Relation from the closed `Relation` set with its subject and Anchors. The prompt lists every Relation, asks for English phrases whatever the Query's language, and has no example reply. `read_parsed_query` checks the reply again; `QueryParseError` answers the Query as a plain one.

A Query with constraints is solved in `spatial.py`: each object's phrase is ranked like a plain Query, cut to `spatial.candidates_per_object` (at least `top_k` for the wanted object) and narrowed by the candidate margins (`caption_margin`, `image_margin`) relative to the best of its shortlist. Every assignment of distinct Instances is scored by the product of its Relation scores, each in [0, 1], with distances relative to the Scene extent and `spatial.falloff`. Matches are the wanted object's Instances by best assignment, carrying their Anchors. No valid assignment falls back to a plain Query.

## Considered Options

- Keyword routing ("near", "on"): misses paraphrases and other languages; the parser reads every Query anyway.
- Relations as a third fused channel: a spatial fit is not a similarity and would be averaged away.
- Absolute distances in metres: Scene scale varies, so every Relation measures against the Scene extent.

## Consequences

- Only Relations independent of the viewer's position are covered; left, right, in front of, behind, negation, counting and ordinals are not.
- Vertical Relations rely on up being −Y (ADR 0005). Boxes are axis-aligned and yaw is arbitrary, so gaps between diagonal objects are underestimated.
- A schema-valid misreading (wrong Relation, swapped subject) is not detected; the Anchors in the API answer are its only trace, and the viewer does not show them.
- `candidates_per_object`, both margins, `falloff` and the Relation formulas are unvalidated guesses.
