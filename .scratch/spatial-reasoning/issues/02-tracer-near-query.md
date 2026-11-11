# 02: Tracer bullet: a "near" Query end to end with a fake parser

**What to build:** A Query such as "the chair near the window" returns the chair that is close to the window, and the viewer marks the window as its Anchor in a different colour. This ticket cuts through every layer with one Relation, `near`, and a fake parser; the real parser comes in ticket 07.

Every Query goes through a parser, behind a protocol, that returns a parsed Query: objects with English phrases, the wanted object, constraints and unsupported parts. A fake parser returns a prepared parsed Query for a known Query text and "no Relation" for anything else. A Query with no constraint runs exactly as today with its original text. A Query with a constraint takes the first M Instances of the existing fused ranking of each object's phrase as candidates (no candidate margin yet), scores every assignment with no Instance used twice, and returns the wanted object's candidates ordered by assignment score, ties broken by position in the phrase's ranking and then Instance id. The Match's `score` is the assignment score; its image and Caption similarities and Aspect are those of the wanted object's phrase.

See "Domain", "Routing", "Candidates", "Relations", "Solver", "API and frontend" and "Fakes and local backend" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The domain has the closed `Relation` set with the number of Anchors each takes, and the parsed Query shape. All thirteen Relations are named now; only `near` has a scoring function.
- [ ] A parser protocol and a fake parser exist. The local backend with the fake Pipeline uses the fake parser and behaves as today for any Query that was not prepared.
- [ ] `near` is a pure function of two bounding boxes and the Scene extent: the exponential falloff of the gap between the boxes, zero gap when they touch or overlap. The falloff fraction is a setting, default 0.1.
- [ ] The Scene extent is the diagonal of the box enclosing every Instance's bounding box.
- [ ] The solver is a pure function from candidates and constraints to ranked assignments.
- [ ] M is a setting in `scenetiq.config`, default 5, validated. The wanted object shortlists `query_top_k` Instances when that is larger.
- [ ] The Query response adds whether Relations were applied, and each Match adds its Anchors (Instance id and phrase). Existing fields and the endpoint path are unchanged.
- [ ] The viewer highlights a Match's Anchors in a colour different from the Match and shows each Anchor's phrase.
- [ ] Seam A: a Query the fake parser reports as having no Relation returns the same Matches as before, with the new fields in their empty state.
- [ ] Seam A: a `near` Query returns the candidate close to the Anchor although another candidate ranks first for the phrase, with a hand-worked score and the right Anchor.
- [ ] Seam A: a wrong first candidate for the Anchor does not win when a later candidate gives a better assignment.
- [ ] Seam A: two objects with the same phrase never resolve to one Instance.
- [ ] Unit: `near` with boxes that touch, overlap and stand apart; the solver's tie-breaks and its rule that no Instance is used twice. Expected values are worked by hand.
- [ ] ADR 0009 is written, with the rejected options the spec lists.
- [ ] GLOSSARY gains Relation and Anchor, and Spatial reasoning loses "Out of scope for v1" and names what the first slice covers.
- [ ] AGENTS.md records that M and the Relation formulas are not validated. The configuration docs list the new settings.
- [ ] The frontend lint, type check and build pass.
