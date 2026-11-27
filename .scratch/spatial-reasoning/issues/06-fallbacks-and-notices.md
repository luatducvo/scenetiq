# 06: Fallbacks and notices

**What to build:** A Query never fails and never misleads because of its spatial part. When the Query asks for something the Relation set cannot express (left, right, in front of, behind, negation, counting, ordinals), when the parser's reply is unusable, or when no assignment is valid, the user still gets Matches and is told what was not taken into account. The viewer shows one line of notice. See "Routing", "Candidates" and "API and frontend" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] The parser protocol can report a parse failure, and the fake parser can be prepared to do so.
- [x] The Query response adds the unsupported parts of the Query, as the text the parser quoted, and whether the parse failed.
- [x] Unsupported parts with supported constraints: the supported Relations are applied and the unsupported parts are reported.
- [x] Unsupported parts with no constraint: the Query runs as today with its original text and the unsupported parts are reported.
- [x] Parse failure: the Query runs as today with its original text, and the response says the parse failed. The Query returns 200.
- [x] No valid assignment (a Scene with fewer Instances than objects in the Query, or objects whose only candidates are the same Instance): the Query runs as today, and the response says Relations were not applied.
- [x] The viewer shows one line of notice when parts of the Query were not taken into account or the parse failed, and nothing extra otherwise.
- [x] Seam A: unsupported parts appear in the response, and the Matches equal those of the same parsed Query without them.
- [x] Seam A: a parse failure returns the plain Matches and the failure flag.
- [x] Seam A: a Scene with fewer Instances than objects returns the plain Matches and reports that Relations were not applied.
- [x] The frontend lint, type check and build pass.

## Comments

- The viewer's notice was checked only by the frontend lint, type check and build. Nobody has looked at it in a browser yet. It is one muted line above the Matches.
- A parser reports a parse failure by raising `scenetiq.domain.QueryParseError(reason)`. The endpoint catches only that exception: it logs the reason on the `scenetiq.query` logger, which `colab_logging.json` lets through, and answers the plain Query with `parse_failed: true`. Any other exception from a parser fails the request. The fake parser is prepared to fail by mapping a Query text to a `QueryParseError` instead of a `ParsedQuery`.
- The response tells "Relations could not be applied" apart from "the Query had no Relation" with the new field `relations_stated`: it is true when the parsed Query had at least one constraint. If it is true while `relations_applied` is false, no assignment was valid.
- `QUERY_LOGGER` moved from `scenetiq.pipeline.siglip` to `scenetiq.query`, so the API can use it without importing the pipeline package. `siglip` imports it from `scenetiq.query`.
