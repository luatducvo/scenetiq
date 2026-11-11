# 06: Fallbacks and notices

**What to build:** A Query never fails and never misleads because of its spatial part. When the Query asks for something the Relation set cannot express (left, right, in front of, behind, negation, counting, ordinals), when the parser's reply is unusable, or when no assignment is valid, the user still gets Matches and is told what was not taken into account. The viewer shows one line of notice. See "Routing", "Candidates" and "API and frontend" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] The parser protocol can report a parse failure, and the fake parser can be prepared to do so.
- [ ] The Query response adds the unsupported parts of the Query, as the text the parser quoted, and whether the parse failed.
- [ ] Unsupported parts with supported constraints: the supported Relations are applied and the unsupported parts are reported.
- [ ] Unsupported parts with no constraint: the Query runs as today with its original text and the unsupported parts are reported.
- [ ] Parse failure: the Query runs as today with its original text, and the response says the parse failed. The Query returns 200.
- [ ] No valid assignment (a Scene with fewer Instances than objects in the Query, or objects whose only candidates are the same Instance): the Query runs as today, and the response says Relations were not applied.
- [ ] The viewer shows one line of notice when parts of the Query were not taken into account or the parse failed, and nothing extra otherwise.
- [ ] Seam A: unsupported parts appear in the response, and the Matches equal those of the same parsed Query without them.
- [ ] Seam A: a parse failure returns the plain Matches and the failure flag.
- [ ] Seam A: a Scene with fewer Instances than objects returns the plain Matches and reports that Relations were not applied.
- [ ] The frontend lint, type check and build pass.
