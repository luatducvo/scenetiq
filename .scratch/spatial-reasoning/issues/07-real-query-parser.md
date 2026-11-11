# 07: The real Query parser on the vision-language model

**What to build:** On the real backend, a Query typed in Vietnamese or English is read by the resident vision-language model and split into objects, the wanted object, constraints and unsupported parts, so that spatial Queries work outside the tests. The reply is forced to the fixed JSON schema in the spec and checked a second time; a reply that fails the check is a parse failure and falls back as ticket 06 defines. See "Query parser" and "Vision-language model" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** 01, 02, 06

**Status:** ready-for-agent

- [ ] The parser is a text-only generation on the shared vision-language model, with its own compiled grammar, a new logits processor per generation, and the per-generation lock of ticket 01.
- [ ] The schema is one constant, not a setting, and is the one in the spec. Its Relation values come from the domain's `Relation` set, so the two cannot drift apart.
- [ ] The prompt is text only. It lists every Relation with a one-line meaning, asks for short English phrases whatever the Query's language, and tells the model to put what it cannot express into the unsupported list. It has no few-shot example.
- [ ] The Query text reaches the parser whole, with no truncation.
- [ ] The second check rejects a reply that is cut off or unreadable, repeats an object id, names a target or an object that is not listed, gives a Relation the wrong number of Anchors, or uses the subject as its own Anchor. The reason is logged through a logger the Colab logging configuration lets through.
- [ ] The parser's token limit is its own setting in `scenetiq.config`, default 1024, validated, and separate from Captioning's.
- [ ] The real backend builds the real parser with the other shared components; the local backend keeps the fake parser.
- [ ] Unit: the second check, with a valid reply and each kind of rejected reply. The check is a pure function; nothing that loads the model is mocked or unit-tested.
- [ ] Seam B: the real-Pipeline smoke test is extended. A Query with a Relation returns between one and `query_top_k` Matches with at least one Anchor, and a Query without one returns no Anchors. It asserts shape, not correctness.
- [ ] AGENTS.md records that every Query now waits for one generation, and that the first Query after a restart also pays the model load. The pipeline and configuration docs describe the parser and its setting.
