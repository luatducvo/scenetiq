# 08: Colab acceptance of spatial Queries

**What to build:** On a real Colab runtime, confirm that spatial Queries work on real Scenes, measure what could not be measured without a GPU, and judge the unvalidated defaults by hand. There is no evaluation set; a numeric benchmark comes later.

**Blocked by:** 01, 02, 03, 04, 05, 06, 07

**Status:** ready-for-human

- [ ] `uv run pytest -m gpu` passes.
- [ ] The text-only generation works with the same xgrammar vocabulary size as Captioning: no error or mask-size mismatch on the first parse.
- [ ] The parse time per Query is noted, for a Query with and without a Relation.
- [ ] The time the first Query after a restart takes, model load included, is noted.
- [ ] The per-stage peak VRAM of a job that starts with the vision-language model already resident is noted and compared with a job on a fresh runtime. Any OOM is recorded with the View count.
- [ ] A Query sent while a job is in its Captioning stage returns after about one Instance's generation, not after the whole stage.
- [ ] Vietnamese Queries are parsed: the phrases come back in English and the Relation is the intended one. Note any Relation the model misreads.
- [ ] Hand-judged Queries on at least two real Scenes: at least one per Relation, each checked for the Match and for the Anchors the viewer marks.
- [ ] Hand-judged Queries that probe the candidate margin: a superlative in a Scene with one object of that kind, and a Relation whose phrase has two real candidates. Note whether a wrong kind of object wins or a real candidate is dropped, and the margins that fix it if the defaults do not.
- [ ] A Query with "left of" or "behind" returns Matches and the viewer shows the notice.
- [ ] A Scene that was left without Gravity alignment, if one is available, is tried with a vertical Relation and the result recorded.
- [ ] Any surprise, and any default that had to change, is recorded under `## Comments` in this file.
