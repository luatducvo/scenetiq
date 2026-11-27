# 02: Colab acceptance of sentence-level SigLIP2 Queries

**What to build:** On a real Scene, confirm by hand that short Queries rank as well as before, that long Queries no longer fail, and how long sentences rank now that they are not wrapped in the template.

**Blocked by:** 01

**Status:** ready-for-human

- [ ] Short Queries ("chair", "red lamp") return the same top Match as before this change.
- [ ] A Query of about 500 characters returns Matches instead of HTTP 500, and the notebook shows a `siglip query truncated` line.
- [ ] Long descriptive sentences rank sensibly. Note any that do worse than the old wrapped form under `## Comments`.
