# 03: Colab acceptance of Multi-Aspect Captioning

**What to build:** On a real Colab runtime, confirm that the new GPU stack installs safely, that the real Pipeline produces per-Aspect Captions under the forced schema, and that Queries describing one Aspect find the right object. Check the parts that could not be verified without a GPU.

**Blocked by:** 01, 02

**Status:** ready-for-human

- [ ] Installing the GPU requirements leaves `torch.__version__` unchanged, and the Pipeline runs on transformers 5.22.2.
- [ ] `uv run pytest -m gpu` passes.
- [ ] The vocabulary size passed to xgrammar is right for Qwen3-VL: no error or mask-size mismatch on the first `generate`.
- [ ] xgrammar enforces the three-Synonym cap. Check whether any reply has more than three Synonyms.
- [ ] The time each Instance's Captioning takes is noted, compared with the previous run if available.
- [ ] One real Scene runs end to end. The log shows how many Instances got a Caption and the reason for any that did not.
- [ ] Hand-judged Queries: one per Aspect (a look, a material, a use, a name or synonym), each finding the intended object, with `caption_aspect` naming the expected Aspect.
- [ ] Any surprise is recorded under `## Comments` in this file.
