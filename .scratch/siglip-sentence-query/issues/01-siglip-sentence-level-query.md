# 01: SigLIP2 Query text at sentence level

**What to build:** A short Query, of at most `siglip.prompt_max_words` words (4 by default), is still wrapped in `siglip.prompt` ("This is a photo of {}."). A longer Query reaches SigLIP2 as written. Every Query is lowercased and padded to 64 tokens. A Query longer than SigLIP2's 64-token text window is truncated for the image channel, and the number of tokens is logged through `scenetiq.query` without the Query text. Before this, such a Query raised `ValueError`: the checkpoint loads `SiglipProcessor`, which does not truncate. The Caption channel still sees the full Query, and the API keeps its 500-character limit.

**Blocked by:** None (can start immediately)

**Status:** done

Decided in a grilling session; there is no separate spec. Out of scope: rewriting the Query with an LLM, an evaluation set, a new ADR.

- [x] `siglip.prompt_max_words` (default 4) is in config and refused below 0. Setting it to 0 means the template is never used.
- [x] A pure function chooses the text SigLIP2 encodes, and unit tests with hand-worked inputs cover it.
- [x] SigLIP2 Query encoding passes `truncation=True, max_length=64` and logs the token count when it truncates.
- [x] `colab_logging.json` lets `scenetiq.query` INFO lines through to stdout.
- [x] A Seam B test checks that a Query of about 500 characters encodes to a unit vector.
- [x] `docs/configuration.md`, `docs/pipeline.md` and AGENTS.md are updated. AGENTS.md records that `SiglipProcessor` does not truncate.
