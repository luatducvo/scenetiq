# 01: Share the per-Instance View planning

**What to build:** A prefactor with no visible change. The Embedding stage's choice of which Views see each Instance best, with the Instance's visible pixels in each, becomes one function the Pipeline calls once and passes to the Embedding stage, so that the Captioning stage can later reuse the same plan. Embeddings come out exactly as before. See "Stage" in `.scratch/instance-captioning/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] View planning is a pure function of the points, Instance ids, cameras, depth and Embedding options, returning per Instance the chosen Views (best first) and their visible pixels.
- [x] The Embedding stage takes the plan instead of computing it, and the real Pipeline computes it once before Embedding.
- [x] A unit test on a hand-built case shows the plan picks the same Views, in the same order, as the code did before the refactor, including an Instance no View sees well enough (empty plan).
- [x] Existing Embedding helper tests and the CPU suite pass unchanged.
- [x] No GPU import moves to module top level.
