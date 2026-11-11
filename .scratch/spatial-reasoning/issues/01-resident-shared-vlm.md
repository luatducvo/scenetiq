# 01: Keep the vision-language model resident and shared

**What to build:** The vision-language model that Captioning uses stays on the GPU after the Captioning stage ends, on success and on failure, so that a later Query can use it without a load. One instance is shared by the Pipeline and the API, built once where the other shared components are built. It loads on first use behind a lock, like SigLIP2 and the Caption encoder, and the lock is held around each single generation, not around the whole Captioning stage, so another thread can generate between two Instances. A job still produces the same Captions as before. DA3 and SAM2 still leave the GPU when their stage ends. This is a prefactor for the Query parser; nothing uses the model from a Query yet. See "Vision-language model" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The Pipeline no longer releases the vision-language model when Captioning ends or fails.
- [ ] One instance is shared by the Pipeline and the API and is built once with the other shared components. Constructing it loads nothing.
- [ ] Loading happens on first use and takes a lock; two threads that need the model at once cause one load.
- [ ] The lock is held per generation. Captioning releases it between Instances.
- [ ] Every generation still gets a new logits processor; the compiled grammar is still reused.
- [ ] Captioning's output holds no GPU tensors, as before.
- [ ] The per-stage peak VRAM log still works for a job that starts with the model already resident.
- [ ] Nothing that loads a model is mocked or unit-tested. The CPU suite and the Seam A tests pass unchanged.
- [ ] ADR 0008 is amended in place: the vision-language model stays on the GPU and three models are resident between jobs. It names the cost: a job now starts with less free GPU memory for DA3 and SAM2.
- [ ] AGENTS.md's GPU memory rule is updated to match. The pipeline docs are updated where they say the model leaves the GPU.
