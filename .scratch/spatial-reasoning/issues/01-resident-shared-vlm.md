# 01: Keep the vision-language model resident and shared

**What to build:** The vision-language model that Captioning uses stays on the GPU after the Captioning stage ends, on success and on failure, so that a later Query can use it without a load. One instance is shared by the Pipeline and the API, built once where the other shared components are built. It loads on first use behind a lock, like SigLIP2 and the Caption encoder, and the lock is held around each single generation, not around the whole Captioning stage, so another thread can generate between two Instances. A job still produces the same Captions as before. DA3 and SAM2 still leave the GPU when their stage ends. This is a prefactor for the Query parser; nothing uses the model from a Query yet. See "Vision-language model" in `.scratch/spatial-reasoning/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] The Pipeline no longer releases the vision-language model when Captioning ends or fails.
- [ ] One instance is shared by the Pipeline and the API and is built once with the other shared components. Constructing it loads nothing.
- [x] Loading happens on first use and takes a lock; two threads that need the model at once cause one load.
- [x] The lock is held per generation. Captioning releases it between Instances.
- [x] Every generation still gets a new logits processor; the compiled grammar is still reused.
- [x] Captioning's output holds no GPU tensors, as before.
- [ ] The per-stage peak VRAM log still works for a job that starts with the model already resident.
- [x] Nothing that loads a model is mocked or unit-tested. The CPU suite and the Seam A tests pass unchanged.
- [x] ADR 0008 is amended in place: the vision-language model stays on the GPU and three models are resident between jobs. It names the cost: a job now starts with less free GPU memory for DA3 and SAM2.
- [x] AGENTS.md's GPU memory rule is updated to match. The pipeline docs are updated where they say the model leaves the GPU.

## Comments

- *Shared by the Pipeline and the API* is left unticked. The model is built once in `server.build_components`, next to SigLIP2 and the Caption encoder, and the Pipeline holds it; constructing it loads nothing (`tests/unit/test_server.py`). The API does not receive it yet, because nothing there uses it: ticket 07 wraps that same `captioner` in the real parser, and ticket 02 reshapes what `create_app` takes. Passing an unused handle now would only add a parameter both tickets change.
- *Peak VRAM log with the model already resident* is left unticked: it needs a GPU. By reading, `track_stage_memory` resets the peak counters at each stage and never assumed a release, so it still logs; every stage's peak now includes the resident models. Ticket 08 measures it on Colab.
- The lock and load behaviour is unit-tested on `ResidentModel` (`tests/unit/test_resident.py`) with a plain counting loader. `QwenVLCaptioner.generate` and `describe`, which use it, have not been run, because they load Qwen3-VL.
