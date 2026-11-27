# 06: Caption Instances with Qwen3-VL

**What to build:** Every new real Scene gets Captions. After the Embedding stage drops unseen Instances, the Captioning stage shows `Qwen/Qwen3-VL-8B-Instruct` each surviving Instance in up to three of its best Views, taken from the shared plan, each as a context crop with the Instance boxed. It parses the reply into a Caption, which is then embedded. An unusable reply leaves that Instance without a Caption and the job continues; a model load failure or out-of-memory fails the job. One log line per job says how many Instances got a Caption. See "Stage", "Captioning model", "Dependencies" and "Logging" in `.scratch/instance-captioning/spec.md`.

**Blocked by:** 01: Share the per-Instance View planning; 05: Embed Captions and Queries with Qwen3-Embedding

**Status:** ready-for-agent

- [x] Pure functions with unit tests: parsing the reply (plain JSON, code fences, trailing text, missing or empty Name, more than three Synonyms, malformed text), and the boxed context crop (bounds and drawn box on a hand-built case).
- [x] A captioner interface takes images and returns raw text. The Qwen3-VL wrapper uses batch size 1, greedy decoding, a pixel cap and a new-token cap; it imports GPU libraries only inside functions, returns no GPU tensors, and leaves the GPU when the stage ends, also on failure.
- [x] The real Pipeline runs `captioning` after `embedding` inside the per-stage VRAM tracking. It captions only surviving Instances using the first N Views of the shared plan, embeds the Caption documents with the Caption encoder, and stores both on the Instances.
- [x] A Captioning settings group (model id, Views per Instance default 3, context fraction default 0.6, pixel cap, max new tokens default 128) is overridable through the config and passed in server component assembly.
- [x] The GPU requirements pin a `transformers` version that supports Qwen3-VL; nothing new goes into the CPU project dependencies.
- [x] One log line per job gives the number of Instances captioned and without a Caption, through a logger the Colab logging configuration lets through.
- [ ] Seam B (written, not yet run on a GPU): `captioning` is in the stage list, at least one Instance has a non-empty Name, a Query returns `query_top_k` Matches, and only SigLIP2 and the Caption encoder remain on the GPU after a run.
- [x] AGENTS.md notes that the captioning model leaves the GPU after its stage. The v1 spec's stage list and the pipeline docs include Captioning, and the v1 spec no longer lists class labels on Instances as out of scope.
