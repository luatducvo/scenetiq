# 05: Embed Captions and Queries with Qwen3-Embedding

**What to build:** With the real Pipeline, Caption documents and Query text are embedded by `Qwen/Qwen3-Embedding-0.6B`. One loaded instance serves both the job and the API. It stays on the GPU between jobs, and it loads under a lock because Queries and the job run on different threads. See "Text embedding model" in `.scratch/instance-captioning/spec.md`.

**Blocked by:** 03: Store Captions and rank Queries against them

**Status:** ready-for-agent

- [x] A wrapper implements the Caption encoder interface: last-token pooling, L2 normalisation, the model's query instruction prefixed to Queries only, and batched documents.
- [x] `torch` and `transformers` are imported only inside functions; the model loads on first use under a lock; outputs are plain arrays.
- [x] A Caption encoder settings group (model id, query instruction, batch size) is overridable through the config.
- [x] Server component assembly builds one instance for the real Pipeline and passes it to both the Pipeline and the API; a unit test covers assembly without loading the model.
- [x] Building the Caption document from a Caption is a pure function with a unit test.
- [x] AGENTS.md notes that the Caption encoder stays resident on the GPU and loads under a lock; the configuration docs list the new settings.
- [ ] A Seam B test (written, not yet run on a GPU) encodes a Query and a few documents, and checks that the document naming the Queried object ranks first.
