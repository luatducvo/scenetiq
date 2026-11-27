# 03: Store Captions and rank Queries against them

**What to build:** A Reconstruction can carry a Caption (Name, up to three Synonyms, Attributes, Use) and a Caption embedding per Instance, and a Query fuses the image ranking with a Caption ranking. With the fake Pipeline, job progress shows a `captioning` stage after `embedding`, some Instances have Captions and one does not, and a Query whose words only the Caption matches still finds its Instance first. See "Caption", "Query", "Storage" and the Seam A part of "Testing Decisions" in `.scratch/instance-captioning/spec.md`.

**Blocked by:** 02: Return top-K Matches ranked by RRF

**Status:** ready-for-agent

- [x] The domain has a Caption type and a `captioning` stage after `embedding`; an Instance has an optional Caption and an optional Caption embedding.
- [x] A Caption encoder interface encodes a batch of documents and one Query, both L2-normalised; a fake implementation maps known strings to known vectors.
- [x] The Scene directory stores each Instance's Caption (or null) in the Instance metadata, and a Caption embeddings file aligned with it (zero row where null), written only when some Instance has a Caption.
- [x] The fake Pipeline reports every stage including `captioning`, and returns Instances with and without Captions and known Caption embeddings.
- [x] The API is built with the Caption encoder. A Query is ranked against image and Caption embeddings and fused by RRF; an Instance without a Caption gets only its image term and a null `caption_score`; `channels` is 2 when any Instance has a Caption, else 1.
- [x] Server component assembly returns the Pipeline, the text encoder and the Caption encoder (fakes for the fake Pipeline).
- [x] Seam A tests with hand-worked values: a Caption-only win, an Instance without a Caption, a Scene without Captions (one channel), and `captioning` in job progress.
- [x] The frontend's job progress lists `captioning` after `embedding`.
