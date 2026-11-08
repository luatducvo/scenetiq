# Captions from a VLM, ranked against Queries by Reciprocal Rank Fusion

A Query used to be scored only against each Instance's SigLIP2 image embedding, and an Instance was a Match when its sigmoid score cleared a threshold. That misses Queries phrased as synonyms, materials or uses ("couch", "wooden thing", "something to sit on"). A Captioning stage after Embedding now asks a local vision-language model (Qwen3-VL-8B-Instruct) to read each Instance from boxed crops of its best Views and return a Caption: a Name, up to three Synonyms, Attributes and Use. The Caption becomes one text document, embedded by a separate text embedding model (Qwen3-Embedding-0.6B). A Query is ranked twice, against the image embeddings and against the Caption embeddings, and the two rankings are fused with Reciprocal Rank Fusion. The Matches are the top K of that fused ranking: there is no threshold and no Low-confidence Match any more.

## Considered Options

- Mixing the Caption's SigLIP2 text embedding into the image embedding: keeps one score, but SigLIP2's calibration is for image–text pairs, and text–text cosines run high enough to saturate its sigmoid.
- A weighted sum of the image score and the Caption cosine: needs two scales made comparable, and the weight and threshold could not be validated.
- A threshold per channel (image OR Caption): keeps the Low-confidence Match, but ranking across two scales stays arbitrary.
- An external VLM API: needs a key, costs per call and sends the user's photos off the Colab runtime.
- Qwen3-VL-32B: better captions, but about 65 GB in bf16 leaves roughly 10 GB beside the resident SigLIP2 and text model for multi-image prompts.
- Captioning between Merging and Embedding: would caption Instances that Embedding then drops, and could not share Embedding's View selection.

## Consequences

Every Query on a Reconstruction with Instances returns K Matches, even when nothing in the Scene fits; the user judges relevance from the ranking and the Name, not from a bar. A fused score depends on ranks, so it says nothing absolute and cannot be compared across Scenes. Instances without a Caption (unparseable model output) get only their image rank, so they rank below captioned Instances of similar image similarity. Two models now stay on the GPU between jobs (SigLIP2 and the text embedding model), and the VLM adds its load time and a generation per Instance to every job.
