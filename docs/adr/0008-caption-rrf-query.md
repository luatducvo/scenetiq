# Captions from a VLM, ranked against Queries by Reciprocal Rank Fusion

A Query used to be scored only against each Instance's SigLIP2 image embedding, and an Instance was a Match when its sigmoid score cleared a threshold. That misses Queries phrased as synonyms, materials or uses ("couch", "wooden thing", "something to sit on").

A Captioning stage after Embedding now asks a local vision-language model (Qwen3-VL-8B-Instruct) to read each Instance from boxed crops of its best Views. It returns a Caption: a Name, up to three Synonyms, and one short English Description for each of four fixed Aspects (Identity, Appearance, Material, Use).

Each non-empty Description is embedded on its own by a separate text embedding model (Qwen3-Embedding-0.6B). An Instance's Caption similarity to a Query is the similarity of its best-matching Description, so a Query about one Aspect is answered by the Description of that Aspect and is not diluted by the others. When the model leaves Identity empty, it is written from the Name and Synonyms, so every Caption has at least one Description. The model's reply is forced, token by token, to follow a fixed JSON schema (constrained decoding with xgrammar), so a reply can be lost only by being cut off at the token limit.

A Query is ranked twice, against the image embeddings and against the best Description of each captioned Instance, and the two rankings are fused with Reciprocal Rank Fusion. The Matches are the top K of that fused ranking, where K is the `top_k` a Query asks for (at most `query_max_top_k`) or `query_top_k` by default: there is no threshold and no Low-confidence Match any more.

## Considered Options

- Mixing the Caption's SigLIP2 text embedding into the image embedding: keeps one score, but SigLIP2's calibration is for image–text pairs, and text–text cosines run high enough to saturate its sigmoid.
- A weighted sum of the image score and the Caption cosine: needs two scales made comparable, and the weight and threshold could not be validated.
- A threshold per channel (image OR Caption): keeps the Low-confidence Match, but ranking across two scales stays arbitrary.
- An external VLM API: needs a key, costs per call and sends the user's photos off the Colab runtime.
- Qwen3-VL-32B: better captions, but about 65 GB in bf16 leaves roughly 10 GB beside the resident SigLIP2 and text model for multi-image prompts.
- Captioning between Merging and Embedding: would caption Instances that Embedding then drops, and could not share Embedding's View selection.
- One keyword-style document per Caption ("sofa; also: couch; grey, fabric; used for sitting"), embedded as one vector: the first version of this stage. A Query about one Aspect is compared with a vector that blends all of them.
- The mean of an Instance's Description similarities: a Query about colour is pulled down by the Material and Use sentences.
- One RRF channel per Aspect: rewards Instances that match many Aspects, but adds rank noise from the Aspects a Query says nothing about.
- A Context Aspect (what surrounds the Instance): the crops show little context, the model invents it, and relations between Instances belong to Spatial reasoning, which needs real geometry.
- A few-shot example in the prompt: the model tends to copy it.
- Asking for JSON by prompt alone: a malformed reply silently costs the Instance its Caption. transformers has no constrained decoding of its own, and outlines would replace our chat template and `generate` call with its wrapper; xgrammar plugs into `generate` as a logits processor.

## Consequences

Every Query on a Reconstruction with Instances returns K Matches, even when nothing in the Scene fits; the user judges relevance from the ranking and the Name, not from a bar. A fused score depends on ranks, so it says nothing absolute and cannot be compared across Scenes. Instances without a Caption (unusable model output) get only their image rank, so they rank below captioned Instances of similar image similarity.

Three models now stay on the GPU between jobs: SigLIP2, the text embedding model and the VLM. The VLM used to leave the GPU when Captioning ended; it now stays, also when the stage fails, so that the Query parser of Spatial reasoning can use it without a load. It loads on first use, by a job or a Query, and each generation holds its lock, so a Query waits for at most one Instance's Caption. The cost: every job starts with about 17 GB less free GPU memory for DA3 and SAM2, and nothing checks the View count against it. The VLM adds its load time and a generation per Instance to every job, and four Descriptions per Instance make that generation longer than a list of keywords.
