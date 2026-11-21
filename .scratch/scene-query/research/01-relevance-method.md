# Research: per-point Relevance method for open-vocabulary Queries

Ticket: `.scratch/scene-query/issues/01-relevance-method.md` · Parent: `.scratch/scene-query/map.md`
Researched: 2026-11-21. Terms follow `GLOSSARY.md` (Dataset, Scene, Scene Build, Point Cloud, Query, Relevance, Heatmap).

## TL;DR

Use **family 1: dense, pixel-aligned, language-aligned features fused per point**, using
**C-RADIOv4 (SigLIP2-g adaptor) with a RayFronts/RADSeg-style locality fix ("NARADIO")**.
At Scene Build time, encode every image, back-project the patch features through DA3 depth and poses,
and average them into each voxel of the Point Cloud. At Query time, encode the text with the
same adaptor's text tower and compute cosine similarity against the per-point features on the GPU, then apply
a LERF-style canonical-negatives relevancy. This is the only family that gives a natively
**continuous** score per point in **milliseconds**, and the strongest primary evidence shows it beats
mask-based (ConceptGraphs) 3D pipelines on zero-shot 3D semantic segmentation. Family 2
(SAM masks + CLIP) gives piecewise-constant per-instance scores and is a heavy pipeline. Family 3
(grounding at query time) has the best 2D quality (SAM 3), but scales with image count at Query
time (up to 2000 forward passes) and returns masks or boxes, not a continuous field.

## Constraints that decide it

- Query in under a few seconds, on Datasets of up to 2000 images. Query-time cost must not scale with image count.
- A Scene Build may take tens of minutes, so heavy per-image work belongs at build time.
- Relevance is continuous, and the Heatmap is not a mask (GLOSSARY: "Relevance … continuous score"; _Avoid: mask_).
- One voxel-downsampled Point Cloud serves both viewing and Querying.
- Research or non-commercial licenses are OK. GPU cost is not a constraint.
- DA3 gives per-image `depth [N,H,W]`, `conf`, `extrinsics [N,3,4]` (w2c) and `intrinsics [N,3,3]`. That is everything needed to back-project per-pixel features ([DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3), "prediction.*" block). The DA3NESTED-GIANT-LARGE-1.1 weights are **CC-BY-NC-4.0** ([HF API tag](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)), so the whole system is already non-commercial.

## Family 1: dense pixel-aligned VL features, lifted and fused per point

**How it works.** A per-image encoder outputs an H'×W'×D feature map in a text-aligned space. Each
pixel goes through DA3 depth into a world-space point, and each Point Cloud voxel averages the
features that land in it, optionally weighted by DA3 `conf`. Relevance(q) = cos(text_emb(q), f_point).
OpenScene's "2D fusion" mode is exactly this, with LSeg/OpenSeg ([OpenScene README](https://github.com/pengsongyou/openscene): "Multi-view fused LSeg/OpenSeg features").

**Encoder options (primary sources):**

| Encoder | Text-aligned dense? | Feature dim | License | Status |
|---|---|---|---|---|
| **C-RADIOv4-H / -SO400M** with `siglip2-g` adaptor | Yes. The adaptor returns spatial features *and* has `tokenizer` + `encode_text` ([RADIO README](https://github.com/NVlabs/RADIO), Example 5). Card lists "Dense processing (semantic segmentation…)"; input up to 2048 px in steps of 16 ([HF card](https://huggingface.co/nvidia/C-RADIOv4-H)) | adaptor = SigLIP2-g embedding width (1536 for SigLIP2-giant; verify at load) | Weights: NVIDIA Open Model License (commercially permissive) ([card](https://huggingface.co/nvidia/C-RADIOv4-H)). Repo code: NVIDIA Source Code License (NC) ([LICENSE](https://github.com/NVlabs/RADIO/blob/main/LICENSE)) | Active. C-RADIOv4 released Jan 2026, teachers SigLIP2-g, DINOv3-7B, SAM3; 412M / 631M params ([arXiv 2601.17237](https://arxiv.org/abs/2601.17237)) |
| NARADIO (RayFronts) | RADIO + SigLIP adaptor plus a Gaussian locality kernel in the last attention block. "State-of-the-art pixel-level language alignment and real-time performance" ([RayFronts §II-A, §III](https://arxiv.org/html/2504.06994v1)) | stored 768 by default; PCA to 100 for large outdoor maps (same paper, App. A / §V-C) | MIT ([repo](https://github.com/RayFronts/RayFronts)) | Active (pushed Oct 2026), IROS'25 |
| RADSeg | RADIOv3 + SigLIP2 adaptor, self-correlating recursive attention + global aggregation, optional RADIO-SAM refinement ([arXiv 2511.19704](https://arxiv.org/html/2511.19704v2)) | as RADIO adaptor | arXiv CC BY 4.0 (paper). Code at radseg-ovss.github.io; check the code license | CVPR 2026 Findings |
| Talk2DINO | Learns a CLIP-text → DINOv2 patch-space projection; SOTA on unsupervised OVS benchmarks ([arXiv 2411.19331](https://arxiv.org/abs/2411.19331)) | DINOv2 patch dim (e.g. 768/1024) | Apache-2.0 ([repo](https://github.com/lorebianchi98/Talk2DINO)) | Active. ICCV'25, DINOv3 support added Oct 2025 |
| SigLIP2 so400m NaFlex | Global alignment only. Dense use needs MaskCLIP/NACLIP-style tricks | 1152 ([config.json](https://huggingface.co/google/siglip2-so400m-patch16-naflex/raw/main/config.json)) | Apache-2.0 | Model only, no 3D tooling |
| DINOv3 + `dino.txt` | Zero-shot head shipped ([DINOv3 README](https://github.com/facebookresearch/dinov3), "Zero-shot tasks with dino.txt") | backbone dim | DINOv3 License (custom, gated) | Active |
| LSeg | Yes, CLIP ViT-L/16 space | 512 | MIT | **Archived** ([isl-org/lang-seg](https://github.com/isl-org/lang-seg), archived=true) |
| OpenSeg | Yes | 768 | Apache-2.0 (google-research) | TF-only, used by OpenScene. Not maintained as a package |
| MaskCLIP | Training-free CLIP dense | 512/768 | Apache-2.0 | Last push Sep 2022 ([repo](https://github.com/chongzhou96/MaskCLIP)) |
| FeatUp | Upsampler for any backbone, not a VL model by itself | backbone dim | MIT | Last push Jun 2024 ([repo](https://github.com/mhamilton723/FeatUp)) |

**Open-vocab quality in 3D (most decision-relevant evidence).** RayFronts Table II, offline zero-shot
3D semantic segmentation, mIoU without background ([arXiv 2504.06994](https://arxiv.org/html/2504.06994v1)):

| Method (family) | Replica | ScanNet |
|---|---|---|
| ConceptGraphs (2: masks + CLIP) | 11.63 | 21.62 |
| ConceptFusion (1/2 hybrid) | 21.07 | 21.76 |
| HOV-SG (2) | 16.93 | 26.79 |
| NACLIP-3D (1) | 20.37 | 31.66 |
| Trident-3D (1) | 21.30 | 29.97 |
| **RayFronts / NARADIO (1)** | **39.37** | **41.29** |

RADSeg reports 3D mIoU of 32.3 / 36.3 / 36.2 (Replica / ScanNet / ScanNet++, feature-space
aggregation), and 33.1 / 38.6 / 39.9 with SAM refinement. That is with a 106M base model at about 0.115 s per image on a
V100 ([RADSeg](https://arxiv.org/html/2511.19704v2)). The benchmarks use different protocols, so compare within a table.
They still agree that **dense RADIO-family features fused in 3D beat mask-graph pipelines** for zero-shot labelling.

**Query latency.** One text-encoder forward pass (about 10 ms), then one matrix-vector product, N_points × D.
At 5M points × 1536 dims in fp16 that is about 15 GB/s of reads, a few ms on a datacenter GPU. Even on CPU with PCA-128 it
is well under a second. **Independent of image count.** (Arithmetic estimate, not a measured benchmark.)

**Build cost per image.** One encoder forward pass plus back-projection. RayFronts runs its NARADIO
encoder at 17.5 Hz end-to-end on embedded hardware, versus 0.03 Hz for ConceptFusion (RayFronts §V-C).
C-RADIOv4-H at about 1024 px on a datacenter GPU should be on the order of 0.1 s per image, so about 3–5 min for 2000
images. (Estimate. Measure in the Scene Build ticket.)

**Per-point memory and storage.** D × 2 bytes (fp16). That is 1536 → 3 KB/pt, 768 → 1.5 KB/pt.
At 1M points that is 1.5–3 GB, and at 5M points 7.7–15 GB. A Scene is session-only and GPU memory is not constrained,
so the features can stay on the GPU. Compression options:
- **PCA** fitted per Scene, top 64–256 components. RayFronts uses top-100 PCA for large maps
  (§V-C). Relevance stays linear: `t·f ≈ (Pt)·(P(f−μ)) + t·μ`, so the text embedding is
  projected with the same P and nothing else changes.
- Store **voxel features only**, never per-pixel. OpenScene's per-point fused ScanNet features are
  175–235 GB for the whole dataset (README download list), which shows why you fuse once and drop the pixels.
- fp16 → int8 per-channel quantization if needed.

**Python / CUDA.** RADIO loads via `torch.hub` or HF `transformers` `AutoModel` ([card](https://huggingface.co/nvidia/C-RADIOv4-H)), so it is
pure PyTorch with no custom CUDA ops. RayFronts pins Python 3.11 and torch 2.4.1 / cu121 (repo), but we only need its
NARADIO attention change (one module), not the whole mapping stack. DA3 needs `xformers`, `torch>=2`, Python ≥3.10 (DA3 README).

**Weaknesses.** Patch-level resolution (16 px) blurs thin objects. CLIP/SigLIP-style "bag of words"
behaviour hurts compositional Queries ("chair left of the table"). Raw cosine scores are poorly
calibrated across Queries, which the relevancy normalisation below handles.

## Family 2: class-agnostic masks (SAM/SAM2) + one CLIP embedding per mask, lifted to 3D

**How it works.** SAM/SAM2 proposes masks per image (or OpenMask3D's 3D proposals are projected to 2D). Each mask
gets a CLIP embedding from multi-scale crops. Masks are merged across views into 3D objects (ConceptGraphs
object graph; OpenMask3D per-instance feature; Open3DIS 2D+3D proposals). A Query scores each object, and
every point in an object inherits that score.

- **Quality.** Strong when the Query names a whole, segmentable object. Weak on stuff, parts and
  attributes that do not match a mask. ConceptGraphs scores 11.6 / 21.6 mIoU vs NARADIO 39.4 / 41.3 in RayFronts
  Table II. The score is **constant within an instance**, so the "Heatmap" collapses into a few flat
  blobs. That is effectively a soft mask, which GLOSSARY tells us to avoid.
- **Query latency.** Fast (one text embedding vs K object embeddings, K ≈ 10²–10³).
- **Build cost per image.** High: SAM "everything" mode (many masks per image) + N crops × CLIP
  per image + cross-view association. ConceptFusion, which is similar in spirit, runs at 0.03 Hz (RayFronts §V-C).
  For 2000 images that is hours unless heavily subsampled.
- **Storage.** Tiny per point (an instance id), plus K × D for the objects.
- **License.** SAM2 Apache-2.0 ([HF](https://huggingface.co/facebook/sam2.1-hiera-large)). OpenMask3D MIT, ConceptGraphs MIT,
  Open3DIS BSD-3, OpenScene Apache-2.0 (GitHub API).
- **Python / CUDA / maintenance.** OpenMask3D pins Python 3.8.5 and last pushed Dec 2023. ConceptGraphs is tested
  on Python 3.10 / torch 2.0.1 / cu118 with PyTorch3D, plus Grounded-SAM built against `CUDA_HOME` (README).
  Open3DIS last pushed Nov 2024. All are research code tied to old stacks and RGB-D SLAM inputs.

## Family 3: 2D open-vocab grounding at query time, back-projected

**How it works.** For each Query, run a text-conditioned detector or segmenter on each image, then back-project the
masks or boxes through DA3 depth into the Point Cloud and accumulate per-point votes.

- **Quality.** Highest 2D open-vocab quality available. **SAM 3** ("Segment Anything with Concepts")
  segments every instance of a text noun phrase and returns masks, boxes and scores. On SA-Co/Gold, cgF1 is
  SAM 3 54.1 vs OWLv2 24.6 vs DINO-X 21.3 vs human 72.8. LVIS mask AP is 48.5 ([SAM 3 README](https://github.com/facebookresearch/sam3)).
  Grounding DINO (Apache-2.0), Florence-2 (MIT) and OWLv2 (Apache-2.0) are older and weaker (OWLv2 row above).
- **Query latency: the deal-breaker.** Cost is (#images) forward passes *per Query*. SAM 3 is 848M
  params (README). At an optimistic 20–50 ms per image on an H100, 2000 images take 40–100 s per Query, which is 10–50× over budget.
  Subsampling to about 50 keyframes would fit, but loses coverage. Back-projected confidences are also not continuous. Box
  detectors (Grounding DINO / OWLv2) smear scores over the whole box.
- **Build cost.** Near zero (only DA3).
- **Storage.** None per point.
- **License.** SAM 3 uses the custom **SAM License** (Nov 2025, gated on HF; allows research use, bans ITAR/military use)
  ([LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)). The others are permissive.
- **Python / CUDA.** SAM 3 needs Python ≥3.12, PyTorch ≥2.7 (install uses torch 2.10 cu128), CUDA ≥12.6 (README).
  Grounding DINO's original repo compiles custom CUDA ops against `CUDA_HOME` (README). HF `transformers` ports
  exist for Grounding DINO, OWLv2 and Florence-2.

## Side-by-side

| | 1. Dense VL features (NARADIO / C-RADIOv4) | 2. SAM masks + CLIP (ConceptGraphs etc.) | 3. Query-time grounding (SAM 3 etc.) |
|---|---|---|---|
| Continuous per-point Relevance | **Yes, natively** | No, constant per instance | No, masks/boxes, vote-averaged |
| Open-vocab quality (3D) | Best zero-shot 3D mIoU in RayFronts | Lower (11.6/21.6 ConceptGraphs) | Best 2D, but coverage-limited if subsampled |
| Query latency @ 2000 imgs | **ms** | ms | 40–100 s (est.) |
| Build cost / image | ~0.1 s (est.) | seconds (SAM-everything + crops) | ~0 |
| Per-point storage | D×2 B (1.5–3 KB); PCA-128 → 256 B | id + K×D | 0 |
| License | NVIDIA Open Model (weights) / NC code; RayFronts MIT | Apache/MIT/BSD | SAM License (custom) / Apache / MIT |
| Stack | Pure PyTorch, hub/transformers | Old pins (py3.8–3.10, torch 2.0, PyTorch3D) | py≥3.12, torch≥2.7, CUDA≥12.6 |
| Maintained | RADIO, RayFronts, Talk2DINO active (2026) | Mostly stale (2023–2024) | SAM 3 very active |

## Recommendation

**Adopt family 1: C-RADIOv4-H + `siglip2-g` adaptor, with NARADIO-style locality attention, fused per voxel.**

1. **Scene Build:** for each image, take adaptor spatial features (fp16) at about 1024 px with bilinear upsampling to
   depth resolution. Back-project through DA3 depth/intrinsics/extrinsics, drop low-`conf` pixels, and
   average into the same voxel grid used to downsample the Point Cloud, weighted by conf. L2-normalise the result. Optionally
   fit a per-Scene PCA (128–256) if memory matters. Since memory is not constrained, keep full D.
2. **Query:** `t = encode_text(q)`, then `s = F · t` on the GPU. Turn it into Relevance with the LERF canonical-negatives
   relevancy (pairwise softmax of the Query vs phrases such as "object", "things", "stuff", "texture", taking the min)
   ([LERF, arXiv 2303.09553](https://arxiv.org/abs/2303.09553)). Then min-max over the cloud for the cold-to-hot Heatmap.

**Why:**
- It is the only family whose output is a **continuous per-point field** by construction, which matches the GLOSSARY definition of Relevance.
- **Query cost is independent of Dataset size** (one text encode + one GEMV), so it meets the latency budget comfortably at 2000 images.
- All the heavy work fits the "tens of minutes" Scene Build budget (about one encoder pass per image).
- It has the **best primary-source zero-shot 3D numbers** among the families compared (RayFronts Table II; RADSeg 3D results), using the same agglomerative backbone.
- It has a **simple, modern stack.** RADIO is plain PyTorch via torch.hub/transformers, actively maintained, and its weights license is permissive. It avoids SAM-everything plus old PyTorch3D pins.

**Fallbacks and follow-ups (not MVP):**
- If RADIO adaptor quality disappoints qualitatively, swap the encoder for Talk2DINO (Apache-2.0) behind the same "per-image dense text-aligned features" seam. Nothing downstream changes.
- If sharper boundaries are wanted later, add **SAM 3 refinement on a small set of top-scoring views** at Query time (family 3 as a re-ranker), or RADSeg's RADIO-SAM refinement at build time.
- Compositional or relational Queries are out of reach for all three families without an LLM/VLM layer. Flag this in the spec as a known limitation.

## Open questions for later tickets

- Exact adaptor feature dim and recommended input resolution for C-RADIOv4 + `siglip2-g`. Check at load time, then fix the per-point memory budget.
- Measured per-image encoder time at the chosen resolution on the target GPU (Scene Build ticket).
- Python: the project is 3.14, but this stack (and SAM 3 if used later) is tested on 3.11–3.12. Pin the backend to 3.12 unless torch/xformers/DA3 wheels are confirmed on 3.14 (packaging ticket).
- Code license of the NARADIO/RADSeg attention modules we would vendor (RayFronts is MIT; confirm RADSeg).
