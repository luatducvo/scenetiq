# Research 02: Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1) at Dataset scale

Question: how to run DA3NESTED-GIANT-LARGE-1.1 on a **Dataset** of 1 to 2000 RGB images so that a **Scene Build** produces one metric, consistent **Point Cloud**, and what per-pixel data the build must keep so that 2D semantic features can later be back-projected onto the same points.

Sources (all primary, read 2026-11-21):

- **[repo]**: `github.com/ByteDance-Seed/Depth-Anything-3`, commit `3d835ec1` (2026-07-27). Paths below are relative to the repo root.
- **[card]**: `huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1`, HF API: sha `b2359bdf`, last modified 2025-12-11.
- **[paper]**: Lin et al., *Depth Anything 3: Recovering the Visual Space from Any Views*, arXiv:2511.10647 (HTML version).
- **[pypi]**: PyPI JSON API (wheel tags), queried 2026-11-21.

How claims are marked: **(verified)** means read in code or docs. **(inferred)** means my own derivation from verified facts. **(uncertain)** means not stated by any primary source.

---

## TL;DR

1. **Python: pin 3.12.** `pyproject.toml` declares `requires-python = ">=3.9, <=3.13"`. Under PEP 440 that excludes 3.13.1 and later as well as 3.14, so pip/uv will refuse to install on 3.14. Separately, `numpy<2` is a hard dependency, and numpy 1.26.4 (the last 1.x release) only ships wheels up to cp312. In practice that means **Python 3.10–3.12** (verified). Torch only needs `torch>=2`. xformers is optional, because the code falls back to PyTorch SDPA (verified).
2. **Outputs per image:** depth `[N,H,W]` in **meters**, conf `[N,H,W]` (≥1), K `[N,3,3]`, w2c extrinsics `[N,3,4]` (OpenCV/COLMAP convention), sky mask, a per-forward-pass `scale_factor`, and the resized RGB. All of these are at the **processed resolution**: by default the longest side is 504 and each side is rounded to a multiple of 14. Built-in exports: `glb` (point cloud + cameras), `npz`/`mini_npz`, `colmap`, `gs_ply`/`gs_video` (3DGS), `depth_vis`, `feat_vis` (verified).
3. **What NESTED means:** a DA3-Giant any-view model (1.15B) and a DA3Metric-Large monocular model (0.35B) both run on the same images. The Giant depth is rescaled by **one least-squares scalar** fitted to the metric depth on confident, non-sky pixels. The same scalar multiplies the camera translations (verified, `model/da3.py`).
4. **Capacity (paper Table 8, A100 80 GB, 504×336):** DA3-Giant runs at **37.6 img/s** on a 32-image scene and fits at most **900–1000 images in one forward pass**. NESTED adds the metric model, so its own limit is somewhat lower (uncertain, not published). **2000 images will not fit in one pass.**
5. **Official answer for long inputs: DA3-Streaming** (`da3_streaming/`). It splits the sequence into chunks (`chunk_size: 120`, `overlap: 60`), aligns consecutive chunks with IRLS Sim(3) on the overlapping frames, detects loops with SALAD retrieval (`similarity_threshold 0.85`, `top_k 5`), and runs a Sim(3) pose-graph optimization. The default weights are DA3NESTED-GIANT-LARGE-1.1. Peak VRAM is **28.3 GB** at chunk 120, 504×378. Measured speed: **8.51 FPS on an A100** over 11,373 KITTI frames (504×154) (verified).
6. **2000-image build estimate:** about **5–15 min on an A100** and about **3–10 min on an H100** for DA3-Streaming at 504-px width, plus I/O and fusion (inferred, see §5). A Scene Build in "tens of minutes" is comfortably achievable.
7. **License:** code is **Apache-2.0**. The checkpoint is **CC BY-NC 4.0 (non-commercial only)**. The loop-closure submodule SALAD is **GPL-3.0** (verified).

---

## 1. Installation and inference API

### Install (verified, `README.md`, `pyproject.toml`)

```bash
pip install xformers "torch>=2" torchvision
pip install -e .            # package name: depth-anything-3, CLI entry point: `da3`
# optional: gsplat (3DGS export), gradio app
```

Declared dependencies include `numpy<2`, `open3d`, `xformers`, `pycolmap`, `e3nn`, `evo`, `trimesh`, `opencv-python`, `huggingface_hub`, `safetensors`, `moviepy==1.0.3`, `fastapi`, `uvicorn` and more. The package is not published on PyPI, so it is installed from git or a vendored copy (verified: README only shows `pip install -e .`). DA3-Streaming additionally needs `faiss-gpu`, `numba`, `pypose`, `pandas`, `prettytable` (`da3_streaming/requirements.txt`), plus the SALAD git submodule. The C++ Sim(3) solver is optional and falls back to Python (`base_config.yaml: SIM3_Optimizer.lang_version`).

### Python version (verified)

| Constraint | Effect |
|---|---|
| `requires-python = ">=3.9, <=3.13"` | PEP 440 pads `3.13` to `3.13.0`. I checked with `packaging.SpecifierSet`: only `3.12.x` and `3.13.0` satisfy it, while `3.13.1+` and `3.14` are **rejected** by pip/uv. |
| `numpy<2` | numpy 1.26.4 wheels are cp39–cp312 only [pypi]. No numpy 1.x release supports 3.13. |
| `open3d` | Current 0.20.0 has cp310–cp314 wheels [pypi], so it is not the binding constraint. |
| `torch>=2` | No upper bound. Current torch 2.15 has cp311–cp315 wheels [pypi]. |
| `xformers` | Optional at runtime: `model/dinov2/layers/swiglu_ffn.py` tries `from xformers.ops import SwiGLU` and falls back to a pure-PyTorch `SwiGLUFFN`. Attention uses `F.scaled_dot_product_attention`. README FAQ links issue #11 for GPUs without xformers. |

**Conclusion:** the GPU backend must pin **Python 3.12** (3.11 also works). **Python 3.14 does not work** without forking the package metadata and moving it to numpy 2. That migration is untested and has no upstream support (uncertain whether it is otherwise compatible). If the rest of scenetiq targets 3.14, the DA3 worker has to be a separate process or image with its own interpreter.

### Python API (verified, `src/depth_anything_3/api.py`, `docs/API.md`)

```python
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda")
pred = model.inference(
    image=paths_or_PIL_or_ndarrays,      # list
    extrinsics=None, intrinsics=None,    # optional (N,4,4) w2c / (N,3,3): pose-conditioned mode
    use_ray_pose=False,                  # True = pose from ray head: slower, more accurate
    ref_view_strategy="saddle_balanced", # "first" | "middle" | "saddle_balanced" | "saddle_sim_range"
    process_res=504, process_res_method="upper_bound_resize",
    export_dir=None, export_format="mini_npz",   # "-"-joined, e.g. "mini_npz-glb"
    export_feat_layers=[],               # backbone features -> pred.aux["feat_layer_i"]
    conf_thresh_percentile=40.0, num_max_points=1_000_000,  # GLB only
)
```

- The forward pass runs under `torch.autocast` with bf16 when the GPU supports it, otherwise fp16 (`api.py:126`).
- Checkpoint download is **6.76 GB** (`model.safetensors` = 6,759,558,100 bytes, HF API). The card table says **1.40B params**, while the HF sidebar says "2B params" (two different counting methods; uncertain which is right).
- `use_ray_pose`: on DA3NESTED-GIANT-LARGE, ray-head AUC3 is HiRoom 84.4 / ETH3D 52.6 / ScanNet++ 89.4, versus 80.3 / 48.4 / 85.0 for the camera head (README FAQ). **Recommend `use_ray_pose=True`**, since cost is not a constraint.
- CLI: `da3 auto <dir> --export-format glb`. `da3 backend` keeps the model resident on the GPU (`docs/CLI.md`).

## 2. Outputs per image

`Prediction` dataclass (`src/depth_anything_3/specs.py`), all numpy:

| Field | Shape | Meaning |
|---|---|---|
| `depth` | N,H,W float32 | z-depth along the optical axis. For NESTED it is **meters** (README FAQ: "output from `DA3NESTED-GIANT-LARGE` is already in meters"). |
| `conf` | N,H,W float32 | Depth confidence. Activation `expp1` = `exp(x)+1`, so **conf ≥ 1** (`model/dualdpt.py:353`). |
| `extrinsics` | N,3,4 | **world-to-camera**, OpenCV/COLMAP convention. The world frame is the selected reference view's camera. |
| `intrinsics` | N,3,3 | Pinhole K, predicted **per image**, in processed-resolution pixels. |
| `sky` | N,H,W bool | Sky mask from the metric branch (`sky >= 0.5`, `utils/io/output_processor.py:147`). |
| `is_metric`, `scale_factor` | scalar | `1` and the least-squares metric scale for this forward pass. |
| `processed_images` | N,H,W,3 uint8 | The resized/cropped RGB the model actually saw. Use these for point colors. |
| `aux["feat_layer_i"]`, `gaussians` | optional | Backbone features and 3DGS parameters (`infer_gs=True`). |

**How NESTED makes depth metric** (verified, `NestedDepthAnything3Net.forward`, `model/da3.py:336-445`):

1. `da3` (Giant, any-view) predicts depth, conf, K and pose up to an unknown scale.
2. `da3_metric` (Large, monocular) predicts canonical depth and sky. It is converted to meters with `depth * mean(fx,fy) / 300` using the **Giant-predicted** K (`utils/alignment.py: apply_metric_scaling`). The paper §4.4 trains with canonical focal length 300, following Metric3Dv2.
3. Alignment mask = non-sky (`sky < 0.3`) AND Giant conf ≥ median conf AND both depths above small minimums. A **single scalar** `scale_factor` is fitted by least squares, then `depth *= s` and `extrinsics[..., :3, 3] *= s`.
4. Sky pixels get depth = `min(p99(non-sky depth), 200 m)` and conf = `1.0`. On the ≥1 conf scale that is the **minimum**, so any conf threshold above 1 drops sky automatically, despite the code comment saying "high confidence".

Consequence: metric scale is **one number per forward pass (chunk)**, not per image. Scale accuracy depends on the monocular metric model and the predicted focal length (inferred).

**Built-in exports** (verified, `docs/API.md`, `utils/export/*`):

- `glb`: fused colored point cloud plus camera frustums. Conf threshold = `clamp(1.05, p40(conf_non_sky), p90(conf_non_sky))`; near-black (<16) and near-white (≥240) pixels are forced to conf 1.0. The cloud is randomly subsampled to `num_max_points` (1M), **not voxel-downsampled**, and recentered/rotated to glTF axes (`glb.py`). This makes it a viewer artifact, not a canonical Scene format.
- `npz` / `mini_npz` (depth, conf, exts, ixts, ±image), `colmap` (needs image paths), `gs_ply`/`gs_video` (needs `infer_gs=True` and gsplat), `depth_vis`, `feat_vis`.
- **Recommendation (inferred):** call `inference()` without export, then do fusion and voxel downsampling ourselves (e.g. Open3D `voxel_down_sample`) from the raw arrays.

## 3. VRAM and throughput

**Paper Table 8** (§7.2.3), A100 80 GB, 504×336 input:

| Model | Speed (img/s, 32-view scene) | Max images per forward pass on 80 GB |
|---|---|---|
| DA3-Giant | 37.6 | **900–1000** |
| DA3-Large | 78.4 | 1500–1600 |
| VGGT (reference) | 34.1 | 400–500 |

The paper adds that offloading intermediate tokens to CPU "could process many more images" (not implemented in the repo, uncertain).

- **NESTED** = Giant plus the metric-Large forward on the same N images. The metric model has no cross-view attention (`alt_start: -1` default), so its memory scales linearly. **The NESTED max is therefore somewhat below 900–1000 (uncertain, no published figure).** Plan for ≤ ~800 views per pass on 80 GB at 504×336.
- **Why throughput falls with N (inferred):** the Giant backbone (ViT-g, 40 blocks) uses **global cross-view attention on alternating blocks from block 13** (`configs/da3-giant.yaml: alt_start: 13`, `vision_transformer.py:333`). Tokens per view = (H/14)·(W/14), e.g. 36×24 = 864 at 504×336. Global attention compute is quadratic in N·tokens, while memory is roughly linear under SDPA/flash. So 37.6 img/s at N=32 overstates per-image speed for N in the hundreds.
- **Resolution:** training used a 504 longest side with aspect ratios 504×{504,378,336,280} plus portrait and 896/756/672×504 (paper §3.4). Views per sample were 2–18 at 504×504. Going above 504 is out of distribution and expensive (tokens grow ∝ pixels, attention ∝ pixels²) (inferred). **Keep `process_res=504`.**
- **DA3-Streaming peak VRAM** (`da3_streaming/README.md`), with overlap = chunk/2:

  | chunk size | 120 | 90 | 60 | 30 |
  |---|---|---|---|---|
  | KITTI 504×154, peak GB | 15.9 | 14.3 | 12.7 | 11.5 |
  | TUM 504×378, peak GB | 28.3 | 25.1 | 21.2 | 18.7 |
  | TUM ATE RMSE (m) | 0.087 | 0.091 | 0.127 | 0.227 |

  Accuracy degrades sharply below chunk 60, so larger chunks are better. With an 80 GB card, chunk sizes well above 120 should fit at 504×378. That is extrapolated, because upstream only tested up to 120 (uncertain).

## 4. Handling hundreds to thousands of images

### Option A: single pass (N ≤ ~800 on 80 GB)
One `inference()` call over every image gives one shared world frame and one metric scale, with no alignment step. This is the simplest correct path for small and medium Datasets (inferred from §3).

### Option B: DA3-Streaming (official, for long inputs) (verified, `da3_streaming/`)
Built on VGGT-Long. Defaults from `configs/base_config.yaml`:

| Parameter | Default | Role |
|---|---|---|
| `chunk_size` / `overlap` | 120 / 60 | Sliding window. Chunk i covers `[i·60, i·60+120)`, so `num_chunks = ceil((N−60)/60)` = **33 for N = 2000**. |
| `align_method` | `sim3` (alternatives `se3`, `scale+se3`) | Chunk-to-chunk transform estimated on the overlapping frames' dense point maps. |
| `align_lib` | `triton` (GPU) | Alternatives `torch`, `numba`, `numpy`. |
| `IRLS` | `delta 0.1, max_iters 5, tol 1e-9` | Huber-weighted robust Sim(3) (`loop_utils/sim3utils.py: robust_weighted_estimate_sim3`). Conf threshold for alignment points = `0.1 × min(median conf)`. |
| `loop_enable` | True | SALAD (DINOv2) global descriptors at 336×336, batch 32. `similarity_threshold 0.85`, `top_k 5`, `nms_threshold 25` frames. |
| `loop_chunk_size` | 20 | Each loop closure re-infers a 40-image chunk (20 around each end of the loop) to measure the loop Sim(3). |
| `SIM3_Optimizer` | `max_iterations 30, lambda_init 1e-6` | Levenberg–Marquardt pose-graph over the chunk Sim(3)s (C++ or Python). |
| `ref_view_strategy` | `saddle_balanced` | Reference view inside each chunk. |
| `Pointcloud_Save` | `conf_threshold_coef 0.75, sample_ratio 0.015` | Per-chunk PLY keeps points with `conf ≥ 0.75·mean(conf)`, then randomly keeps 1.5% of them. |
| `save_depth_conf_result` | True | Writes a per-frame npz (rgb, depth × chunk scale, conf, K) plus `camera_poses.txt` (c2w, all chunks mapped into chunk 0's frame) and `intrinsic.txt`. |
| `depth_threshold` | 15.0 | Present in YAML but **not read by any `.py` file** (verified with grep). |

Streaming details that matter for scenetiq:

- **Streaming assumes ordered input.** Images are `sorted(glob(*.jpg)+glob(*.png))` (`da3_streaming.py:703`), and chunks are consecutive slices. Each chunk must reconstruct well on its own, so neighbours in the list must actually overlap in view. A Dataset of unordered photos must first be put in a covisibility order. Options: capture time or filename for video-like captures, or a greedy nearest-neighbour path over global descriptors such as the SALAD/DINOv2 descriptors the pipeline already computes (inferred, not provided upstream).
- **Global metric scale is anchored to chunk 0.** With `sim3`, every chunk is rescaled into chunk 0's frame, so chunk 0's NESTED `scale_factor` sets the metric scale of the whole Scene. Per-chunk metric scales are discarded. `scale+se3` or `se3` keep more of each chunk's metric estimate (inferred from `da3_streaming.py`, `align_2pcds`).
- Streaming inference uses API defaults: `process_res=504` and `use_ray_pose=False` (`da3_streaming.py:273`). Because it is plain Python, these can be changed. Confidence is shifted with `conf -= 1.0`, so streaming conf starts at 0 (README: "minimum value of confidence is 0").
- Intermediate chunks are saved to disk as `.npy` and deleted at the end (`delete_temp_files`). The README warns that this needs significant disk space.

### Option C: hybrid (inferred, not upstream)
Pick ~200–800 keyframes covering the Dataset and run them in one pass to get the global frame and scale. Then register the remaining images either by pose-conditioned inference in chunks that include anchor keyframes (`extrinsics=`/`intrinsics=` inputs with `align_to_input_ext_scale=True`, which Umeyama-aligns to the given poses, `api.py:341-365`), or with Streaming-style Sim(3) on shared anchors. Avoids loop-closure machinery and GPL SALAD. Untested.

**Recommendation for the MVP (inferred):** N ≤ ~800 → Option A. N > 800 → DA3-Streaming with a larger chunk on 80 GB (e.g. 200–300 with overlap = chunk/2), after covisibility ordering. Loop closure stays enabled unless the GPL-3.0 SALAD licence is unacceptable.

## 5. Estimated build time for 2000 images (inferred)

- **Measured anchor:** DA3-Streaming took **22 min 17 s for 11,373 frames = 8.51 FPS on an A100**, chunk 120 / overlap 60 with loop closure. This excludes warm-up, model loading and PLY saving, at KITTI 504×154 (≈396 tokens/view) (`da3_streaming/README.md`).
- At 504×336 or 504×378 (864–972 tokens/view, about 2.2–2.5× more), per-frame cost rises by at least 2.2×, and more because of attention. That gives an estimated **3–4 FPS on an A100**, so **2000 images ≈ 8–11 min**, plus loop-closure re-inference and SALAD retrieval.
- H100: commonly ~1.5–2.5× an A100 for bf16 transformer inference (uncertain, not measured for DA3). Estimate **~4–7 min**.
- Single-pass Option A for 800 images: well under 1 min of forward time on an A100 (37.6 img/s at N=32 gives 21 s, slower at large N because of quadratic attention). Call it **~0.5–3 min** (uncertain).
- Add: R2 download of 2000 images, model load (6.8 GB weights, ~30–60 s cold), back-projection plus voxel downsampling (seconds to a minute on GPU), and the later semantic feature pass (separate model, not covered here).
- **Planning number: about 10–20 min wall-clock per 2000-image Scene Build on one A100, less on H100.**

## 6. Per-pixel data to keep for semantic back-projection

The model works on a **resized** image, and every geometric output (depth, conf, K, sky) lives in that processed pixel grid. Back-projection therefore needs the exact mapping between original pixels and processed pixels (verified, `utils/io/input_processor.py`):

1. `upper_bound_resize` (default) scales so the longest side = 504, preserving aspect ratio. Then **each side is rounded to the nearest multiple of 14** with a small extra resize (`_make_divisible_by_resize`), so x and y scales can differ slightly. K is rescaled at both steps (`_resize_ixt`).
2. `*_crop` methods center-crop to a multiple of 14 instead and shift cx, cy (`_crop_ixt`).
3. **If images in one call have different sizes**, all are **center-cropped to the smallest H,W** and cx, cy are shifted (`_unify_batch_shapes`). Normalize or group a Dataset's image sizes up front, or record the crop offsets.

**Keep per image** (inferred, needed for any re-projection):

| Item | Why |
|---|---|
| original `(W0,H0)` and processed `(W,H)`; scale factors `sx=W/W0, sy=H/H0` (before any crop) and any crop offsets | Map semantic-feature pixels (computed on the original or another resolution) to the depth grid. |
| `K` (processed px) | Unprojection and projection. |
| final **w2c** after any chunk Sim(3) (`[s·R | t]` folded in, or store c2w plus the chunk scale) | Common world frame. |
| `depth` (meters, after chunk scale) | Unproject: `X_w = c2w · (d · K⁻¹ [u,v,1]ᵀ)` (`utils/export/glb.py: _depths_to_world_points_with_colors`). |
| `conf` | Filtering and fusion weights. |
| `sky` mask | Exclude sky explicitly instead of relying on conf. |
| `processed_images` | Point colors aligned pixel-for-pixel with the depth grid. |

Storage: about 504×378×(4+4+1) B ≈ 1.7 MB per image, or **≈3.4 GB for 2000 images** as float32 (fp16 depth/conf halves that) (inferred).

**Recommended back-projection scheme (inferred):** assign features to the fused, voxel-downsampled points by **projecting points into views**, rather than unprojecting feature pixels. For each fused point and each view: project it with K and w2c, check the pixel is in bounds, run a visibility test (`|z_proj − depth[v,u]| < τ·depth`), and require `conf[v,u] ≥ thr` and not sky. Then average the 2D feature sampled at that location, weighted by conf. This keeps exactly one feature per Point Cloud point and handles occlusion. It only needs the per-view arrays above, not the full-resolution point maps.

**Confidence filtering, upstream practice** (verified):

- API scale: conf ≥ 1, and sky is set to exactly 1.0.
- GLB export: `thr = clamp(1.05, p40, p90)` over non-sky conf, plus a black/white background drop.
- Streaming (conf shifted −1): fusion keeps `conf ≥ 0.75·mean(conf)` per chunk. Sim(3) alignment uses `conf ≥ 0.1·min(median)`.
- NESTED's metric alignment uses `conf ≥ median` on non-sky pixels.
- Suggested starting point: drop sky, then keep pixels at or above the per-image 40th percentile of non-sky conf (matches the GLB default). Tune visually. No absolute threshold is documented (uncertain).

## 7. Licenses (verified)

| Artifact | License | Source |
|---|---|---|
| DA3 code (repo) | **Apache-2.0** | `LICENSE`, `pyproject.toml`, GitHub API |
| DA3NESTED-GIANT-LARGE-1.1 weights | **CC BY-NC 4.0, non-commercial only** | README model table, HF card metadata `cc-by-nc-4.0` |
| DA3-GIANT(-1.1), DA3-LARGE(-1.1) | CC BY-NC 4.0 | README model table |
| DA3-BASE, DA3-SMALL, DA3METRIC-LARGE, DA3MONO-LARGE | Apache-2.0 | README model table |
| SALAD (DA3-Streaming loop-closure submodule plus `dino_salad.ckpt`) | **GPL-3.0** | GitHub API for `serizba/salad` |

Implication: any commercial use of scenetiq is blocked by the checkpoint's licence. The only commercially usable fully-multi-view option is DA3-BASE/SMALL (Apache-2.0), and those are not metric. Pairing them with DA3METRIC-LARGE (Apache-2.0) would need a reimplementation of the nested alignment, which is about 30 lines in `model/da3.py` (inferred). Shipping DA3-Streaming with loop closure brings GPL-3.0 code into the worker.

## 8. Open questions / uncertain items

- The actual NESTED max views per pass on 80 GB, and throughput at N = 500–1000 (not published, measure on target hardware).
- Whether DA3-Streaming's 8.51 FPS was measured with NESTED-1.1 (the download script fetches it, but the README table does not name the checkpoint).
- Python 3.14 / numpy 2 compatibility of the code itself (blocked by metadata, untested).
- Absolute quality of metric scale indoors versus outdoors (one scalar per pass, from monocular metric depth).
- The best covisibility ordering for unordered Datasets before streaming (not provided upstream).
