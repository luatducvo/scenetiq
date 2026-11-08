# Configuration

Every backend setting and its default lives in [`src/scenetiq/config.py`](../src/scenetiq/config.py), which is the source of truth if this page ever disagrees with it.

## How to set a value

A value is chosen in this order:

1. The `CONFIG` cell of the notebook (or `SCENETIQ_CONFIG` outside Colab).
2. The default in `config.py`.

Write only the settings you want to set. Stage settings are nested under their stage:

```python
CONFIG = {
    "storage_root": "/content/scenetiq-data",
    "pipeline": "real",
    "view_stride": 7,
    "max_views": 5000,
    "da3": {"process_res": 392, "use_ray_pose": True},
    "sam2": {"points_per_side": 24},
}
```

**On Colab**, edit `CONFIG` in the first code cell and run the cells from there. Edits made in Colab last only for that copy of the notebook. Running the server cell again replaces the running server, which stops any job in progress.

**Outside Colab**, put the same object in `SCENETIQ_CONFIG` as JSON:

```bash
SCENETIQ_TOKEN=dev SCENETIQ_CONFIG='{"view_stride": 2}' uv run uvicorn --factory scenetiq.server:build_app --port 8001
```

The backend reads its settings once, at startup. They apply to new Captures and Queries; a Scene already reconstructed keeps its result.

`SCENETIQ_TOKEN` and `SCENETIQ_CONFIG` are the only two environment variables the backend's settings come from. On Colab the token comes from Colab Secrets, never from `CONFIG`.

## General settings

| Key | Default | Meaning |
|---|---|---|
| `storage_root` | `./data` | Directory for Scenes and the SQLite database. |
| `pipeline` | `"fake"` | `"real"` runs the GPU models. `"fake"` runs none and returns a fixed result, for testing. |
| `query_top_k` | `1` | How many Matches a Query returns: the top K of the fused ranking (ADR 0008). There is no relevance threshold, so a Query always returns Matches when the Scene has Instances. |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant. Each ranking adds `1 / (rrf_k + rank)` to an Instance's fused value; a larger value flattens the difference between ranks. |
| `min_views` | `1` | Fewest photos a Capture may upload. |
| `max_views` | `2000` | Most photos a Capture may upload. It has no upper limit. |
| `view_stride` | `1` | Reconstruct the first photo and every Nth one after it, in upload order. |
| `cors_origins` | `["http://localhost:3000", "http://127.0.0.1:3000"]` | Browser origins allowed to call the API. |

`min_views` and `max_views` count uploaded photos, before `view_stride`. A count outside them is refused before any photo is sent. Photos the stride skips are still uploaded and stored.

### Choosing `view_stride`

Nothing checks a Capture against GPU memory. DA3 takes every selected View in one forward pass, so a Capture too large for the GPU uploads completely and then fails with out-of-memory.

Number of Views reconstructed = uploaded photos ÷ `view_stride`, rounded up. For example, 2000 photos at stride 7 give 286 Views.

As a rough guide, peak VRAM ≈ 10 GB + 0.08–0.11 GB per View, for 3:2 or 4:3 photos at `process_res` 504. This is an estimate from published DA3 figures and the model sizes; it has not been measured on this pipeline.

To measure a real run, read the notebook's log. Each job prints the Merging input size and one line per GPU stage. For example (these numbers are illustrative):

```
merge-input points=1200000 primitives=30000
gpu-memory stage=merging peak_allocated=24.50GB peak_reserved=26.00GB allocated_after=3.74GB
```

DA3 and SAM2 leave the GPU when their stage ends and load again for the next job. SigLIP2 stays, so Queries answer without loading it again.

## Stage settings

### `da3`: depth and cameras

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Weights. The pipeline needs depth in metres, which only the Nested models give (ADR 0001). |
| `process_res` | `504` | Long side, in pixels, that DA3 processes at. It sets most of the VRAM use and detail. DA3 rounds sides to multiples of 14. |
| `process_res_method` | `upper_bound_resize` | How photos are resized to `process_res`. |
| `ref_view_strategy` | `saddle_balanced` | How DA3 picks its reference View: `first`, `middle`, `saddle_balanced` or `saddle_sim_range`. |
| `use_ray_pose` | `False` | Derive cameras from the ray head, which upstream reports as slower but generally more accurate. |
| `conf_percentile` | `30.0` | Drop depth from this percentage of pixels with the lowest confidence, over the whole Capture. Higher gives a cleaner point cloud with more holes. |
| `max_depth` | `12.0` | Drop depth beyond this many metres. No later stage sees it, so this also limits fusion. Raise it for Scenes larger than a room. |

### `fusion`: TSDF mesh

| Key | Default | Meaning |
|---|---|---|
| `voxel_size` | `0.02` | Voxel edge in metres, which sets the point spacing. Smaller gives more points and superpoints, so later stages use more memory. |
| `sdf_trunc_voxels` | `4.0` | Truncation distance, in voxels. |
| `block_resolution` | `16` | Voxels along each edge of a TSDF block. |
| `initial_block_count` | `1000` | Blocks allocated up front. The grid grows beyond this as needed, so it only affects speed and the first allocation. |

### `alignment`: Gravity alignment

Each job logs one line, for example `gravity-alignment method=plane angle=12.3deg views=12 consensus=0.83 plane=0.42`. The `method` is one of:

- `plane`: up came from the normal of a large horizontal surface, which holds the share of the points shown by `plane`.
- `geocalib`: the Views agreed on up but no surface qualified, so their agreed direction was used.
- `none`: the Views did not agree on up, so the Scene was not rotated. `consensus` shows how far they were from agreeing.

| Key | Default | Meaning |
|---|---|---|
| `max_views` | `12` | Most Views to estimate up from, spaced evenly through the Capture. |
| `use_focal_prior` | `True` | Give GeoCalib the focal length from the Reconstructing stage. |
| `consensus_angle_deg` | `10.0` | Largest angle, in degrees, between estimates that count as agreeing. |
| `consensus_fraction` | `0.5` | Agreeing estimates must be more than this share of them all, or the Scene is not rotated. |
| `plane_distance` | `0.03` | How close, in metres, a point must be to count as on a plane. |
| `max_planes` | `5` | Most planes to look for. |
| `min_plane_fraction` | `0.05` | A plane must hold at least this share of the searched points. |
| `max_plane_angle_deg` | `10.0` | Largest angle, in degrees, between a plane's normal and the direction the Views agree on. A larger value accepts more planes, including slopes. |
| `plane_max_points` | `200000` | The plane search subsamples to this many points. The Scene keeps all its points. |
| `plane_iterations` | `1000` | RANSAC iterations per plane. |

These starting values have not been validated on real Captures.

### `superpoints`: groups of mesh points

| Key | Default | Meaning |
|---|---|---|
| `primitives` | `"superpoints"` | `"superpoints"` segments the mesh. `"voxel"` is a voxel-grid fallback for debugging. |
| `k_thresh` | `0.01` | Segmentation threshold. Larger gives larger, fewer superpoints. |
| `min_verts` | `20` | Smallest superpoint, in vertices. |
| `primitive_voxel_size` | `0.15` | Voxel edge in metres. Used only with `"voxel"`. |

MV3DIS allocates matrices of superpoint count × superpoint count on the GPU, so its memory grows with the square of that count.

### `sam2`: Masks

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `facebook/sam2.1-hiera-large` | Weights. |
| `points_per_side` | `32` | Density of the prompt grid. Higher finds smaller objects, more slowly. |
| `pred_iou_thresh` | `0.8` | Drop Masks SAM2 itself rates below this. |
| `stability_score_thresh` | `0.9` | Drop unstable Masks. |
| `min_area_fraction` | `0.001` | Drop Masks smaller than this fraction of the image. |

### `merging`: Masks into Instances

| Key | Default | Meaning |
|---|---|---|
| `visibility_rtol` | `0.05` | Relative depth tolerance for a point to count as seen by a View. |
| `merge_small_points` | `200` | Regions smaller than this are merged or dropped. |
| `min_instance_points` | `50` | Instances with fewer points are dropped. |

### `embedding`: one embedding per Instance

| Key | Default | Meaning |
|---|---|---|
| `views_per_instance` | `5` | How many of the best Views to crop each Instance from. |
| `min_visible_pixels` | `30` | A View must see at least this many of the Instance's points. |
| `expansions` | `[0.1, 0.4]` | Context added around the tight crop, as fractions of its size. Each value gives one crop per View. |
| `visibility_rtol` | `0.05` | Relative depth tolerance of the occlusion test. |

### `siglip`: crops and Query text

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `google/siglip2-giant-opt-patch16-384` | Weights. Scenes store embeddings from the model that built them, so a new model cannot query older Scenes correctly. |
| `prompt` | `This is a photo of {}.` | Template for Query text. It must contain `{}`, or every Query encodes the same text. |
| `batch_size` | `64` | Crops encoded per forward pass. Lower it if this stage runs out of memory. |

### `captioning`: a Caption per Instance

The Captioning stage runs after Embedding, on the Instances it kept, and uses its choice of Views. The model leaves the GPU when the stage ends.

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `Qwen/Qwen3-VL-8B-Instruct` | Vision-language model. About 17 GB at bf16 while the stage runs. |
| `views_per_instance` | `3` | How many of the Instance's best Views go into its prompt. At most `embedding.views_per_instance`. |
| `context_expand` | `0.6` | Context added around the Instance's tight box, as a fraction of its size. The tight box is drawn in red on each crop, and the prompt refers to it as the red box. |
| `max_pixels` | `200704` | Pixel budget per crop (448 × 448). Larger crops are shrunk, keeping their shape. |
| `max_new_tokens` | `320` | Longest reply per Instance. The reply is forced to follow a fixed JSON schema (Name, up to three Synonyms, one Description per Aspect), so the only way to lose a Caption is a reply cut short here; the log then says why. |

### `caption_embedding`: Descriptions and Query text

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `Qwen/Qwen3-Embedding-0.6B` | Text embedding model. It stays on the GPU for Queries (about 1.2 GB). Scenes store Description embeddings from the model that built them, so a new model cannot rank older Scenes' Captions correctly. |
| `query_instruction` | `Given a description of an object in a scene, retrieve the caption of that object` | Task description put before Query text, not before Descriptions. |
| `batch_size` | `32` | Descriptions encoded per forward pass. |
| `max_length` | `256` | Longest input in tokens; longer text is cut. |

## Invalid configuration

The backend refuses to start on any of these. On Colab, the install cell reports them before anything is installed.

| Mistake | Example | Message |
|---|---|---|
| Unknown key | `{"da3": {"proces_res": 392}}` | `unknown config key: da3.proces_res` |
| Wrong type | `{"view_stride": "7"}` | `config value view_stride has the wrong type: '7'` |
| Value not among the choices | `{"pipeline": "reel"}` | `config value pipeline has the wrong type: 'reel'` |
| Token in the config | `{"token": "abc"}` | `token must not be in the config; it is a secret` |
| Impossible limits | `{"min_views": 11, "max_views": 10}` | `min_views (11) must not exceed max_views (10)` |
| Values below 1 | `{"view_stride": 0}` | `view_stride must be at least 1, got 0` |
| Negative RRF constant | `{"rrf_k": -1}` | `rrf_k must not be negative, got -1` |
| Removed setting | `{"match_threshold": 0.5}` | `unknown config key: match_threshold` |

Other values are not range-checked, so `{"captioning": {"max_new_tokens": 1}}` starts normally and then leaves every Instance without a Caption.

## Not configurable

| What | Where | Why |
|---|---|---|
| `REPO_URL`, `BRANCH`, `PORT` | Notebook, beside `CONFIG` | Needed to fetch and start the backend, before `config.py` is available. |
| 255 Masks per View | `MAX_MASKS_PER_VIEW` in `pipeline/masking.py` | MV3DIS stores mask ranks as `uint8`. |
| Patch size 14, sky threshold 0.3 | Vendored DA3 | Part of the upstream model code. |
| MV3DIS `DEFAULT_ARGS` | Vendored MV3DIS | Upstream defaults of the algorithm. |
