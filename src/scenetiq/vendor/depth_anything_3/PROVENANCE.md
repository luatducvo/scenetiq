# Depth Anything 3 (vendored, inference subset)

- Upstream: https://github.com/ByteDance-Seed/Depth-Anything-3 (`src/depth_anything_3/`)
- Pinned commit: `40674546e40144452ec93035ac77d762f4b5d16f`
- Licence: Apache-2.0 (`LICENSE`). Only the code is vendored. The `DA3NESTED-GIANT-LARGE` weights are downloaded from Hugging Face at runtime and are CC BY-NC 4.0 (ADR 0001).

Why it is vendored: ADR 0005. Upstream declares `requires-python <=3.13` and `numpy<2`, so it cannot be installed on Colab's Python 3.13.x.

## Files

Only the modules that `api.DepthAnything3` reaches with the bundled configs are kept: `api.py`, `cfg.py`, `registry.py`, `specs.py`, `configs/*.yaml`, `model/` (except `utils/gs_renderer.py`) and these modules from `utils/`: `alignment`, `constants`, `geometry`, `io/input_processor`, `io/output_processor`, `logger`, `parallel_utils`, `pose_align`, `ray_utils`, `sh_helpers`.

Not vendored: `app/`, `services/`, `cli.py`, `utils/export/`, `model/utils/gs_renderer.py` and the other `utils/` modules (visualisation, COLMAP, Gaussian-splat helpers, model_loading).

## Local modifications

- Every absolute import `depth_anything_3.` is rewritten to `scenetiq.vendor.depth_anything_3.`, in the `.py` files and in the `__object__.path` entries of `configs/*.yaml`.
- The `__inherit__: depth_anything_3.configs.<name>` entries in `configs/da3nested-giant-large.yaml` are left as they are. `cfg.load_config` drops the first dotted segment and resolves the rest relative to `cfg.py`, so they still resolve.
- Empty `__init__.py` files are added where upstream relies on implicit namespace packages: the package root, `model/dinov2/`, `model/utils/`, `utils/`, `utils/io/`.
- `api.py`: the `utils.export` import is removed. `DepthAnything3._export_results` raises `NotImplementedError`, so `inference(export_dir=...)` is unsupported. SceneTIQ never passes `export_dir`.
- `utils/pose_align.py`: `from evo.core.trajectory import PosePath3D` moved from module level into `_umeyama_sim3_from_paths`. evo is needed only when input extrinsics or the Gaussian branch are used, and SceneTIQ uses neither.

## Optional upstream dependencies left out

- `xformers`: `model/dinov2/layers/swiglu_ffn.py` falls back to its pure-PyTorch `SwiGLUFFN` (same `w12`/`w3` weights). Not installed, because an xformers wheel pins one torch version and could replace Colab's CUDA torch.
- `evo`, `e3nn`: imported lazily, only on the code paths above.
