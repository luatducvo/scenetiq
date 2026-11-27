# GPU stack

## Dependencies

The GPU stack is the `gpu` extra in `pyproject.toml`, locked in `uv.lock`. `uv sync` does not select it, so the CPU suite installs no torch. ty whitelists the lazily imported third-party packages in `[tool.ty.analysis]`.

- Never `uv sync` on Colab: the lock pins PyPI's newest torch, which would replace Colab's CUDA-matched one. `uv pip` ignores the lock, so `scenetiq.colab.install_packages` exports it (`uv export --frozen --extra gpu`) without the torch family, then runs `uv pip install --system -e <repo> -r <exported> -c <held>`. The fake Pipeline installs `-e <repo>` alone.
- The torch family is `torch`, `torchvision`, `triton` and every `nvidia-*` and `cuda-*` package, read from `uv.lock`. Rerun `uv lock` after changing the extra: the export is `--frozen`, so an out-of-date lock is installed as it is.
- `<held>` pins the installed `torch` and `torchvision` without the local segment (`2.11.0+cu126` becomes `2.11.0`), so a package that requires torch is satisfied by Colab's build. Hold only these two: Colab's whole `pip freeze` would also pin its `transformers`, which clashes with ours.
- DA3 comes from our fork, pinned by commit in the extra (ADR 0013); the fork declares its own dependencies, so none are listed here. Move DA3 by changing that pin and running `uv lock`. The fork caps `requires-python` at `<3.14`, and so does the project: without it `uv lock` fails. Its `[tool.uv]` indexes and CUDA extras do not apply here. Leave `xformers` out: its wheels pin a torch version and can replace Colab's CUDA torch. The fork brings `opencv-python-headless` and GeoCalib `opencv-python`; both provide `cv2`, and Colab ships both.
- SAM2 has no wheel, and its `setup.py` imports torch, so uv could not lock it without building it. Its metadata is declared in `[[tool.uv.dependency-metadata]]` (name `sam-2`); update it with `setup.py`'s `NAME`, `VERSION` and `REQUIRED_PACKAGES` whenever the SAM2 pin moves.
- Meta publishes no SAM2 wheel and its build requires torch, so uv would download a fresh torch with its CUDA libraries into an isolated build environment and compile a CUDA extension. The install sets `SAM2_BUILD_CUDA=0` and passes `--no-build-isolation-package sam-2` on the command line (not in `[tool.uv]`, which uv reads from the working directory, and the notebook does not run uv from the checkout), so SAM2 builds in seconds against Colab's torch and setuptools. SceneTIQ does not use `sam2._C`: it only fills holes in masks when `max_hole_area > 0`, and `SAM2AutomaticMaskGenerator` on images does not call it; SAM2 warns and skips that step.
- Install SAM2 from Meta's GitHub repo only; the PyPI `sam2` package is a third-party upload.
- `xgrammar` and `transformers` are pinned together. Before changing either pin, check on Colab that installing it leaves `torch.__version__` unchanged.

## Vendored code

Vendored code is in `src/scenetiq/vendor/`, excluded from ruff and ty.

- Each directory needs `LICENSE` and `PROVENANCE.md` (upstream URL, pinned commit, every local modification).
- Keep changes minimal and record each one in `PROVENANCE.md`.
- Wrappers live outside `vendor/` and are fully checked.

## TSDF

Fusion uses Open3D's tensor `VoxelBlockGrid`. The legacy `ScalableTSDFVolume` keeps almost nothing with metric depth in 0.20.0 ([Open3D#7592](https://github.com/isl-org/Open3D/issues/7592)), and no older release has Python 3.13 wheels.

## Memory and resident models

- DA3 and SAM2 leave the GPU when their stage ends, also on failure.
- SigLIP2, Qwen3-Embedding and Qwen3-VL stay resident for Queries. Build one instance of each in `server.build_components`, shared by the Pipeline and the API; their loads take a lock, because Queries and the job run on different threads.
- Qwen3-VL's lock is held for one generation at a time, through `ResidentModel.use()`, so a Query can generate between two Instances. The lock is not re-entrant: each generation runs outside any other.
- With Qwen3-VL resident, a job starts with about 17 GB less free memory for DA3 and SAM2.
- A stage's output holds only CPU data; a GPU tensor left in it keeps its model's memory allocated.
- Nothing checks the View count against GPU memory: DA3 takes every View left after the View stride in one forward pass, so an oversized Capture fails with OOM.
