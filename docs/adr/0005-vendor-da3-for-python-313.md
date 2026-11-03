# Vendor the DA3 inference code so the GPU stack runs on Colab's Python 3.13

Colab moved to Python 3.13, and the GPU install failed. At the pinned commit, and still on upstream `main` and PyPI `0.1.1`, Depth Anything 3 declares `requires-python <=3.13` (which excludes 3.13.x) and `numpy<2` (no numpy 1.x wheel exists for 3.13). We vendor only the inference subset of DA3 into `src/scenetiq/vendor/depth_anything_3/`, list its runtime dependencies in `requirements-gpu.txt`, and run on Colab's own Python with numpy 2. `open3d` moves to 0.20.0, the first release with cp313 wheels.

## Considered Options

- Install DA3 with `pip install --no-deps --ignore-requires-python` and list its dependencies by hand: no code change, but it depends on an installer escape hatch, and the hand-kept list silently drifts whenever the pin moves.
- Pin a Colab runtime version that still ships Python 3.12: no change at all, but Colab retires old runtimes, so it only defers the problem.
- Run the server from a separate Python 3.12 venv inside Colab: keeps upstream DA3 untouched, but every session downloads a CUDA torch and must match it to Colab's driver.

## Consequences

- Moving to a newer DA3 commit is a manual diff against upstream, re-applying the modifications listed in `PROVENANCE.md`.
- DA3 now runs on numpy 2 and without xformers (pure-PyTorch SwiGLU fallback). Upstream never declared support for either, so the GPU smoke tests (`pytest -m gpu`) on Colab must pass before this is trusted.
