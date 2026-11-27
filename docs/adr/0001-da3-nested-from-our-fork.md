# Reconstruct with DA3 Nested Giant, installed from our fork

The Reconstructing stage runs Depth Anything 3 `DA3NESTED-GIANT-LARGE-1.1` (`DA3Options.model_id`), the only DA3 variant that outputs depth in metres, so Instance centroids and bounding boxes, and the Relations scored on them, are in real units. Its weights are CC BY-NC 4.0: SceneTIQ is a non-commercial research project.

Upstream DA3 declares `requires-python <=3.13` and `numpy<2`, so it cannot be installed on Python 3.13. The `gpu` extra installs it from our fork, `luatducvo/Depth-Anything-3`, pinned by commit; the fork fixes the packaging and declares its own dependencies. `DA3Reconstructor` imports `depth_anything_3.api` inside `_load`.

## Considered Options

- `DA3-Base`/`DA3-Small` with `DA3METRIC-LARGE` (all Apache-2.0): commercially usable, but needs a separate scale-alignment step and gives lower quality.
- Vendoring DA3's inference code: every move to a newer upstream is a manual diff, and its dependency list is kept by hand.
- Upstream DA3: not installable on Python 3.13 with numpy 2.

## Consequences

- Going commercial means replacing the Reconstructing stage with the Apache-2.0 combination; `Reconstructor` is the seam.
- The GPU install needs the fork reachable on GitHub, and the project's `requires-python` stays `<3.14` until the fork lifts its cap.
- DA3 runs on numpy 2 and without xformers, which upstream never declared; not yet verified on a GPU.
