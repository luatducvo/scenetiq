# Non-commercial use: DA3 Nested-Giant-Large for reconstruction

We reconstruct Captures with Depth Anything 3 `DA3NESTED-GIANT-LARGE-1.1`, the only current DA3 variant that outputs depth directly in meters. Upstream retrained the `-1.1` checkpoints after fixing a training bug and deprecated the original `DA3NESTED-GIANT-LARGE`; the architecture is unchanged. Metric scale matters because Instance positions and bounding boxes must be in real units for future spatial reasoning. Its weights are CC BY-NC 4.0, so SceneTIQ is a non-commercial / research project.

## Considered Options

- `DA3-Large` + `DA3METRIC-LARGE` for scale: `DA3-Large` is also CC BY-NC, so it gains nothing.
- `DA3-Base`/`DA3-Small` + `DA3METRIC-LARGE` (all Apache-2.0): commercially usable, but needs a separate scale-alignment step and gives lower quality.

## Consequences

If the project ever goes commercial, the reconstruction stage must switch to the Apache-2.0 combination. Keep the reconstruction stage behind its own interface so the model can be swapped.
