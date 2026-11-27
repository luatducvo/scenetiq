# Install DA3 from our fork

ADR 0005 vendored the inference code of Depth Anything 3 because upstream declares `requires-python <=3.13` and `numpy<2`, and listed its runtime dependencies by hand. Moving to a newer upstream commit was a manual diff that re-applied every local modification. The owner now maintains a fork, `luatducvo/Depth-Anything-3`, that fixes DA3's packaging; its model code is upstream `3d835ec1`, the commit the vendored copy was last synced to, so DA3's output does not change.

The `gpu` extra of `pyproject.toml` now installs DA3 from the fork, pinned by commit, and `src/scenetiq/vendor/depth_anything_3/` is deleted. The fork declares DA3's dependencies, so the extra no longer lists them, and moving DA3 is a change of one pin. The project's `requires-python` becomes `>=3.12,<3.14`, the fork's own cap, because `uv lock` fails without it. The code imports `depth_anything_3.api.DepthAnything3` inside `DA3Reconstructor._load`, as before.

## Considered Options

- Keep the vendored copy (ADR 0005): no new dependency on a git repository, but every move to a newer upstream is a manual diff, and the hand-kept dependency list drifts.
- Install upstream DA3: upstream still declares `requires-python <=3.13` and `numpy<2`, so it cannot be installed on Colab's Python 3.13 with numpy 2.
- Install the fork by tag: easier to read than a commit, but the fork has none yet; the pin is commit `879fcab51a72518a9249989e6dd352a9aac32f7d` until the owner tags one.

## Consequences

- The GPU install depends on the fork's repository being reachable on GitHub.
- DA3 still runs on numpy 2 and without xformers, which upstream never declared support for; `pytest -m gpu` on Colab and `.scratch/da3-fork/issues/04-accept-on-colab.md` must pass before this is trusted.
- `requires-python` is capped at `<3.14` until the fork lifts its own cap.
