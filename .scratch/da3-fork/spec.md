# Spec: Install DA3 from our fork, from a locked GPU stack

Status: ready-for-agent

Vocabulary follows `GLOSSARY.md`. Respect the ADRs in `docs/adr/`, in particular 0001 (DA3 weights licence), 0003 (SAM2 automatic Masks) and 0005 (why DA3 is vendored), which this work supersedes.

## Problem Statement

The GPU stack is installed on Colab from `requirements-gpu.txt`, which no lock covers. Every Colab session resolves it again, so a package declared as `einops>=0.8` or `huggingface-hub>=0.25` can come out at a different version from one session to the next, and a Scene built today may not be reproducible tomorrow.

Depth Anything 3 is vendored in `src/scenetiq/vendor/depth_anything_3/` (ADR 0005), because upstream declares `requires-python <=3.13` and `numpy<2`. Its runtime dependencies are listed by hand, and moving to a newer upstream commit is a manual diff that re-applies every local modification in `PROVENANCE.md`. The owner now maintains a fork, `luatducvo/Depth-Anything-3` (branch `chore/uv-deps`), that fixes DA3's packaging, so the vendored copy is no longer needed.

Installing SAM2 is also slower than it should be. Meta publishes no wheel, so uv builds SAM2 from git. Its build requires torch, so uv downloads a fresh torch with its CUDA libraries into an isolated build environment, although Colab already has one, and `setup.py` compiles a CUDA extension that SceneTIQ does not use.

## Solution

The whole GPU stack becomes a `gpu` extra in `pyproject.toml` and is resolved into `uv.lock`. Colab installs it from the lock: `scenetiq.colab` exports the lock to a pinned requirements file without the torch family, and installs that file into Colab's own Python while holding Colab's preinstalled torch and torchvision with a constraints file. `requirements-gpu.txt` is deleted.

DA3 is installed from the fork at a pinned commit, and the vendored copy is deleted. The fork's model code is upstream `3d835ec1`, the commit the vendored copy was last synced to (commit `a98f58c`), so DA3's output does not change.

SAM2 is built without its CUDA extension and against Colab's torch, so its build takes seconds and downloads no torch.

The user sees no change. The owner gets identical package versions in every Colab session, one place to declare a dependency, and a one-line change to move DA3.

## User Stories

1. As the owner, I want every Colab session to install the same versions of the GPU stack, so that a Scene can be rebuilt with the same results.
2. As the owner, I want Colab's CUDA torch and torchvision left untouched by the install, so that the models run on the GPU driver Colab ships.
3. As the owner, I want the notebook's install step to stay a single call, so that deployment does not get harder.
4. As the owner, I want a failed install to show uv's last lines in the notebook, as it does now, so that I can tell what went wrong.
5. As the owner, I want SAM2's install to take seconds rather than minutes, so that a fresh runtime is ready sooner.
6. As the owner, I want to move DA3 to a newer commit by changing one pin, so that I no longer diff and copy upstream code by hand.
7. As the owner, I want DA3's own dependencies declared by DA3, so that a new upstream dependency cannot be silently missing.
8. As a user, I want Reconstructions to be identical before and after this change, so that existing and new Scenes agree.
9. As a developer, I want `uv sync` to install no torch, as now, so that the CPU suite stays light.
10. As a developer, I want `uv lock` to resolve on a CPU-only machine without building SAM2, so that locking is fast and needs no GPU.
11. As a developer, I want ty and the CPU suite to pass without DA3, SAM2 or torch installed, so that nothing changes for day-to-day work.
12. As a developer, I want the decision to drop the vendored DA3 recorded in an ADR that supersedes ADR 0005, so that nobody re-vendors it by accident.
13. As a developer, I want the install rules for the GPU stack documented in `docs/agents/gpu-stack.md`, so that the next dependency change follows them.

## Implementation Decisions

**The `gpu` extra**
- `[project.optional-dependencies] gpu` holds what `requirements-gpu.txt` holds now: `open3d==0.20.0`, SAM2 and GeoCalib from git at their current pins, `tqdm>=4.66` (vendored MV3DIS), `transformers==5.22.2`, `xgrammar==0.2.8`, and DA3. Until DA3 comes from the fork, its runtime dependencies stay listed by hand in the extra.
- `uv lock` resolves the extra, torch included, against PyPI. `uv sync` does not select it.
- SAM2's metadata is declared statically in `[[tool.uv.dependency-metadata]]` (name `sam-2`, version `1.0`, and the seven `REQUIRED_PACKAGES` of its `setup.py` at the pinned commit), so `uv lock` never builds it. It must be updated whenever the SAM2 pin moves.

**Installing on Colab from the lock**
- `uv pip` ignores `uv.lock`, and `uv sync` would replace Colab's torch, so the lock reaches Colab through `uv export`.
- The install runs two uv commands: `uv export --frozen --extra gpu --no-hashes --no-emit-project` with `--no-emit-package` for every package of the torch family, written to a temporary file; then `uv pip install --system -e <repo> -r <exported> -c <held>`. The fake Pipeline installs `-e <repo>` alone, as now.
- The torch family is `torch`, `torchvision`, `triton`, and every package named `nvidia-*` or `cuda-*`, read from `uv.lock`. They are left out because the lock pins PyPI's newest torch, not Colab's.
- The constraints file pins the installed `torch` and `torchvision` to their versions without the local segment (`2.11.0+cu126` becomes `2.11.0`, which still matches the installed build). A package that is not installed is skipped. This is a second guard: a package that requires torch is satisfied by Colab's build and uv keeps it.
- `install_packages` keeps its progress bar and its error with uv's last lines. The export is part of the "Resolve packages" stage.

**SAM2 build**
- The install sets `SAM2_BUILD_CUDA=0` and passes `--no-build-isolation-package sam-2` on the command line. The flag goes on the command line, not in `[tool.uv]`, because uv reads `[tool.uv]` from the working directory, and the notebook does not run uv from the checkout.
- Without isolation, SAM2 builds against the system's torch and setuptools, both preinstalled on Colab.
- SceneTIQ does not need `sam2._C`: it only fills holes in masks when `max_hole_area > 0`, and `SAM2AutomaticMaskGenerator` on images does not use it. SAM2 warns and skips that step when the extension is missing. Ticket 04 confirms this.

**DA3 from the fork**
- The extra declares `depth-anything-3 @ git+https://github.com/luatducvo/Depth-Anything-3.git@<pin>`. The pin is a tag on the fork if the owner creates one, otherwise commit `879fcab51a72518a9249989e6dd352a9aac32f7d`. The hand-listed DA3 dependencies leave the extra; the fork declares them.
- `requires-python` becomes `>=3.12,<3.14`, the fork's own cap; without it `uv lock` fails.
- `RealReconstructor._load` imports `depth_anything_3.api.DepthAnything3`, still inside the function. Nothing else in the call changes.
- `src/scenetiq/vendor/depth_anything_3/` is deleted, with its `LICENSE` and `PROVENANCE.md`.
- ty allows `depth_anything_3.**` as an unresolved import.
- The fork's `[tool.uv]` indexes and CUDA extras do not carry over to SceneTIQ, and its `xformers` extra is not used.
- The fork brings `opencv-python-headless` and GeoCalib brings `opencv-python`; both provide `cv2`. Colab already ships both.

**Documentation**
- ADR 0013, "Install DA3 from our fork", supersedes ADR 0005, which gets a superseded note.
- `docs/agents/gpu-stack.md` replaces its Dependencies section: the `gpu` extra, install from the exported lock, the torch family and held packages, SAM2's static metadata and build flags.
- `AGENTS.md` names `depth_anything_3` instead of `scenetiq.vendor.depth_anything_3` in the lazy-import rule, and drops "vendored DA3".
- `docs/agents/unverified.md` replaces the DA3 row with one that points at ADR 0013 and ticket 04.
- The docstring of `server.build_components` names the `gpu` extra.

## Testing Decisions

- The install commands, the torch-family selection from a lock file and the held constraints are pure functions in `scenetiq.colab`, unit-tested like the existing `install_command` test, with a small lock fixture and a fake version lookup. No test runs uv.
- No test touches DA3, SAM2 or the install itself on CPU. Seam B (`pytest -m gpu` on Colab) and ticket 04 cover them.
- Done means the commands in `AGENTS.md` pass. `ruff format --check .` currently fails on markdown under `.agents/skills/` and `.claude/skills/`, unrelated to this work.

## Out of Scope

- Replacing the SAM2 package with the SAM2 classes in `transformers` or with Ultralytics (different mask generation, and Ultralytics is AGPL-3.0).
- Hosting a prebuilt SAM2 wheel.
- Moving MV3DIS or the segmentator out of `vendor/`.
- Sending the fork's packaging fixes upstream.

## Further Notes

- The fork's `plan.md` suggests using the host's whole `pip freeze` as constraints. That would also pin Colab's `transformers` and clash with `transformers==5.22.2`, which is why only torch and torchvision are held.
- Resolving the extra for Colab (Linux x86_64, Python 3.13) with torch held at 2.11.0 was checked with `uv pip compile` during planning, and `uv lock` resolved with the fork in the extra.
