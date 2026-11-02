# 11: Real open-vocabulary Query: SigLIP2 Instance embeddings

**What to build:** On Colab, the user types any free-text Query against a real Scene and gets meaningful Matches, including objects outside any fixed class list.

Spec: `.scratch/scene-reconstruction-query/spec.md` (Pipeline stage 6, Query module)

**Blocked by:** 06 (query and Matches), 09 (real masking and merging)

**Status:** ready-for-agent

- [ ] The **Embedding** stage, for each Instance:
  - [ ] selects the Views where the Instance is visible
  - [ ] crops the Instance from those Views
  - [ ] encodes the crops with the SigLIP2 (large) image tower
  - [ ] aggregates the results into one L2-normalised embedding, stored with the Scene
- [ ] The real text encoder (SigLIP2 text tower) replaces the fake on Colab. CPU tests keep the fake.
- [ ] Instances with no usable View are handled explicitly, either dropped or given a defined embedding. The docstring documents the choice.
- [ ] A `gpu`-marked smoke test runs a Query for a clearly present object on a small real Capture and checks that it returns a Match whose Instance overlaps that object.
- [ ] The Definition of Done in AGENTS.md is met.
