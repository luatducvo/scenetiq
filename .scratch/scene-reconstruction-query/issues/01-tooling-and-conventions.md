# 01: Tooling and conventions: ruff, ty, pytest layout, AGENTS.md

**What to build:** A developer (human or agent) who clones the repo can lint, check formatting, type-check and test the project with one documented set of commands. AGENTS.md states the rules that every later ticket must follow. This is the prefactor that every other ticket builds on.

Spec: `.scratch/scene-reconstruction-query/spec.md`

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Dev tooling lives in a uv dev dependency group: ruff, ty, pytest, and what the API tests need (httpx).
- [ ] Ruff is configured in pyproject for both lint and format.
  - The lint rule set is broad and explicit, not just the defaults. It includes at least: pycodestyle, pyflakes, isort, pyupgrade, bugbear, simplify, pep8-naming, and pydocstyle (`D`) with a single docstring convention (Google).
  - The target Python version is 3.12.
- [ ] Docstrings are required on every public module, class, function and method in the first-party package, enforced through ruff's `D` rules. Tests may be exempted from `D`.
- [ ] ty is configured in pyproject, and `ty check` reports zero diagnostics on the package and the tests.
- [ ] A location is reserved for vendored third-party code (added in later tickets) and excluded from ruff and ty. First-party adapters that wrap vendored code are not excluded.
- [ ] A complete pytest layout exists:
  - [ ] a top-level tests directory, split into `unit` (pure functions), `api` (Seam A: HTTP API with a fake Pipeline) and `eval` (Seam B: GPU only)
  - [ ] shared fixtures in conftest
  - [ ] a registered `gpu` marker, deselected by default, so that a plain `pytest` passes on a CPU-only machine
- [ ] The placeholder `main` gets a docstring or is replaced, so that the package passes every check.
- [ ] AGENTS.md gains a "Development" section with:
  - [ ] the exact commands for lint, format check, type check, the default test run and the GPU test run
  - [ ] the docstring convention
  - [ ] the test layout, and the two test seams with what belongs in each
  - [ ] a **Definition of Done** for every ticket:
    - ruff lint and format pass
    - ty passes
    - docstrings are present
    - tests are added at the right seam and pass
    - AGENTS.md is updated whenever the ticket adds a command, directory, convention or config value that a future agent needs to know
  - [ ] pointers to `GLOSSARY.md` and `docs/adr/`
- [ ] All checks pass on the repo as it stands after this ticket.
