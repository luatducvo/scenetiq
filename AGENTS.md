## Agent skills

### Issue tracker

Issues and specs are tracked as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Development

Python 3.12, managed with uv. Use the vocabulary in `GLOSSARY.md` (Scene, Capture, View, Reconstruction, Mask, Instance, Query, Match) in names, docstrings and tests. Respect the decisions in `docs/adr/`. The current feature spec is `.scratch/scene-reconstruction-query/spec.md`.

### Commands

| Purpose | Command |
|---|---|
| Install / sync dev env | `uv sync` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run ty check` |
| Tests (CPU, default) | `uv run pytest` |
| GPU tests (Colab only) | `uv run pytest -m gpu` |

### Docstrings

Google convention, enforced by ruff's `D` rules. Every public module, class, function and method in `src/scenetiq` needs a docstring. Tests are exempt.

### Layout

- `src/scenetiq/`: first-party package (pipeline library and HTTP API).
- `src/scenetiq/vendor/`: vendored third-party code. Excluded from ruff and ty. Every subdirectory needs a `PROVENANCE.md` with the upstream URL, pinned commit, licence and local modifications. First-party wrappers around vendored code live outside `vendor/` and are fully checked.
- `tests/unit/`: pure functions (coordinate and unit adapters, geometry, metrics).
- `tests/api/`: **Seam A**. The HTTP API driven through FastAPI's TestClient, with a fake Pipeline and a fake text encoder on a temporary storage root. This is the main suite and must pass on CPU.
- `tests/eval/`: **Seam B**. The real Pipeline on a GPU, marked `@pytest.mark.gpu` and deselected by default.

Test only at these seams. Do not test internal stage functions directly.

### Definition of Done (every ticket)

1. `uv run ruff check .` and `uv run ruff format --check .` pass.
2. `uv run ty check` passes with zero diagnostics.
3. Every new public first-party symbol has a Google-style docstring.
4. Tests are added at the right seam, and `uv run pytest` passes.
5. AGENTS.md is updated whenever the ticket adds a command, directory, convention or configuration value that a future agent needs to know.
