# 3D Scene Understanding

Turns photos or one short video of a room into a metric 3D Scene whose Objects can be searched and inspected: a Python backend in `backend/` and a React frontend in `frontend/`.

## Tooling

- Python is one uv workspace rooted at the repository root. Run backend commands from the root as `uv run main.py <command>`, never from `backend/`; `uv run main.py --help` lists them.
- The frontend uses npm, run inside `frontend/`.
- The backend runs only on a Google Colab runtime, started by `backend/deploy/colab/GPU.ipynb`; locally you run the tests, the linters and the frontend.

| Command | Does |
| --- | --- |
| `uv run main.py test` | every pytest test, against a disposable Postgres container (needs Docker) |
| `uv run main.py test-unit` | the tests that need no database or Docker |
| `uv run main.py lint` / `format` | Ruff check / format |
| `uv run main.py openapi` | regenerate `backend/openapi/openapi.json` from the API |
| `npm run typecheck` / `test` / `build` | frontend checks |
| `npm run api:sync` | copy the backend's OpenAPI contract into the frontend and regenerate its types |

## Always

- Name domain concepts with the terms in [GLOSSARY.md](GLOSSARY.md) and never with their _Avoid_ words.
- Write all repository text in English (code, comments, docs, commit messages), whatever language the conversation uses.

## Read when relevant

- Finishing a change, or writing tests: [docs/agents/testing.md](docs/agents/testing.md)
- Raising, handling or showing an error: [docs/agents/failures.md](docs/agents/failures.md)
- Changing an API route or model, a file format, or a constant both sides share: [docs/agents/api-contract.md](docs/agents/api-contract.md)
- Choosing an Algorithm or Target, or adding an environment variable: [docs/agents/configuration.md](docs/agents/configuration.md)
- Committing: [docs/agents/git-workflow.md](docs/agents/git-workflow.md)
- How the system fits together, and where code lives: [docs/architecture/overview.md](docs/architecture/overview.md)
- Adding an Algorithm, viewer layer, endpoint or Target: [docs/extending.md](docs/extending.md)

`backend/` and `frontend/` each have an AGENTS.md with their own rules.

## Agent skills

- **Issue tracker**: local markdown files under `.scratch/<feature>/`. See [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md).
- **Triage labels**: the five default roles, on a `Status:` line in each issue file. See [docs/agents/triage-labels.md](docs/agents/triage-labels.md).
- **Domain docs**: single context, `GLOSSARY.md` plus `docs/adr/`. See [docs/agents/domain.md](docs/agents/domain.md).
