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
| Run backend locally | `SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8000` |
| Frontend install | `npm --prefix frontend install` |
| Frontend dev server | `npm --prefix frontend run dev` (http://localhost:5173) |
| Frontend lint | `npm --prefix frontend run lint` |
| Frontend type check | `npm --prefix frontend run typecheck` |
| Frontend build | `npm --prefix frontend run build` |

### Configuration

The backend reads environment variables (see `scenetiq.config.Settings`):

| Variable | Default | Meaning |
|---|---|---|
| `SCENETIQ_TOKEN` | (required) | Shared bearer token every request must send |
| `SCENETIQ_STORAGE_ROOT` | `./data` | SQLite database plus one directory per Scene |
| `SCENETIQ_MATCH_THRESHOLD` | `0.5` | Minimum relevance score for a Match |
| `SCENETIQ_PIPELINE` | `fake` | `fake` (no models) or `real` (GPU model stack) |
| `SCENETIQ_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated browser origins |

### Docstrings

Google convention, enforced by ruff's `D` rules. Every public module, class, function and method in `src/scenetiq` needs a docstring. Tests are exempt.

### Layout

- `src/scenetiq/`: first-party package.
  - `domain.py`: the Capture, View, Camera, Instance, Reconstruction and Stage types, plus the `Pipeline` and `TextEncoder` protocols.
  - `api.py`: the FastAPI app factory `create_app(settings, pipeline, text_encoder)`.
  - `server.py`: the process entry point, which reads env config and picks the Pipeline.
  - `storage.py`: SQLite metadata and per-Scene directories.
  - `artifacts.py`: the on-disk Reconstruction format.
  - `jobs.py`: the single-slot background job runner.
  - `fakes.py`: the deterministic fake Pipeline and text encoder.
  - `query.py`: scores Instances against a Query embedding, applies the Match threshold, and returns the Low-confidence Match fallback.
  - `geometry.py`: Instance centroid and bounding-box computation, plus L2 normalisation.
- `frontend/`: React + Vite + TypeScript client. It has no router; `App.tsx` switches between screens.
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
5. If the ticket touches `frontend/`: `npm --prefix frontend run lint`, `run typecheck` and `run build` pass.
6. AGENTS.md is updated whenever the ticket adds a command, directory, convention or configuration value that a future agent needs to know.

### Colab (backend deployment, ADR 0004)

`notebooks/colab_server.ipynb` is the only way the backend is deployed. It holds parameters and calls into `scenetiq.colab`; put any new logic in the package, not the notebook. Notebooks are excluded from ruff and ty.

One-time setup:
1. In the Cloudflare Zero Trust dashboard, create a named tunnel (Networks → Tunnels → Create → Cloudflared). Add a public hostname (e.g. `scenetiq.<your-domain>`) whose service is `http://localhost:8000`. Copy the tunnel token.
2. In Colab, add the secrets `SCENETIQ_TOKEN` (any long random string, also entered in the frontend Settings) and `CLOUDFLARE_TUNNEL_TOKEN`, and grant the notebook access to both.
3. Pick a GPU runtime.

Each run: open the notebook, set `PIPELINE` (`fake` or `real`) and run all cells. Scenes, the SQLite database and logs live under `DRIVE_ROOT` on Google Drive, so they survive runtime restarts. Jobs that were running when the runtime died are marked failed at the next start. Then run the frontend locally with the backend URL set to `https://<tunnel hostname>`. Add the frontend's origin to `SCENETIQ_CORS_ORIGINS` if it is not on `localhost:5173`.
