# The backend runs on one Colab runtime

Postgres, scene-api and scene-worker used to run in Docker Compose on the user's machine, with only scene-compute on Google Colab behind a Cloudflare Tunnel. Now every backend process runs on one Colab runtime, started by `backend/deploy/colab/GPU.ipynb`: Postgres 16 with pgvector installed from the PostgreSQL apt repository, scene-compute, scene-api and scene-worker. The tunnel exposes scene-api alone, and the user's machine runs only the frontend's `npm run dev`. We made this change so the machine needs no Docker to run the app. Colab has no Docker and its quick tunnel carries only HTTP, so the database could not move without the processes that use it.

## Consequences

- Nothing outlives the runtime. The notebook's backend cell creates an empty database and empties `/content/data` on every run, so Scenes, Jobs, Objects and Publications last one session.
- scene-api is public with no authentication; the tunnel URL is random and changes every session, and the frontend's `VITE_API_URL` changes with it.
- The `colab` Target is scene-compute on `http://127.0.0.1:9001`, with no token. scene-compute stays a separate process so the worker's Stages and the API's Search share one set of loaded models instead of each loading the embed models.
- Compose, the Dockerfiles, the `local_gpu` Target, `pipeline.fake-gpu.yaml` and the `main.py` commands that drove them are gone. Docker remains only for the disposable Postgres of `uv run main.py test`. Without a GPU, the frontend's mock mode and the tests are the only way to run the system.

## Considered Options

- Moving only Postgres to Colab, with the API and worker still local: rejected because the quick tunnel cannot carry Postgres's protocol, and every query would cross the internet.
- A hosted Postgres with pgvector (for example Supabase or Neon): it would keep Scenes between sessions, but we accepted losing them to keep everything in one place with no extra account.
- Running Algorithms in-process in the worker and the API: rejected because both processes would load the embed models, doubling their VRAM.
- Keeping the Compose backend beside Colab for local development with Fake Algorithms: rejected to keep a single way to run the backend.
