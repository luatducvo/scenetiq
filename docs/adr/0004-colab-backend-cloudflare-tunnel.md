# Backend runs on Google Colab, reached through a Cloudflare named tunnel

The pipeline needs a large GPU (DA3 Giant on up to 300 Views, SAM2, SigLIP2). We run the FastAPI backend on Google Colab (80GB VRAM). The React frontend runs on localhost and connects through a Cloudflare named tunnel, so the URL stays stable across Colab restarts.

## Consequences

- Colab is ephemeral. All Scene data and job state persist on mounted Google Drive (SQLite for jobs). On restart, finished Scenes are reloaded, and in-flight jobs are marked failed rather than resumed.
- The tunnel exposes the backend publicly. Every request must carry a shared token (stored as a Colab secret). v1 is single-user, with no accounts.
- One job runs at a time.
