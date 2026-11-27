# Backend runs on Google Colab, reached through a Cloudflare named tunnel

The pipeline needs a large GPU (DA3 Giant on hundreds of Views, SAM2, SigLIP2). We run the FastAPI backend on Google Colab (80GB VRAM). The React frontend runs on localhost and connects through a Cloudflare named tunnel, so the URL stays stable across Colab restarts.

## Consequences

- Colab is ephemeral. All Scene data and job state (SQLite) live on the runtime's local disk, chosen over mounted Google Drive so that uploads and pipeline I/O are not limited by Drive. Scenes are lost when the runtime disconnects or is deleted, and must be uploaded again.
- Google Drive is not mounted at all. The API log and job progress appear only in the output of the notebook's last cell, so they are lost with the runtime too.
- If only the server process restarts, the local disk remains: finished Scenes are reloaded, and in-flight jobs are marked failed rather than resumed.
- The tunnel exposes the backend publicly. Every request must carry a shared token (stored as a Colab secret). v1 is single-user, with no accounts.
- One job runs at a time.
