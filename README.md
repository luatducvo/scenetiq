# SceneTIQ

Upload photos of a static scene to get a metric 3D point cloud of it. Then find objects in the scene by describing them in free text, for example "red chair".

> The GPU pipeline has not been run end to end yet. Expect to fix issues on the first Colab run.

## 1. One-time setup

1. **Push this repo to GitHub.** The notebook clones `luatducvo/scenetiq@main`, so the repo must be public or the clone step adapted to use a token.
2. **Create a Cloudflare named tunnel:** in Zero Trust, open **Networks → Tunnels → Create → Cloudflared**, add a public hostname (for example `scenetiq.example.com`) pointing to `http://localhost:8001`, and copy the tunnel token.
3. **Add two Colab secrets** (🔑 sidebar), and give the notebook access to both:
   - `SCENETIQ_TOKEN`: any long random string.
   - `CLOUDFLARE_TUNNEL_TOKEN`: the token from step 2.

## 2. Start the backend on Colab

1. Open the notebook in Colab and pick an **A100 80 GB** GPU runtime:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luatducvo/scenetiq/blob/main/notebooks/colab_server.ipynb)

   This link always loads the latest `notebooks/colab_server.ipynb` from `main`, so a push needs no manual copy. Edits made in Colab are not saved back; change parameters in the repo instead. Do not use a "Save a copy in Drive" version: it stops following the repo.
2. Edit `CONFIG` in the first code cell. Use `"pipeline": "fake"` on your first run to check the tunnel and the frontend without models, and `"real"` for real reconstruction. `view_stride` reconstructs only every Nth photo, in upload order. `CONFIG` overrides `src/scenetiq/config.py`, which lists every setting and its default.
3. Run all cells. The last cell keeps running and shows the API log, with a progress bar for the photo upload and the job being processed. Stop it if you need to run another cell; the backend keeps going, and running the cell again resumes the display.

Model weights download automatically the first time each session uses them (about 15 GB), so the first job stays on `reconstructing` for several minutes. Scenes are saved on the runtime's local disk (`/content/scenetiq-data`), and logs are not saved anywhere. **When the runtime disconnects or is deleted, all Scenes are lost:** run the notebook again and upload them again.

## 3. Open the frontend

1. Copy `frontend/.env.example` to `frontend/.env`, then set `VITE_BACKEND_URL` to `https://<your tunnel hostname>` and `VITE_BACKEND_TOKEN` to your `SCENETIQ_TOKEN`.
2. Start the frontend. Restart it after any change to `.env`.

   ```bash
   npm --prefix frontend install
   npm --prefix frontend run dev
   ```
3. Open http://localhost:3000.

Vite compiles these values into the JavaScript bundle, so anyone who can load a build can read the token. Do not host a build publicly.

## 4. Use it

The header shows the backend's status from when the page loaded; click **Ping** to check again. **Offline** usually means the Colab runtime or the tunnel has stopped.

1. **New Scene:** name the Scene and select JPG/PNG photos, between `min_views` and `max_views` of `CONFIG`.
2. **Processing:** watch the progress through each stage. Only one Scene is processed at a time.
3. **Viewer:**
   - Drag to orbit, right-drag to pan, scroll to zoom.
   - Type a Query to highlight and box the matching objects.
   - A **low confidence** badge means nothing matched well; you are seeing the closest object.
4. **Scenes:** reopen or delete processed Scenes.

**Photo tips:**
- Nothing in the scene should move between photos.
- Walk around the space instead of turning on the spot.
- Keep plenty of overlap between photos, use even light, and avoid blur.

## Try it locally without a GPU

The fake pipeline returns a small fixed scene. It ignores the uploaded photos and answers `red chair`, `table`, `lamp` and `chair`.

```bash
SCENETIQ_TOKEN=dev uv run uvicorn --factory scenetiq.server:build_app --port 8001
```

Run the frontend as in step 3, using `VITE_BACKEND_URL=http://localhost:8001` and `VITE_BACKEND_TOKEN=dev`.

## License note

The Depth Anything 3 weights used here are licensed CC BY-NC 4.0, so the real pipeline is for **non-commercial use only**.
