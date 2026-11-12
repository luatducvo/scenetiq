# SceneTIQ

Upload photos of a static scene to get a metric 3D point cloud of it. Then find objects in the scene by describing them in free text, for example "red chair".

> The GPU pipeline has not been run end to end yet. Expect to fix issues on the first Colab run.

## Before you start

1. Push this repo to GitHub. The notebook clones `luatducvo/scenetiq@main`, so the repo must be public.
2. In Cloudflare Zero Trust, open **Networks → Tunnels → Create → Cloudflared**, add a public hostname (for example `scenetiq.example.com`) pointing to `http://localhost:8001`, and copy the tunnel token.
3. In Colab, add two secrets (🔑 sidebar) and give the notebook access to both:
   - `SCENETIQ_TOKEN`: any long random string.
   - `CLOUDFLARE_TUNNEL_TOKEN`: the token from step 2.

## 1. Start the backend

1. Open the notebook in Colab and pick an GPU runtime:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luatducvo/scenetiq/blob/main/notebooks/colab_server.ipynb)

   The link always loads the latest notebook from `main`. Edits made in Colab are not saved back, so change parameters in the repo instead. Do not use "Save a copy in Drive": that copy stops following the repo.
2. In the first code cell, edit `CONFIG`. Use `"pipeline": "fake"` for a first run to check the tunnel and the frontend without models, and `"real"` for real reconstruction. `view_stride` reconstructs only every Nth photo, in upload order. See [Configuration](docs/configuration.md) for every setting.
3. Run all cells. The last cell keeps running and shows the API log, with progress for the photo upload and the current job. Stopping it does not stop the backend, and running the cell again resumes the display.

Model weights download the first time each session uses them (about 15 GB), so the first job stays on `reconstructing` for several minutes.

**Scenes are stored on the runtime's local disk, and logs are not saved anywhere. When the runtime disconnects or is deleted, all Scenes are lost:** run the notebook again and upload them again.

## 2. Open the frontend

1. Copy `frontend/.env.example` to `frontend/.env`, then set:
   - `VITE_BACKEND_URL` to `https://<your tunnel hostname>`
   - `VITE_BACKEND_TOKEN` to your `SCENETIQ_TOKEN`
2. Start the frontend, and restart it after any change to `.env`:

   ```bash
   npm --prefix frontend install
   npm --prefix frontend run dev
   ```

3. Open http://localhost:3000.

Vite compiles these values into the JavaScript bundle, so anyone who can load a build can read the token. Do not host a build publicly.

The header shows the backend's status from when the page loaded. Click **Ping** to check again. **Offline** usually means the Colab runtime or the tunnel has stopped.

## 3. Create a Scene

1. Click **New Scene**, name the Scene, and select JPG/PNG photos. The number of photos must be between `min_views` and `max_views` in `CONFIG`.
2. Check the preview. Each photo is shown with its size before and after processing. The browser turns each photo upright, shrinks it to at most `upload_max_side` px (2048 by default), and removes metadata such as GPS. Remove any photo with ×.
3. Click **Build 3D scene**. Only one Scene is processed at a time.

**Photo tips:**
- Nothing in the scene should move between photos.
- Walk around the space instead of turning on the spot.
- Keep plenty of overlap between photos, use even light, and avoid blur.

## 4. Follow the processing

The Processing page shows each stage, the time it took, and the total. Notes below the stages report how many Views were reconstructed, how many objects were found and named, and whether the Scene could not be levelled. If it could not, "above", "on" and similar Queries may be wrong.

To stop a job, click **Cancel job** (on the Processing page; you can also cancel from the Scene list). It stops at the next checkpoint (the end of the current stage, photo or object). A model call already running is not interrupted. The job's photos are then deleted, and the Scene stays in the list as cancelled until you delete it.

## 5. View and search a Scene

- The Scene fills the browser window. Click **Full screen** to use the whole display (Esc to leave), and **← Scenes** to go back.
- Navigate with the mouse: drag to orbit, right-drag to pan, scroll to zoom.
- **Points** sets the point size. The browser remembers the setting.

### Search with a Query

Type a Query to colour the matching objects with a heatmap:

- The best Match is red. A Match is cooler the further its score is below the best one's. The rest of the Scene keeps its own colours.
- A score compares a Match with the best one for the same Query. It says nothing absolute.
- Only objects that score close enough to the best are returned, so a Query can return fewer objects than you asked for. A Query whose object is not in the Scene (a negative Query) returns nothing.
- **Results to show** (in Display) sets the most Matches returned, and re-runs the Query.
- The key in the bottom-right corner shows the colour scale.
- Hover over a coloured object (tap on a touch screen) to see its name. The names are read from the photos by a vision-language model.
- A Query in English or Vietnamese can also refer to an object by position, such as "the box on the shelf", "the chair nearest the window" or "the largest table".
- Each Query waits for the vision-language model to read it. The first Query after the backend starts also waits for that model to load.

### Check the object split with Instance labels

Turn on **Instance labels** to give each object its own colour and check how the Scene was split and named. Hover over an object (tap on a touch screen) to see its name beside the pointer. Black objects have no name, and grey points belong to no object. Clear any Query first, because its colours take priority.

## 6. Manage Scenes

In the **Scenes** list, open a processed Scene to view it. To delete one, use **Delete scene** in its card's menu; this cannot be undone. Each finished Scene shows its number of objects (Instances) and how long it took to process.
