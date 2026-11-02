# Frontend

`frontend/`: React SPA with Vite and React Router, TanStack Query for server state, Zustand for viewer state, react-three-fiber for 3D, use-intl for strings. English is the only locale.

## Routes

| Route | File | Shows |
| --- | --- | --- |
| `/` | `src/App.tsx` | redirects to `/en` |
| `/en` | `src/pages/SceneList.tsx` | the Scene library |
| `/en/scenes/new` | `src/pages/NewScene.tsx` | name a Scene, add files, upload |
| `/en/scenes/{id}` | `src/pages/ScenePage.tsx` | depends on the Scene's status (below) |

The Scene page shows:

- `created` or `uploaded`: `UploadDropzone` (more files) and, once uploaded, a button to process;
- `processing` or `failed`: `StageStepper` with live progress, and on failure the failure and a link to start a new Scene;
- before `ready`, once ingest has run: the `IngestReport` of kept and rejected Frames;
- `ready`: `SceneExplorer`.

## Data flow

- `sceneApi` (`src/lib/api/http.ts`) is the only way to the backend: typed `openapi-fetch` calls, plus hand-written upload (XMLHttpRequest, for upload progress) and SSE readers (`sse.ts`).
- Uploading first shrinks photos to `UPLOAD_LONG_SIDE` in the browser (`src/lib/upload/shrink.ts`); `UploadDropzone` scores each photo's sharpness and exposure in a worker (`src/lib/quality/`) to warn before upload.
- `useSceneProgress` (`src/lib/sceneProgress.ts`) seeds per-Stage state from the Scene's Stage Runs, follows `GET /events` while the Scene processes, reconnects on a dropped stream, and refetches the Scene on a terminal status.
- Query keys: `["scenes"]`, `["scene", id]`, `["objects", id]`, plus the membership file.

## Scene Explorer

`src/components/SceneExplorer.tsx` holds the 3D canvas, a topbar (the Points/Mesh switch with point size in Points mode, Instance Colors, the Search box) and a side panel that shows one of: `object` (`ObjectDetails` for the Selection), `measure` or `info`. Each panel opens from its spatial-tools button; the `object` one only while there is a Selection, which it follows, and it closes when the Selection is dropped.

- The viewer is in one interaction at a time (`interaction` in the viewer store): idle (hover paints the Object under the pointer), Search (while the Search text is not blank) or Selection. The canvas reports hovers and clicks to the store, which applies that interaction's rules; `clickable` says which Objects a click reaches, and drives the cursor.
- A Search Highlights its hits and colours them as an Object Heatmap (`HeatmapLegend`). `showResult` drops a Search response whose version is stale, so a slow Search cannot overwrite a newer one or a Selection made meanwhile.
- The camera moves only through `goHome` (whole room, clears Search, Selection and Highlight) and `focus(ids)` in `viewer/camera.ts`. A Search focuses its hits; a Selection does not move the camera, and is labelled with its Object's name on top of its box.

## Viewer

`src/components/viewer/`:

| File | Holds |
| --- | --- |
| `SceneCanvas.tsx` | the react-three-fiber canvas, controls, picking, the cursor, hover and Selection labels; renders every entry of `sceneLayers` |
| `layers/index.tsx` | `sceneLayers`: `points` (`PointCloudLayer`, `points.ply`), `mesh` (`MeshLayer`, `mesh.glb`) and the metric floor `grid` (`FloorGridLayer`); repaints vertex colours for Highlight, hover and Instance Colors |
| `view.ts` | which layer shows, room bounds, home and focus poses, where a Selection's label sits, whether highlighting is available |
| `highlight.ts` | which Object owns each point (the smallest one), heatmap and dim colours, Instance Colors for hover and Selection, heatmap normalisation |
| `picking.ts` | a hidden twin of each layer coloured by Object id, read back to pick the Object under the pointer |
| `camera.ts` | `goHome`, `focus` and the driver the canvas attaches |

`src/lib/store/viewerStore.ts` holds viewer state: the Search text, Selection, Highlight, similarities, hover, Measurement points, layers, appearance, loaded bounds.

Object Membership (`src/lib/api/membership.ts`) maps each Object to its points and vertices; the viewer refuses to highlight when the membership is absent or its counts do not match the loaded geometry.

## Mock mode

With `VITE_API_MODE=mock`, `src/Providers.tsx` starts MSW (`public/mockServiceWorker.js`) before rendering. `src/lib/api/mock/handlers.ts` serves every API path, including the SSE progress stream, with a demo Scene built in `mock/room.ts` and its membership in `mock/membership.ts`.

## Build and deployment

`src/main.tsx` mounts `App` inside React Router. The routes and Three.js canvas are lazy-loaded. `use-intl` reads the existing English message catalog. The visual style uses a warm canvas, navy typography, teal controls and a metric floor grid registered in `sceneLayers`.

Vite serves development on port 3000 and builds static files into `dist/`. `VITE_API_URL` and `VITE_API_MODE` are browser build-time variables; restart `npm run dev` or rebuild after changing them. There is no server-side API origin.

Copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_URL` to the scene-api URL the Colab notebook prints; it changes with every tunnel.

The editable desktop design is indexed in [the complete desktop screen set](https://www.figma.com/design/s0m5M8f2bvU2JaUEuNGnhH/Untitled?node-id=66-2533). Static capture illustrations and Scene previews have been removed; the viewer loads Publication geometry, and uploaded Frame previews come only from the selected files.

## Fixed viewport and application shell

`AppShell` owns the viewport (`100dvh`), the route slot and the shared navigation. Each route supplies its Header and main content. Header and navigation stay visible while the content changes. The desktop shell has a minimum width of 1024 pixels; there is no separate mobile layout.

The document does not scroll vertically. The library reserves rows for its introduction and filters, then scrolls the Scene collection. Capture and processing screens reserve their heading and give each content panel its own scroll region. In the explorer, the toolbar, canvas, spatial tools and hints occupy separate grid rows; the canvas gets the remaining height and the inspector scrolls independently. All shrinking grid boundaries use `min-height: 0` so long content cannot push the navigation out of the viewport. Native dialogs scroll within the viewport as well.
