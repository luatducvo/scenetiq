# 02: Upload preview and photo preparation in the browser

**What to build:** After choosing photos, the user sees a thumbnail grid before anything is sent. The browser prepares every photo:

- it applies the EXIF orientation to the pixels;
- it shrinks the photo so that its long side is at most `upload_max_side` (default 2048), never enlarging it;
- it re-encodes it as JPEG at `upload_jpeg_quality` (default 0.9), which drops all metadata.

The grid shows the size before and after, for each photo and in total. The user can remove single photos, and a warning appears while the count is outside the View limits. Only the prepared photos are uploaded, in grid order. The frontend reads the settings from a new `GET /options`.

Spec: *Upload preview and photo preparation* user stories; *Configuration*, *HTTP API* (`GET /options`), *Frontend* (photo preparation, upload form) and *Documents* (ADR 0010).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Settings gain `upload_max_side` (at least 1) and `upload_jpeg_quality` (in (0, 1]). Out-of-range values are refused, with unit tests next to the existing config tests.
- [ ] `GET /options` requires the token and returns `min_views`, `max_views`, `upload_max_side` and `upload_jpeg_quality`. A Seam A test covers the defaults and an override.
- [ ] Photo preparation is a pure frontend module. It returns the prepared `.jpg` file, a thumbnail URL of about 160 px, both byte sizes and the pixel size. It processes a few files at a time with progress, and always re-encodes.
- [ ] A file that cannot be decoded is reported by name and left out.
- [ ] Thumbnail URLs are revoked when a photo is removed or the form unmounts.
- [ ] The upload form shows the grid with a remove button on each photo, the totals before and after, the preparation progress, and a warning while the count is outside `min_views`–`max_views`.
- [ ] The upload button is disabled while preparing, while the count is out of range, or with no photos.
- [ ] If `GET /options` fails, the form shows the error and cannot upload.
- [ ] The upload sends the prepared files in grid order. Upload progress and retries behave as before.
- [ ] ADR 0010 records the decision. AGENTS.md updates the EXIF constraint: the frontend applies orientation, the backend does not. The configuration docs list both settings, and the README describes the preview.
- [ ] The frontend passes lint, typecheck and build.
- [ ] Manual check against the local fake backend:
  - a portrait phone photo shows upright in the grid;
  - a large JPEG comes out at most 2048 px on its long side and smaller in bytes;
  - removing a photo updates the count and the totals;
  - a non-image file renamed `.jpg` is reported and skipped.
