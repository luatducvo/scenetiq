# Orient, shrink and strip photos in the browser before upload

Every byte of a Capture crosses the Cloudflare tunnel (ADR 0003), and phone photos are 3–8 MB with metadata no stage reads (GPS, maker notes, previews). Portrait photos are often landscape pixels plus an EXIF orientation flag, which no stage applies. So `frontend/src/photos.ts` decodes each photo with its EXIF orientation applied, shrinks its long side to at most `upload_max_side` (2048) without enlarging, and re-encodes it as JPEG at `upload_jpeg_quality` (0.9), which drops all metadata. It always re-encodes, so every photo is upright and stripped. Both settings live in `scenetiq.config` and reach the frontend through `GET /options`.

2048 px rather than DA3's 504 px, because the Embedding and Captioning stages crop Instances from the uploaded photo.

## Considered Options

- Shrinking on the backend: every byte still crosses the tunnel.
- Stripping metadata without re-encoding: saves little, needs a JPEG parser in the browser and still has to apply the orientation.

## Consequences

- Only the frontend applies EXIF orientation. A client calling the API directly sends raw photos, and every stage reads them sideways.
- Photos are stored at the prepared size; PNG becomes JPEG with transparency filled white.
