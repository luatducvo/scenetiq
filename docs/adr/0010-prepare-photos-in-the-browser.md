# Orient, shrink and strip photos in the browser before upload

Phone photos are 3–8 MB each and carry metadata that no stage reads: GPS position, camera maker notes, embedded preview pictures. Every byte goes through the Cloudflare tunnel (ADR 0004, ADR 0006), which is the slowest part of a Capture's upload. A portrait photo is often stored as landscape pixels plus an EXIF orientation flag, and no stage applies the flag, so such a photo reached every stage sideways.

The frontend now prepares each photo before sending it:

1. It decodes the photo with the EXIF orientation applied to the pixels.
2. It shrinks the photo so that its long side is at most `upload_max_side` pixels (2048 by default), never enlarging it.
3. It re-encodes the photo as JPEG at `upload_jpeg_quality` (0.9 by default). A canvas export carries no metadata, so this drops every field.

It always re-encodes, even when the result is larger than the original, so that every photo is upright and stripped. The upload form shows the photos, and their sizes before and after, before anything is sent. Both settings live in `scenetiq.config` and reach the frontend through `GET /options`.

2048 px is a trade-off. DA3 reconstructs at 504 px, but the Embedding and Captioning stages crop each Instance from the uploaded photo, so a small Instance in a wide Scene needs the extra pixels to stay legible.

## Considered Options

- **Upload the originals and shrink them on the backend:** still sends every byte through the tunnel.
- **Strip metadata without re-encoding**, by rewriting the JPEG segments: keeps full resolution and quality, but saves little and needs a JPEG parser in the browser. It would also still have to apply the orientation.
- **Shrink harder (1600 px or less):** a smaller upload, but small Instances get fewer pixels in their crops.

## Consequences

- A Capture uploads several times faster from a phone, and the user's location never leaves the browser.
- Portrait photos reach every stage upright, but only through the frontend. A client that calls the API directly still sends raw photos, and the backend still applies no orientation.
- Photos are stored at the prepared size. A Scene uploaded before this change keeps its full-size photos.
- PNG input becomes JPEG, and transparency is filled white.
