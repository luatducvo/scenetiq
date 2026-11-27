# 04: Show each Match's Name in the viewer

**What to build:** The user sees what the system thinks each Match is: its Name next to its score in the Match list, and as a label on its 3D bounding box. A Match without a Caption shows no Name. Synonyms, Attributes and Use are never shown or exposed. See "HTTP API" and "Frontend" in `.scratch/instance-captioning/spec.md`.

**Blocked by:** 03: Store Captions and rank Queries against them

**Status:** ready-for-agent

- [x] The Instance listing returns `name` per Instance (string or null) and nothing else from the Caption.
- [x] Seam A tests: `name` for a captioned Instance; null for one without a Caption and for a Scene without Captions.
- [x] The Match list shows the Name beside the score when present.
- [x] Each Match's bounding box carries its Name as a readable label that follows the box.
- [ ] Frontend lint, typecheck and build pass (done), and the labels are checked in the browser against the fake Pipeline (not done: the browser pane could not open localhost).
