# 03: Viewer cleanup: no boxes or Anchors, Names only, point size, Names view

**What to build:** A cleaner viewer.

- Matches are no longer boxed or labelled in 3D, and Anchors are no longer coloured, boxed or listed. Spatial Queries still work.
- The Match list shows only each Match's Name, best first, or "Instance <id>" without a Caption. It shows no score.
- The notice line about the unsupported parts of a Query stays.
- A slider scales the point size relative to the automatic size, and the browser remembers the choice.
- The Instances toggle becomes a Names view:
  - every Instance with a Name keeps its Instance colour and shows its Name at its centroid;
  - an Instance without a Caption is black;
  - points of no Instance stay grey;
  - it is disabled while a Query result is shown, with a hint.

Until ticket 04, Match points keep their current rank colours. Every other point keeps its original colour, no longer dimmed. Spec: *Point size* and *Names view* user stories; stories 20–27; *Frontend* (viewer, point size slider, Names view).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] No Box3 helpers, floating labels or Anchor colour, label or list entry remain in the viewer.
- [ ] Points outside the Matches keep their original colour, undimmed, while a result is shown.
- [ ] The Match list shows a colour swatch and the Name, or "Instance <id>", with no score. The Query notice line still appears when relevant.
- [ ] A point size slider (about 0.25× to 4× of the automatic size) changes the rendered size live. Its value persists in local storage, and reading or writing storage is wrapped so a throwing storage does not break the viewer.
- [ ] The Names view colours named Instances by Instance colour, draws their Names at their centroids, colours unnamed Instances black and leaves no-Instance points grey.
- [ ] The Names view is disabled while a result is shown.
- [ ] The README usage section no longer mentions boxes or Anchor colours, and describes the point size slider and the Names view.
- [ ] The frontend passes lint, typecheck and build.
- [ ] Manual check against the local fake backend:
  - `red chair` highlights the chair with no box;
  - the Names view shows "chair" and "table" and a black lamp, which has no Caption;
  - the point size survives a page reload.
