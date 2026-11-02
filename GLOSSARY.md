# SceneTIQ

Turns a set of 2D photos of a static scene into a 3D point cloud whose objects can be found by free-text queries.

## Input

**Scene**:
The physical space the user photographed. Must be static (nothing moves between photos).
_Avoid_: Room, environment

**Capture**:
The set of 10–300 unordered photos uploaded together for one Scene, with no camera poses or intrinsics. In v1 one Scene has exactly one Capture.
_Avoid_: Upload, dataset, batch

**View**:
One photo within a Capture, together with the camera it was taken from once reconstructed.
_Avoid_: Frame, image (when the camera matters)

## Reconstruction

**Reconstruction**:
The 3D result of one Capture: a point cloud of the Scene plus the cameras of every View.
_Avoid_: Model, 3D scan, map

**Mask**:
A candidate object region in a single View, produced without any class label. Masks are inputs to Instances, never shown to the user as results.
_Avoid_: Segment, instance (for 2D regions)

**Instance**:
One distinct physical object in a Reconstruction: the set of points that belong to it, with its position and 3D bounding box. Instances carry no class label.
_Avoid_: Object, segment, mask, cluster

## Querying

**Query**:
A free-text description from the user (e.g. "red chair") asked against one Reconstruction.
_Avoid_: Search, prompt, question

**Match**:
An Instance returned for a Query, with a relevance score. A Query may have no Matches.
_Avoid_: Hit, result, detection

**Low-confidence Match**:
The single closest Instance returned when a Query has no Matches, explicitly flagged as below the relevance bar.
_Avoid_: Fallback, best guess

**Spatial reasoning**:
Answering Queries about relations between Instances (e.g. "the cup left of the laptop"). Out of scope for v1.
