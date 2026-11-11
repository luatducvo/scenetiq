# SceneTIQ

Turns a set of 2D photos of a static scene into a 3D point cloud whose objects can be found by free-text queries.

## Input

**Scene**:
The physical space the user photographed. Must be static (nothing moves between photos).
_Avoid_: Room, environment

**Capture**:
The photos uploaded together for one Scene, in upload order, with no camera poses or intrinsics. In v1 one Scene has exactly one Capture.
_Avoid_: Upload, dataset, batch

**View**:
One photo within a Capture, together with the camera it was taken from once reconstructed.
_Avoid_: Frame, image (when the camera matters)

**View stride**:
The spacing N at which a Capture's Views are reconstructed: the first View and every Nth one after it, in upload order. The other Views are kept but unused.
_Avoid_: Frame skip, sampling rate

## Reconstruction

**Reconstruction**:
The 3D result of one Capture: a Gravity-aligned point cloud of the Scene plus the cameras of every View the View stride selected.
_Avoid_: Model, 3D scan, map

**Gravity alignment**:
The rotation that places a Reconstruction so that the Scene's physical up matches the conventional up direction. It says nothing about which way the Scene faces horizontally.
_Avoid_: Leveling, calibration, upright, orientation

**Mask**:
A candidate object region in a single View, produced without any class label. Masks are inputs to Instances, never shown to the user as results.
_Avoid_: Segment, instance (for 2D regions)

**Instance**:
One distinct physical object in a Reconstruction: the set of points that belong to it, with its position and 3D bounding box. Instances carry no class from a fixed list; they may have a Caption.
_Avoid_: Object, segment, mask, cluster

**Caption**:
What a vision-language model reads from an Instance's Views: a Name, up to three Synonyms, and one Description for each Aspect. Not drawn from a fixed class list. Only the Name is shown to the user.
_Avoid_: Label, class, tag, annotation

**Aspect**:
One of the fixed sides of an Instance that a Caption describes: Identity (what it is and what else it is called), Appearance (colour, shape, relative size), Material (what it is made of, its parts) and Use (what it is for).
_Avoid_: Facet, field, attribute

**Description**:
One short sentence of a Caption about one Aspect of an Instance. It may be empty when the Views do not show that Aspect.
_Avoid_: Sentence, caption (for a single sentence)

## Querying

**Query**:
A free-text description from the user (e.g. "red chair") asked against one Reconstruction.
_Avoid_: Search, prompt, question

**Match**:
One of the top-K Instances for a Query, ranked by fusing image similarity with the similarity of its best-matching Description; carries a normalised fused score. For a Query with a Relation it is ranked instead by how well its combination with the Anchors satisfies the Relations, and carries those Anchors. A Query on a Reconstruction with Instances always has at least one Match.
_Avoid_: Hit, result, detection, low-confidence match

**Relation**:
A spatial relationship between Instances that a Query states, such as near, on or between. The set is closed; each Relation takes a fixed number of Anchors and is scored from stored bounding boxes and centroids.
_Avoid_: Predicate, spatial filter, edge

**Anchor**:
An Instance a Query uses as a reference to single out the Match (the window in "the chair near the window"). A Match carries its Anchors in the API's answer; the viewer does not show them.
_Avoid_: Reference object, landmark, context object

**Spatial reasoning**:
Answering Queries about Relations between Instances. The first slice covers Relations that do not depend on where the user stands: above, below, on, under, near, far from, between, nearest to, farthest from, largest, smallest, tallest and lowest. Left, right, in front of, behind, negation, counting and ordinals are not covered.
