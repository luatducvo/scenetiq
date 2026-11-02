# 3D Scene Understanding

Turns RGB photos or one short video of a room into a metric 3D Scene whose Objects can be searched and inspected.

## Language

### Capture

**Scene**:
One captured room and everything understood from its capture.
_Avoid_: room (for the record), project

**Frame**:
One image of a capture: an uploaded photo or a moment sampled from the video.
_Avoid_: photo, picture, image

**Keyframe**:
A Frame kept for understanding the Scene after blurry, duplicate or unreadable Frames are rejected.
_Avoid_: selected image

### Processing

**Pipeline**:
The fixed sequence of eight Stages, each bound to one Algorithm, that turns a Scene's Frames into its Objects.
_Avoid_: workflow

**Job**:
One request to process a Scene through every Stage of the Pipeline, from its uploaded Frames.
_Avoid_: run, task, pipeline run

**Stage**:
One named step of the Pipeline: ingest, recon, align, mesh, segment, lift, objects, embed.
_Avoid_: step, phase

**Algorithm**:
One interchangeable method of performing a Stage, named `<stage>.<method>`.
_Avoid_: model, plugin, adapter

**Fake Algorithm**:
A deterministic stand-in for an Algorithm that needs no GPU or model weights, named `<stage>.fake`.
_Avoid_: mock, stub

**Manifest**:
The summary a Stage leaves of what it produced.
_Avoid_: metadata, result

**Stage Run**:
The record of one Stage within a Scene's latest Job: its Algorithm, status, progress and Manifest.
_Avoid_: run

**Vocabulary**:
The list of concept names the segment Stage may give to what it finds.
_Avoid_: label list, prompts, classes

### Compute

**Target**:
The one place where every Stage of the Pipeline executes: in-process on the worker, or scene-compute on the GPU.
_Avoid_: GPU worker, backend, provider

**Provider**:
A service that hosts the machine the backend and its Target run on; Google Colab is the only one.
_Avoid_: target, cloud

**Target Workspace**:
A Scene's uploaded Frames and Stage outputs, kept on its Target; only Publications leave it.
_Avoid_: remote workspace, compute storage

**Compute Run**:
One execution of one Algorithm on a Target, for one Stage of one Job.
_Avoid_: run, remote job

**Publication**:
The immutable set of results from one successful Job that a Scene serves, identified by that Job.
_Avoid_: results folder, version, import

### Understanding

**Instance**:
A thing recognised as the same across several Frames, with a stable identity.
_Avoid_: detection, mask, cluster

**Object**:
An Instance placed in the Scene with its box and label; it keeps the Instance's identity.
_Avoid_: item, entity

**Object Description**:
Three sentences describing an Object by its appearance, its use and its physical place in the Scene; the Object's embedding is made from them.
_Avoid_: document, caption, doc

**Best View**:
A Frame region that shows an Object well, ranked by score.
_Avoid_: thumbnail, crop

**Object Membership**:
Which points of the point cloud and which vertices of the mesh belong to each Object.
_Avoid_: labels, segmentation (for this)

### Exploring

**Search**:
Finding a Scene's Objects that match a free-text query: the Objects whose embeddings are most similar to the query's are re-judged against their Object Descriptions, with a bonus for a label match.
_Avoid_: query (for this), lookup

**Search Hit**:
One Object returned by a Search, with a ranking score that adds a bonus for a label match on top of how well its Object Description answers the query.
_Avoid_: match, result

**Object Heatmap**:
Colouring of one Search's hits by their similarity relative to each other; it is not recognition confidence and cannot be compared across Searches.
_Avoid_: confidence map, probability map

**Highlight**:
Marking a set of Objects in the viewer as the answer to a Search, dimming everything else.
_Avoid_: selection, focus

**Selection**:
The one Object the user has clicked to inspect.
_Avoid_: highlight, active object

**Instance Colors**:
A viewer appearance that paints each Object in its own colour instead of the captured colours.
_Avoid_: segmentation view, labels view

**Measurement**:
A distance the user takes in the viewer between two picked points.
_Avoid_: ruler

## Relationships

- A **Scene** has at most one active **Job**; a **Job** runs every **Stage** in order on one **Target**, one **Compute Run** per **Stage**.
- Each **Stage** is performed by exactly one configured **Algorithm** and leaves one **Manifest** and one **Stage Run**.
- A **Scene** has one **Target Workspace** and serves at most one **Publication**; a failed **Job** leaves the current **Publication** untouched.
- Each **Instance** becomes exactly one **Object** with the same id; nothing merges or splits after lift.
- Each **Object** has exactly one **Object Description**, written once its box exists, and its embedding is made from that Description alone.
- A point or vertex may belong to several **Objects** (a handle and the wardrobe around it) or to none; pointing at it picks the smallest.
- Selecting an Object ends the **Search** and clears its **Highlight**; a new **Search** clears the **Selection**.

## Flagged ambiguities

- "Run" is never used bare: say **Job**, **Stage Run** or **Compute Run**.
- "Workspace" alone means only the folder layout that a **Target Workspace** and a **Publication** share; it is not a domain concept of its own.
- "Publish" is the verb for turning a successful **Job**'s results into the Scene's **Publication**.
- "Scene Graph" and "graph" are retired: what a **Publication** serves is the Scene's **Objects**, their embeddings and its mesh.
