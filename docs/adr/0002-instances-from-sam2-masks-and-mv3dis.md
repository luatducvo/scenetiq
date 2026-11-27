# Instances from SAM2 automatic Masks, merged by MV3DIS on TSDF mesh superpoints

Instances must be findable by any free-text Query, so segmentation uses no vocabulary. The Masking stage runs SAM2's prompt-free automatic mask generator on every View, and the vendored MV3DIS merges those Masks into 3D Instances. MV3DIS groups points into superpoints with the ScanNet Segmentator, which needs a mesh, while DA3 outputs only depth maps and cameras; so the Fusing stage integrates DA3 depth into an Open3D TSDF volume and extracts a mesh. The mesh vertices are the Reconstruction's point cloud: the points MV3DIS labels are the points the viewer shows, and the mesh itself is never sent.

## Considered Options

- Grounded-SAM2 or YOLO-World prompted with ScanNet200 class names (MV3DIS's default): an object outside the list never becomes an Instance.
- Superpoints computed directly on the point cloud (cut-pursuit, VCCS): no mesh, but strays from the setup MV3DIS was tuned on.
- `point_level=True`, every point its own primitive: broken at the vendored commit and allocates N×N matrices. The fallback primitive is a voxel grid (`superpoints.primitives = "voxel"`).

## Consequences

- Points keep the mesh vertex order and count from fusion onward; the PLY, `instance_ids.bin` and the MV3DIS input all rely on it.
- TSDF fusion also denoises depth, which helps MV3DIS's 5 % relative depth visibility check.
- Automatic generation over-segments; MV3DIS's merging filters the fragments, at the cost of runtime. At most 255 Masks per View fit its `uint8` label map.
