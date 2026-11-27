# TSDF-fused mesh exists only to feed MV3DIS superpoints

MV3DIS groups points into superpoints with the ScanNet Segmentator, which runs on a mesh. DA3 outputs only depth maps and cameras, not a mesh. We fuse DA3 depth into a TSDF volume, extract a mesh, and run the Segmentator on it, just as the original pipeline does. The mesh vertices are the Reconstruction's point cloud, so the points MV3DIS labels are the same points the frontend shows. The mesh itself is never sent to the user.

## Considered Options

- Superpoints computed directly on the point cloud (cut-pursuit, VCCS): needs no mesh, but strays from the setup MV3DIS was tuned on.
- `point_level=True` (every point is its own primitive): rejected. It is broken at the vendored MV3DIS commit (return arity mismatch), and it would allocate dense N×N matrices over all points. The fallback primitive is a voxel grid of about 15 cm instead.

## Consequences

TSDF fusion also denoises the DA3 depth. This helps MV3DIS's visibility check, which requires depth to agree within a relative 5% tolerance.
