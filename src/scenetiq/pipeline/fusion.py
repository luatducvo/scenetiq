"""Fusing stage: TSDF-fuse the per-View depth into one mesh (ADR 0002).

The mesh vertices become the Reconstruction's point cloud, so the points that
Instances are made of are exactly the points the user sees.

This uses Open3D's tensor ``VoxelBlockGrid``. Do not go back to the legacy
``ScalableTSDFVolume``: in Open3D 0.20.0, the only release with Python 3.13 wheels, it
divides metric depth by 1000 when activating blocks, so only surfaces within about one
block of a camera survive (https://github.com/isl-org/Open3D/issues/7592).
"""

import numpy as np

from scenetiq.config import FusionOptions
from scenetiq.pipeline.adapters import c2w_to_w2c
from scenetiq.pipeline.view_data import FusedGeometry, ViewData


def fuse(view_data: ViewData, options: FusionOptions | None = None) -> FusedGeometry:
    """Integrate every View's depth into a TSDF volume and extract its mesh.

    The Reconstructing stage has already dropped depth beyond ``DA3Options.max_depth``,
    so everything left in ``view_data`` is fused. A View with no depth left adds nothing.

    Raises:
        RuntimeError: If fusion yields an empty mesh (e.g. no reliable depth).
    """
    import open3d as o3d

    opts = options or FusionOptions()
    device = o3d.core.Device("CPU:0")
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=opts.voxel_size,
        block_resolution=opts.block_resolution,
        block_count=opts.initial_block_count,
        device=device,
    )
    depth_max = float(view_data.depth.max()) + 1.0  # Open3D requires one; this cuts nothing.
    tsdf_args = {
        "depth_scale": 1.0,
        "depth_max": depth_max,
        "trunc_voxel_multiplier": opts.sdf_trunc_voxels,
    }
    integrated = False
    for i, cam in enumerate(view_data.cameras):
        depth = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(view_data.depth[i])))
        intrinsics = o3d.core.Tensor(cam.intrinsics.astype(np.float64))
        w2c = o3d.core.Tensor(c2w_to_w2c(cam.cam_to_world).astype(np.float64))
        try:
            blocks = grid.compute_unique_block_coordinates(depth, intrinsics, w2c, **tsdf_args)
        except RuntimeError as exc:
            # Open3D samples every 4th pixel and raises when none of them has depth.
            if "No block is touched" not in str(exc):
                raise
            continue
        rgb = np.ascontiguousarray(view_data.images[i]).astype(np.float32) / 255.0
        color = o3d.t.geometry.Image(o3d.core.Tensor(rgb))
        grid.integrate(blocks, depth, color, intrinsics, intrinsics, w2c, **tsdf_args)
        integrated = True
    if not integrated:
        raise RuntimeError("TSDF fusion produced an empty mesh; no View has reliable depth")

    # Threshold 0 keeps every observed voxel; the default of 3 would need three observations.
    mesh = grid.extract_triangle_mesh(weight_threshold=0.0).to_legacy()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        raise RuntimeError("TSDF fusion produced an empty mesh; the Capture may lack overlap")
    colors = (np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0) * 255).round().astype(np.uint8)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    return FusedGeometry(vertices=vertices, colors=colors, triangles=triangles)
