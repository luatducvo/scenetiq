"""Fusing stage: TSDF-fuse the per-View depth into one mesh (ADR 0002).

The mesh vertices become the Reconstruction's point cloud, so the points that
Instances are made of are exactly the points the user sees.
"""

import numpy as np

from scenetiq.config import FusionOptions
from scenetiq.pipeline.adapters import c2w_to_w2c
from scenetiq.pipeline.view_data import FusedGeometry, ViewData


def fuse(view_data: ViewData, options: FusionOptions | None = None) -> FusedGeometry:
    """Integrate every View's depth into a TSDF volume and extract its mesh.

    The Reconstructing stage has already dropped depth beyond ``DA3Options.max_depth``,
    so everything left in ``view_data`` is fused.

    Raises:
        RuntimeError: If fusion yields an empty mesh (e.g. no reliable depth).
    """
    import open3d as o3d

    opts = options or FusionOptions()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=opts.voxel_size,
        sdf_trunc=opts.voxel_size * opts.sdf_trunc_voxels,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    depth_trunc = float(view_data.depth.max()) + 1.0  # Open3D requires one; this cuts nothing.
    for i, cam in enumerate(view_data.cameras):
        color = o3d.geometry.Image(np.ascontiguousarray(view_data.images[i]))
        depth = o3d.geometry.Image(np.ascontiguousarray(view_data.depth[i]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        k = cam.intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            cam.width, cam.height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        )
        volume.integrate(rgbd, intrinsic, c2w_to_w2c(cam.cam_to_world).astype(np.float64))

    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        raise RuntimeError("TSDF fusion produced an empty mesh; the Capture may lack overlap")
    colors = (np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0) * 255).round().astype(np.uint8)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    return FusedGeometry(vertices=vertices, colors=colors, triangles=triangles)
