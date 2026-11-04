# Gravity alignment with GeoCalib and RANSAC, stored in the Reconstruction

DA3's world frame is not tied to gravity, and TSDF fusion keeps whatever frame the cameras use, so Scenes came out tilted and axis-aligned bounding boxes larger than their objects. An Aligning stage after Fusing now estimates up with GeoCalib per View, refines it with the normal of a large horizontal plane found by RANSAC when one qualifies, and rotates the points and cameras so that physical up is **−Y** in every Reconstruction. −Y matches the viewer's existing camera and the OpenCV camera convention. Changing it later would make stored Scenes disagree with new ones.

## Considered Options

- RANSAC alone: precise with a clear floor, but the largest plane may be a wall, a plane's normal has no sign, and it fails with no plane. GeoCalib supplies the direction and the sign.
- GeoCalib alone: works with no plane, but carries per-photo error plus DA3's rotation error. It is the fallback when no plane qualifies.
- Rotating in the viewer only: leaves stored centroids and bounding boxes in the tilted frame, which is what spatial reasoning would read.
- Aligning before Fusing: RANSAC would then work on noisy per-View depth. A rotation after Fusing keeps the vertex order, so the point-order contract holds.
- Yaw alignment to walls: gravity cannot determine yaw, and a Manhattan assumption fails on Scenes without clear walls. Left out, so diagonal objects keep larger boxes.
- GeoCalib's `camera_R_rig` joint estimation: built for rigid rigs with accurate rotations, while DA3's rotations carry error. Revisit if per-View estimation proves too inaccurate.

## Consequences

Weak evidence never fails a job: Views that do not agree leave the Scene unrotated, and the log line is the only explanation. Technical failures do fail the job, so a broken install cannot silently leave every Scene tilted. The stage has no off switch, so if GeoCalib cannot run, every real job fails until it is fixed. A rotation cannot fix geometry DA3 warped, and a large slope within the plane angle limit tilts the Scene by that slope. Scenes stored before this stage keep their old frame.
