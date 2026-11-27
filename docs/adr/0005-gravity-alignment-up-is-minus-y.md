# Gravity alignment: up is −Y in every Reconstruction

DA3's world frame is not tied to gravity, so Scenes came out tilted and axis-aligned bounding boxes larger than their objects. The Aligning stage, after Fusing, estimates up with GeoCalib on up to `alignment.max_views` evenly spaced Views, keeps it only when more than `consensus_fraction` of them agree within `consensus_angle_deg`, then refines it with the normal of the largest RANSAC plane that holds at least `min_plane_fraction` of the points and lies within `max_plane_angle_deg` of it. Points and cameras get the smallest rotation that takes up to `CONVENTIONAL_UP = (0, -1, 0)`, so a smaller y is higher. −Y matches the viewer's camera and the OpenCV convention.

## Considered Options

- RANSAC alone: the largest plane may be a wall, a normal has no sign, and some Scenes have no plane. GeoCalib supplies direction and sign.
- Rotating in the viewer only: stored centroids and boxes, which Relations read, would stay tilted.
- Aligning before Fusing: RANSAC on noisy per-View depth. Rotating after Fusing keeps the vertex order.
- Yaw alignment to walls: gravity cannot fix yaw, and a Manhattan assumption fails without clear walls.

## Consequences

- Weak evidence never fails a job: without consensus the Scene stays in DA3's frame, and a job note says vertical Relations may be wrong. A technical failure fails the job, and the stage has no off switch.
- GeoCalib follows a photo's roll only to about ±45° and answers wrongly beyond it; Views all rolled the same way past that (sideways portrait photos from a non-browser client) can agree on a wrong up.
- Changing the convention would make stored Scenes disagree with new ones.
