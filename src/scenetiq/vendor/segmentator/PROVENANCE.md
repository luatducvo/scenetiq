# ScanNet Segmentator (ported)

- Upstream: https://github.com/ScanNet/ScanNet/tree/master/Segmentator (`segmentator.cpp`)
- Pinned commit: `2261e3849f331ef3e9429ac7c956e2691fe37fe6` (2018-01-31)
- Licence: MIT-style ScanNet licence (`LICENSE`)

## Local modifications

`segmentator.py` is a line-by-line Python/numpy port of `segment()` and `segment_graph()`, so no C++ build is needed. Differences:

- Mesh I/O (tinyply / tinyobj) is dropped. Vertices and faces are passed as arrays.
- Vertex normals are computed as the vectorised mean of the incident face normals. Upstream gets the same value as a running `lerp` mean.
- Degenerate faces and zero-length edges get a zero normal or zero direction instead of NaN.
- Edges are sorted with a stable sort. Upstream `std::sort` leaves the order of equal weights unspecified.
- The output is relabelled to consecutive ids starting at 0. Upstream returns raw union-find roots.

Defaults match upstream: `kThresh=0.01`, `segMinVerts=20`.
