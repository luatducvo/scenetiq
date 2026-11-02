# MV3DIS (vendored)

- Upstream: https://github.com/zybjn/MV3DIS
- Pinned commit: `3f1b7023c45c6d08dedbf9db3201e2f0620378c0` (2026-10-10)
- Licence: MIT (`LICENSE`, Copyright (c) 2026 Yibo Zhao)

## Files

| File | Origin | Local modifications |
|---|---|---|
| `mv3dis_base.py` | `mv3dis_base.py` | Import changed to `from . import mv3dis_utils as utils`. Otherwise verbatim. |
| `mv3dis_utils.py` | `helpers/mv3dis_utils.py` | Only the functions `MV3DISBase.assign_label` reaches are kept, verbatim, except that `filter_masks` uses the module-level `device` instead of a hard-coded `"cuda"` (so it also runs on CPU). Header imports reduced to numpy / torch / tqdm / defaultdict. |
| `in_memory.py` | `mv3dis_main.py` (`ScanNetMV3DIS`) | New `InMemoryMV3DIS` takes arrays instead of ScanNet files. `get_seg_data` is copied with two changes: the file-loading branches are removed, and the centroid distance uses `scipy.spatial.distance.cdist` (same values, avoids an (S,S,3) temporary). |

## Known upstream issue

At this commit, `get_seg_data(point_level=True)` returns 4 values while `MV3DISBase._load_segment_data` unpacks 6, so point-level mode is broken. It would also allocate dense (N×N) matrices over all points. SceneTIQ therefore always passes primitive ids: the ScanNet Segmentator superpoints, or voxel-grid primitives as a fallback.
