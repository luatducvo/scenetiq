import numpy as np

from scenetiq.pipeline.superpoints import mesh_superpoints


def _grid(n: int, to_xyz) -> tuple[np.ndarray, np.ndarray]:
    """An n x n vertex grid mapped by ``to_xyz(u, v)``, triangulated."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    verts = np.stack(to_xyz(u.ravel(), v.ravel()), axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tris = np.concatenate(
        [np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)]
    )
    return verts.astype(np.float32), tris.astype(np.int32)


def test_flat_plane_is_one_superpoint():
    verts, tris = _grid(20, lambda u, v: (u, v, np.zeros_like(u)))
    ids = mesh_superpoints(verts, tris)
    assert len(np.unique(ids)) == 1


def test_two_planes_at_a_right_angle_split_at_the_fold():
    floor_v, floor_t = _grid(20, lambda u, v: (u, v, np.zeros_like(u)))
    # A wall standing on the floor's far edge, not sharing vertices with it.
    wall_v, wall_t = _grid(20, lambda u, v: (u, np.ones_like(u) + 0.001, v))
    verts = np.concatenate([floor_v, wall_v])
    tris = np.concatenate([floor_t, wall_t + len(floor_v)])
    # Stitch the two along the fold so they form one connected, folded mesh.
    n = 20
    floor_edge = np.arange(n * (n - 1), n * n)  # v == 1 row of the floor
    wall_edge = len(floor_v) + np.arange(n)  # v == 0 row of the wall
    stitch = [[floor_edge[i], floor_edge[i + 1], wall_edge[i + 1]] for i in range(n - 1)] + [
        [floor_edge[i], wall_edge[i + 1], wall_edge[i]] for i in range(n - 1)
    ]
    tris = np.concatenate([tris, np.array(stitch, dtype=np.int32)])

    ids = mesh_superpoints(verts, tris)

    floor_ids = set(ids[: len(floor_v) - n].tolist())  # away from the fold
    wall_ids = set(ids[len(floor_v) + n :].tolist())
    assert len(floor_ids) == 1
    assert len(wall_ids) == 1
    assert floor_ids != wall_ids


def test_small_segments_merge_into_neighbours():
    verts, tris = _grid(20, lambda u, v: (u, v, np.zeros_like(u)))
    # A tiny bump: lift one vertex so it would form its own segment.
    verts[210, 2] = 0.3
    ids = mesh_superpoints(verts, tris, seg_min_verts=20)
    _, counts = np.unique(ids, return_counts=True)
    assert counts.min() >= 20


def test_superpoint_ids_are_consecutive_from_zero():
    verts, tris = _grid(10, lambda u, v: (u, v, np.zeros_like(u)))
    ids = mesh_superpoints(verts, tris)
    assert ids.min() == 0
    assert sorted(np.unique(ids).tolist()) == list(range(ids.max() + 1))
    assert len(ids) == len(verts)
