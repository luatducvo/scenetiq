"""Felzenszwalb-Huttenlocher segmentation on mesh normals (ScanNet Segmentator port)."""

import numpy as np


class _Universe:
    """Disjoint-set forest with union-by-rank and path compression (as upstream)."""

    def __init__(self, elements):
        self.rank = [0] * elements
        self.p = list(range(elements))
        self.size = [1] * elements

    def find(self, x):
        p = self.p
        y = x
        while y != p[y]:
            y = p[y]
        p[x] = y
        return y

    def join(self, x, y):
        if self.rank[x] > self.rank[y]:
            self.p[y] = x
            self.size[x] += self.size[y]
        else:
            self.p[x] = y
            self.size[y] += self.size[x]
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1


def _segment_graph(num_vertices, edge_a, edge_b, edge_w, c):
    order = np.argsort(edge_w, kind="stable")
    u = _Universe(num_vertices)
    threshold = [c] * num_vertices
    for a0, b0, w in zip(edge_a[order].tolist(), edge_b[order].tolist(), edge_w[order].tolist()):
        a = u.find(a0)
        b = u.find(b0)
        if a != b and w <= threshold[a] and w <= threshold[b]:
            u.join(a, b)
            a = u.find(a)
            threshold[a] = w + c / u.size[a]
    return u, order


def segment(vertices, faces, kthr=0.01, seg_min_verts=20):
    """Return the segment id of every vertex (consecutive from 0)."""
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(verts)
    i1, i2, i3 = faces[:, 0], faces[:, 1], faces[:, 2]
    p1, p2, p3 = verts[i1], verts[i2], verts[i3]

    face_n = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(face_n, axis=1, keepdims=True)
    face_n = np.divide(face_n, norm, out=np.zeros_like(face_n), where=norm > 0)
    normals = np.zeros((n, 3))
    counts = np.zeros(n)
    for idx in (i1, i2, i3):
        np.add.at(normals, idx, face_n)
        np.add.at(counts, idx, 1)
    normals = np.divide(normals, counts[:, None], out=np.zeros_like(normals), where=counts[:, None] > 0)

    edge_a = np.stack([i1, i1, i3], axis=1).reshape(-1)
    edge_b = np.stack([i2, i3, i2], axis=1).reshape(-1)
    n1, n2 = normals[edge_a], normals[edge_b]
    d = verts[edge_b] - verts[edge_a]
    dd = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, dd, out=np.zeros_like(d), where=dd > 0)
    dot = np.sum(n1 * n2, axis=1)
    dot2 = np.sum(n2 * d, axis=1)
    w = 1.0 - dot
    w = np.where(dot2 > 0, w * w, w)

    u, _ = _segment_graph(n, edge_a, edge_b, w.astype(np.float32), np.float32(kthr))

    for a0, b0 in zip(edge_a.tolist(), edge_b.tolist()):
        a = u.find(a0)
        b = u.find(b0)
        if a != b and (u.size[a] < seg_min_verts or u.size[b] < seg_min_verts):
            u.join(a, b)

    roots = np.array([u.find(q) for q in range(n)], dtype=np.int64)
    _, ids = np.unique(roots, return_inverse=True)
    return ids.reshape(-1)
