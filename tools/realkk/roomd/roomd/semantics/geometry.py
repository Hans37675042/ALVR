"""Geometric primitives: floor fit, walls, 2.5D raster, components, rectangles, IoU."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from .params import SemanticsParams
from .basics import Obb

_NEIGHBOUR_SHIFTS = ((0, 1), (1, 0), (1, 1), (1, -1))


# ---------------------------------------------------------------- floor

def fit_floor(points, normals, up_min=0.9, band=0.03):
    """Floor height and rms from upward-facing points: the densest 1 cm height bin
    (the floor is the largest horizontal surface), refined by a robust mean."""
    pts = np.asarray(points, dtype=float)
    up = np.asarray(normals)[:, 1] >= up_min
    y = pts[up, 1]
    if y.size == 0:
        return None
    edges = np.arange(y.min() - 0.005, y.max() + 0.015, 0.01)
    counts, edges = np.histogram(y, bins=edges)
    mode = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    near = y[np.abs(y - mode) <= band]
    floor_y = float(np.median(near))
    rms = float(np.sqrt(np.mean((near - floor_y) ** 2)))
    return floor_y, rms


# ---------------------------------------------------------------- raster

@dataclass
class Raster:
    """Max-height raster: ``top[iz, ix]`` (NaN = no sample), cell centre at
    ``(ox + (ix + .5) * cell, oz + (iz + .5) * cell)``."""
    cell: float
    ox: float
    oz: float
    top: np.ndarray

    def centres(self, iz, ix):
        return np.stack([self.ox + (np.asarray(ix) + 0.5) * self.cell,
                         self.oz + (np.asarray(iz) + 0.5) * self.cell], axis=-1)


def rasterize_max(points, cell) -> Raster:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return Raster(cell, 0.0, 0.0, np.full((0, 0), np.nan))
    # Points on a lattice (TSDF voxel columns, sampled grids) must sit mid-cell: on a
    # cell edge, float rounding splits neighbours and leaves empty rows.
    ox = _aligned_origin(pts[:, 0], cell)
    oz = _aligned_origin(pts[:, 2], cell)
    ix = np.floor((pts[:, 0] - ox) / cell).astype(int)
    iz = np.floor((pts[:, 2] - oz) / cell).astype(int)
    w, h = ix.max() + 2, iz.max() + 2
    flat = np.full(h * w, -np.inf)
    np.maximum.at(flat, iz * w + ix, pts[:, 1])
    top = flat.reshape(h, w)
    top[np.isneginf(top)] = np.nan
    return Raster(cell, ox, oz, _fill_dropouts(top))


def _aligned_origin(v, cell):
    """Grid origin below min(v) whose cell centres match the dominant lattice phase of v."""
    a = 2 * math.pi * (v / cell)
    phase = math.atan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * math.pi)
    return (math.floor(v.min() / cell) - 2 + (phase - 0.5) % 1.0) * cell


def _fill_dropouts(top):
    """Fill empty cells that have at least 3 of their 4 edge neighbours (sampling
    dropouts) with the neighbours' max."""
    pad = np.pad(top, 1, constant_values=np.nan)
    nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
    count = np.isfinite(nb).sum(axis=0)
    hole = np.isnan(top) & (count >= 3)
    if hole.any():
        top = top.copy()
        top[hole] = np.nanmax(nb[:, hole], axis=0)
    return top


def _shift_pairs(shape, dy, dx):
    h, w = shape
    a = (slice(0, h - dy), slice(max(0, -dx), w - max(0, dx)))
    b = (slice(dy, h), slice(max(0, dx), w - max(0, -dx)))
    return a, b


def label_components(mask, top, max_step):
    """8-connected components of ``mask`` where neighbours connect only if their
    heights differ by at most ``max_step``. Returns (labels with -1 outside, count)."""
    mask = np.asarray(mask, dtype=bool)
    labels = np.full(mask.shape, -1, dtype=int)
    n = int(mask.sum())
    if n == 0:
        return labels, 0
    idx = np.full(mask.shape, -1, dtype=int)
    idx[mask] = np.arange(n)
    rows, cols = [], []
    for dy, dx in _NEIGHBOUR_SHIFTS:
        a, b = _shift_pairs(mask.shape, dy, dx)
        ok = mask[a] & mask[b]
        with np.errstate(invalid="ignore"):
            ok &= np.abs(top[a] - top[b]) <= max_step
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
    comp, count = connected_labels(n, np.concatenate(rows), np.concatenate(cols))
    labels[mask] = comp
    return labels, count


def connected_labels(n, rows, cols):
    """Connected components of an undirected graph on n nodes given as edge lists.
    Returns (labels 0..count-1, count). Root hooking + pointer jumping, numpy only."""
    labels = np.arange(n)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if n == 0:
        return labels, 0
    while len(rows):
        lr, lc = labels[rows], labels[cols]
        low = np.minimum(lr, lc)
        new = labels.copy()
        np.minimum.at(new, lr, low)
        np.minimum.at(new, lc, low)
        while True:
            nxt = new[new]
            if np.array_equal(nxt, new):
                break
            new = nxt
        if np.array_equal(new, labels):
            break
        labels = new
    _, inv = np.unique(labels, return_inverse=True)
    return inv, int(inv.max()) + 1


def linear_assignment(cost):
    """Minimum-cost assignment (Hungarian, shortest augmenting path), like
    scipy.optimize.linear_sum_assignment: returns (rows, cols) sorted by row."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    transposed = cost.shape[0] > cost.shape[1]
    a = cost.T if transposed else cost
    n, m = a.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)    # p[j]: row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = a[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            on = np.nonzero(used)[0]
            u[p[on]] += delta
            v[on] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def adjacency_counts(labels, reach=1):
    """{(a, b): cell pairs within ``reach`` cells (Chebyshev)} for distinct labels
    a < b (both >= 0); reach 1 = shared 8-neighbour pairs."""
    counts = {}
    shifts = [(dy, dx) for dy in range(reach + 1) for dx in range(-reach, reach + 1)
              if dy > 0 or dx > 0]
    for dy, dx in shifts:
        a, b = _shift_pairs(labels.shape, dy, dx)
        la, lb = labels[a], labels[b]
        ok = (la >= 0) & (lb >= 0) & (la != lb)
        if not ok.any():
            continue
        pairs = np.stack([np.minimum(la[ok], lb[ok]), np.maximum(la[ok], lb[ok])], axis=1)
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        for (p, q), k in zip(uniq.tolist(), cnt.tolist()):
            counts[(p, q)] = counts.get((p, q), 0) + k
    return counts


# ---------------------------------------------------------------- rectangles

def convex_hull(points):
    """Andrew's monotone chain; returns hull vertices CCW (x, z)."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(points):
    """Minimum-area rectangle of 2D points (x, z).

    Returns (centre[2], u[2], v[2], extent_u, extent_v) with u, v unit axes."""
    pts = np.asarray(points, dtype=float)
    hull = convex_hull(pts)
    if len(hull) < 3:
        c = pts.mean(axis=0)
        span = pts.max(axis=0) - pts.min(axis=0) if len(pts) else np.zeros(2)
        return c, np.array([1.0, 0.0]), np.array([0.0, 1.0]), float(span[0]), float(span[1])
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        length = math.hypot(e[0], e[1])
        if length < 1e-9:
            continue
        u = e / length
        v = np.array([-u[1], u[0]])
        pu, pv = hull @ u, hull @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or area < best[0] - 1e-12:
            best = (area, u, v, pu.min(), pu.max(), pv.min(), pv.max())
    _, u, v, u0, u1, v0, v1 = best
    centre = u * (u0 + u1) / 2 + v * (v0 + v1) / 2
    return centre, u, v, float(u1 - u0), float(v1 - v0)


def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def _ccw(poly):
    return poly if _polygon_area(poly) >= 0 else poly[::-1]


def _clip(subject, clip):
    """Sutherland-Hodgman: convex ``subject`` clipped by convex CCW ``clip``."""
    out = list(subject)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append(p + t * (q - p))
    return np.array(out)


def footprint_iou(a: Obb, b: Obb) -> float:
    pa, pb = _ccw(a.corners()), _ccw(b.corners())
    inter = _clip(pa, pb)
    ia = abs(_polygon_area(inter)) if len(inter) >= 3 else 0.0
    union = a.sx * a.sz + b.sx * b.sz - ia
    return ia / union if union > 1e-12 else 0.0


# ---------------------------------------------------------------- walls

def detect_walls(points, normals, floor_y, params: Optional[SemanticsParams] = None
                 ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Wall segments ((ax, az), (bx, bz)) from points with horizontal normals.

    Manhattan: the dominant normal direction (mod 90 deg) gives two axes; along
    each axis, dense offset bins with enough vertical span and length are walls."""
    p = params or SemanticsParams()
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    if len(pts) == 0:
        return []
    rel = pts[:, 1] - floor_y
    sel = (np.abs(nrm[:, 1]) <= p.wall_max_ny) & (rel >= 0.1)
    pts, nrm = pts[sel], nrm[sel]
    if len(pts) < 50:
        return []
    ang = np.degrees(np.arctan2(nrm[:, 2], nrm[:, 0])) % 90.0
    hist, _ = np.histogram(ang, bins=90, range=(0.0, 90.0))
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    theta0 = (np.argmax(smooth) + 0.5)
    walls = []
    for axis_deg in (theta0, theta0 + 90.0):
        a = math.radians(axis_deg)
        n_ax = np.array([math.cos(a), math.sin(a)])
        t_ax = np.array([-n_ax[1], n_ax[0]])
        nd = nrm[:, [0, 2]] @ n_ax
        nlen = np.linalg.norm(nrm[:, [0, 2]], axis=1) + 1e-12
        on_axis = np.abs(nd) / nlen >= math.cos(math.radians(p.wall_angle_tol))
        q = pts[on_axis][:, [0, 2]]
        ys = pts[on_axis][:, 1]
        if len(q) == 0:
            continue
        d = q @ n_ax
        t = q @ t_ax
        bins = np.floor(d / p.wall_bin).astype(int)
        uniq, counts = np.unique(bins, return_counts=True)
        used = set()
        for b in uniq[np.argsort(-counts)]:
            if b in used:
                continue
            group = {b - 1, b, b + 1} - used
            used |= group
            m = np.isin(bins, list(group))
            if m.sum() < 30:
                continue
            for seg in _split_runs(t[m], ys[m], p):
                t0, t1 = seg
                off = float(np.median(d[m]))
                pa = off * n_ax + t0 * t_ax
                pb = off * n_ax + t1 * t_ax
                walls.append(((float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1]))))
    return walls


def _split_runs(t, ys, p):
    order = np.argsort(t)
    t, ys = t[order], ys[order]
    gaps = np.where(np.diff(t) > p.wall_gap)[0]
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps + 1, [len(t)]])
    out = []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        if t[e - 1] - t[s] < p.wall_min_len:
            continue
        if ys[s:e].max() - ys[s:e].min() < p.wall_min_span:
            continue
        out.append((float(t[s]), float(t[e - 1])))
    return out
