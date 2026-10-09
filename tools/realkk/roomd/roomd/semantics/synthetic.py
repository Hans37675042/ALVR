"""Procedural room for tests: a :class:`MapView` built from yaw-only boxes.

Furniture builders return lists of :class:`Obb` parts in their object frame
(front = local +Z, origin = bottom-face centre) placed at (x, z, yaw).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np

from .mapview import FLAG_KNOWN, FLAG_OBSTACLE, FLAG_WALKABLE, HeightMap
from .basics import Obb, local_to_world


def _place(x, z, yaw, parts) -> List[Obb]:
    out = []
    for lx, lz, sx, sz, y0, y1 in parts:
        wx, wz = local_to_world(x, z, yaw, lx, lz)
        out.append(Obb(wx, wz, yaw, sx, sz, y0, y1))
    return out


def _legs(w, d, h, leg):
    return [(sx * (w / 2 - leg / 2), sz * (d / 2 - leg / 2), leg, leg, 0.0, h)
            for sx in (-1, 1) for sz in (-1, 1)]


def chair(x, z, yaw=0.0, seat_h=0.45, width=0.45, depth=0.45, back_h=0.9,
          back_t=0.05, slab=0.04, leg=0.04):
    parts = [(0.0, 0.0, width, depth, seat_h - slab, seat_h)]
    parts += _legs(width, depth, seat_h - slab, leg)
    parts.append((0.0, -depth / 2 + back_t / 2, width, back_t, seat_h, back_h))
    return _place(x, z, yaw, parts)


def stool(x, z, yaw=0.0, h=0.45, w=0.35, slab=0.04, leg=0.04):
    return _place(x, z, yaw, [(0.0, 0.0, w, w, h - slab, h)] + _legs(w, w, h - slab, leg))


def couch(x, z, yaw=0.0, length=2.0, depth=0.9, seat_h=0.45, back_h=0.85,
          back_t=0.2, arm_w=0.15, arm_h=0.6):
    parts = [(0.0, 0.0, length, depth, 0.0, seat_h),
             (0.0, -depth / 2 + back_t / 2, length, back_t, seat_h, back_h)]
    for s in (-1, 1):
        parts.append((s * (length / 2 - arm_w / 2), 0.0, arm_w, depth, seat_h, arm_h))
    return _place(x, z, yaw, parts)


def table(x, z, yaw=0.0, w=1.2, d=0.8, h=0.75, top_t=0.04, leg=0.05):
    return _place(x, z, yaw, [(0.0, 0.0, w, d, h - top_t, h)] + _legs(w, d, h - top_t, leg))


def coffee_table(x, z, yaw=0.0, w=1.0, d=0.5, h=0.42):
    return table(x, z, yaw, w=w, d=d, h=h)


def bed(x, z, yaw=0.0, w=1.6, l=2.0, h=0.5, head_h=1.0, head_t=0.06):
    return _place(x, z, yaw, [(0.0, 0.0, w, l, 0.0, h),
                              (0.0, -l / 2 + head_t / 2, w, head_t, h, head_h)])


def storage(x, z, yaw=0.0, w=0.8, d=0.4, h=0.9):
    return _place(x, z, yaw, [(0.0, 0.0, w, d, 0.0, h)])


def clothes_pile(x, z, base_y=0.0, size=0.3, height=0.2, layers=4):
    """Stacked shrinking layers: an uneven mound, no flat top."""
    parts = []
    for k in range(layers):
        s = size * (1.0 - 0.2 * k)
        parts.append((0.0, 0.0, s, s, base_y + k * height / layers,
                      base_y + (k + 1) * height / layers))
    return _place(x, z, 0.0, parts)


def user_torso(x, z, yaw=0.0, seat_h=0.45):
    """A seated user's thighs and torso on a chair placed at (x, z, yaw)."""
    return _place(x, z, yaw, [(0.0, 0.05, 0.36, 0.35, seat_h, seat_h + 0.15),
                              (0.0, -0.08, 0.36, 0.22, seat_h, seat_h + 0.6)])


class SyntheticRoom:
    """Rectangular room (walls at |x| = half_x, |z| = half_z) with furniture boxes.

    ``hidden`` footprints are unobserved: no points, unknown cells, zero
    free/occupied fractions. ``noise`` (m, sigma) is drawn fresh on every call."""

    def __init__(self, half_x=2.5, half_z=2.5, floor_y=0.0, wall_h=2.5, walls=True,
                 noise=0.0, seed=0, point_step=0.02, side_step=0.05, floor_step=0.04,
                 hm_cell=0.05):
        self.half_x, self.half_z = half_x, half_z
        self.floor_y = floor_y
        self.wall_h = wall_h
        self.noise = noise
        self.point_step, self.side_step, self.floor_step = point_step, side_step, floor_step
        self.hm_cell = hm_cell
        self.items: Dict[str, List[Obb]] = {}
        self.hidden: List[Obb] = []
        self._rng = np.random.default_rng(seed)
        self._walls: List[Obb] = []
        if walls:
            t = 0.1
            self._walls = [
                Obb(half_x + t / 2, 0.0, 0.0, t, 2 * half_z + 2 * t, floor_y, floor_y + wall_h),
                Obb(-half_x - t / 2, 0.0, 0.0, t, 2 * half_z + 2 * t, floor_y, floor_y + wall_h),
                Obb(0.0, half_z + t / 2, 0.0, 2 * half_x, t, floor_y, floor_y + wall_h),
                Obb(0.0, -half_z - t / 2, 0.0, 2 * half_x, t, floor_y, floor_y + wall_h),
            ]

    # ------------------------------------------------------------ scene editing
    def place(self, name: str, boxes: Iterable[Obb]):
        fy = self.floor_y
        self.items[name] = [Obb(b.cx, b.cz, b.yaw, b.sx, b.sz, b.y0 + fy, b.y1 + fy)
                            for b in boxes]

    def remove(self, name: str):
        self.items.pop(name, None)

    def set_hidden(self, regions: Iterable[Obb]):
        self.hidden = list(regions)

    def boxes(self) -> List[Obb]:
        out = list(self._walls)
        for parts in self.items.values():
            out.extend(parts)
        return out

    # ------------------------------------------------------------ helpers
    def _inside_any(self, pts, boxes, skip=None):
        inside = np.zeros(len(pts), dtype=bool)
        for i, b in enumerate(boxes):
            if i == skip:
                continue
            inside |= b.contains(pts)
        return inside

    def _hidden_xz(self, x, z):
        m = np.zeros(np.shape(x), dtype=bool)
        for h in self.hidden:
            m |= h.contains_xz(x, z)
        return m

    def _interior(self, x, z, margin=0.01):
        return (np.abs(x) <= self.half_x + margin) & (np.abs(z) <= self.half_z + margin)

    @staticmethod
    def _grid(length, step):
        n = max(1, int(round(length / step)))
        return (np.arange(n) + 0.5) * (length / n) - length / 2

    # ------------------------------------------------------------ MapView
    def floor_plane(self):
        return self.floor_y, max(self.noise, 0.0)

    def surface_points(self, min_y, max_y, region: Optional[Obb] = None):
        boxes = self.boxes()
        pts_list, nrm_list = [], []
        eps = 0.005
        # floor
        xs = self._grid(2 * self.half_x, self.floor_step)
        zs = self._grid(2 * self.half_z, self.floor_step)
        fx, fz = np.meshgrid(xs, zs)
        fp = np.column_stack([fx.ravel(), np.full(fx.size, self.floor_y), fz.ravel()])
        probe = fp + [0.0, eps, 0.0]
        fp = fp[~self._inside_any(probe, boxes)]
        pts_list.append(fp)
        nrm_list.append(np.tile([0.0, 1.0, 0.0], (len(fp), 1)))
        for i, b in enumerate(boxes):
            # top face
            lx, lz = np.meshgrid(self._grid(b.sx, self.point_step), self._grid(b.sz, self.point_step))
            lx, lz = lx.ravel(), lz.ravel()
            wx, wz = local_to_world(b.cx, b.cz, b.yaw, lx, lz)
            tp = np.column_stack([wx, np.full(wx.size, b.y1), wz])
            keep = ~self._inside_any(tp + [0.0, eps, 0.0], boxes, skip=i)
            tp = tp[keep]
            pts_list.append(tp)
            nrm_list.append(np.tile([0.0, 1.0, 0.0], (len(tp), 1)))
            # side faces
            ys = self._grid(b.y1 - b.y0, self.side_step) + (b.y0 + b.y1) / 2
            a = math.radians(b.yaw)
            ex = np.array([math.cos(a), -math.sin(a)])
            ez = np.array([math.sin(a), math.cos(a)])
            for axis, half, along, along_len in ((ex, b.sx / 2, ez, b.sz), (-ex, b.sx / 2, ez, b.sz),
                                                 (ez, b.sz / 2, ex, b.sx), (-ez, b.sz / 2, ex, b.sx)):
                ts = self._grid(along_len, self.side_step)
                tt, yy = np.meshgrid(ts, ys)
                tt, yy = tt.ravel(), yy.ravel()
                cx = b.cx + axis[0] * half + along[0] * tt
                cz = b.cz + axis[1] * half + along[1] * tt
                sp = np.column_stack([cx, yy, cz])
                out = sp + [axis[0] * eps, 0.0, axis[1] * eps]
                keep = ~self._inside_any(out, boxes, skip=i) & self._interior(out[:, 0], out[:, 2])
                keep &= (yy > self.floor_y + eps)
                sp = sp[keep]
                pts_list.append(sp)
                nrm_list.append(np.tile([axis[0], 0.0, axis[1]], (len(sp), 1)))
        pts = np.vstack(pts_list)
        nrm = np.vstack(nrm_list)
        keep = ~self._hidden_xz(pts[:, 0], pts[:, 2]) & self._interior(pts[:, 0], pts[:, 2], 0.02)
        if region is not None:
            keep &= region.contains_xz(pts[:, 0], pts[:, 2])
        pts, nrm = pts[keep], nrm[keep]
        if self.noise > 0:
            pts = pts + self._rng.normal(0.0, self.noise, pts.shape)
        sel = (pts[:, 1] >= min_y) & (pts[:, 1] <= max_y)
        return pts[sel], nrm[sel]

    def heightmap(self) -> HeightMap:
        c = self.hm_cell
        w = int(round(2 * self.half_x / c))
        h = int(round(2 * self.half_z / c))
        ox, oz = -self.half_x, -self.half_z
        iz, ix = np.mgrid[0:h, 0:w]
        x = ox + (ix + 0.5) * c
        z = oz + (iz + 0.5) * c
        top = np.full((h, w), self.floor_y, dtype=float)
        for b in self.boxes():
            m = b.contains_xz(x, z)
            top[m] = np.maximum(top[m], b.y1)
        if self.noise > 0:
            top = top + self._rng.normal(0.0, self.noise, top.shape)
        floor = np.full((h, w), self.floor_y, dtype=float)
        hidden = self._hidden_xz(x, z)
        flags = np.full((h, w), FLAG_KNOWN, dtype=np.uint8)
        obstacle = (top - floor) > 0.05
        flags[obstacle] |= FLAG_OBSTACLE
        flags[~obstacle] |= FLAG_WALKABLE
        flags[hidden] = 0
        top[hidden] = np.nan
        floor[hidden] = np.nan
        return HeightMap(c, ox, oz, floor, top, flags)

    def _volume_samples(self, obb: Obb, step_xz=0.05, step_y=0.02):
        lx = self._grid(obb.sx, step_xz) if obb.sx > 2 * step_xz else np.array([-obb.sx / 4, obb.sx / 4])
        lz = self._grid(obb.sz, step_xz) if obb.sz > 2 * step_xz else np.array([-obb.sz / 4, obb.sz / 4])
        hy = obb.y1 - obb.y0
        ys = self._grid(hy, step_y) + (obb.y0 + obb.y1) / 2
        gx, gz, gy = np.meshgrid(lx, lz, ys)
        wx, wz = local_to_world(obb.cx, obb.cz, obb.yaw, gx.ravel(), gz.ravel())
        return np.column_stack([wx, gy.ravel(), wz])

    def _fractions(self, obb: Obb):
        pts = self._volume_samples(obb)
        known = ~self._hidden_xz(pts[:, 0], pts[:, 2])
        occ = self._inside_any(pts, self.boxes())
        n = float(len(pts))
        return (known & ~occ).sum() / n, (known & occ).sum() / n, known.sum() / n

    def free_fraction(self, obb: Obb) -> float:
        return float(self._fractions(obb)[0])

    def occupied_fraction(self, obb: Obb) -> float:
        return float(self._fractions(obb)[1])

    def visible_fraction(self, obb: Obb) -> float:
        return float(self._fractions(obb)[2])
