"""Shared value types and yaw helpers.

Coordinates: Unity left-handed stage space, +Y up, metres. Yaw is in degrees,
clockwise seen from above (Unity / RealKK.Core ``Quat.FromYaw``): an object's
local +Z (its front) points to world ``(sin yaw, 0, cos yaw)`` and its local +X
to ``(cos yaw, 0, -sin yaw)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


class Kind:
    FLOOR = "Floor"
    WALL = "Wall"
    COUCH = "Couch"
    CHAIR = "Chair"
    TABLE = "Table"
    BED = "Bed"
    OTHER = "Other"


class ObjectState:
    PRESENT = "Present"
    MOVING = "Moving"
    MISSING = "Missing"
    REMOVED = "Removed"
    REJECTED = "Rejected"


class SeatState:
    AVAILABLE = "Available"
    MOVING = "Moving"
    BLOCKED = "Blocked"
    OCCUPIED_BY_USER = "OccupiedByUser"
    MISSING = "Missing"
    REMOVED = "Removed"


class Source:
    MANUAL = "Manual"
    SCENE_API = "SceneApi"
    FUSED = "Fused"


def wrap_deg(a):
    """Wrap an angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def yaw_diff(a, b, symmetric=False):
    """Absolute yaw difference in degrees; ``symmetric`` treats yaw and yaw+180 as equal."""
    d = abs(wrap_deg(a - b))
    return min(d, 180.0 - d) if symmetric else d


def yaw_quat(yaw):
    """Yaw-only rotation around +Y as (x, y, z, w)."""
    h = math.radians(yaw) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def quat_yaw(q):
    """Yaw of the rotated forward vector, like RealKK.Core ``Quat.Yaw``."""
    x, y, z, w = q
    # forward (0,0,1) rotated by q
    fx = 2.0 * (x * z + w * y)
    fz = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.atan2(fx, fz))


def local_to_world(cx, cz, yaw, lx, lz):
    a = math.radians(yaw)
    c, s = math.cos(a), math.sin(a)
    return cx + lx * c + lz * s, cz - lx * s + lz * c


def world_to_local(cx, cz, yaw, x, z):
    a = math.radians(yaw)
    c, s = math.cos(a), math.sin(a)
    dx = np.asarray(x) - cx
    dz = np.asarray(z) - cz
    return dx * c - dz * s, dx * s + dz * c


def yaw_from_front(fx, fz):
    """Yaw whose local +Z points along (fx, fz)."""
    return math.degrees(math.atan2(fx, fz))


def yaw_from_right(ux, uz):
    """Yaw whose local +X points along (ux, uz)."""
    return math.degrees(math.atan2(-uz, ux))


@dataclass(frozen=True)
class Obb:
    """Yaw-only oriented box: footprint centre (cx, cz), yaw, full extents sx (local X)
    and sz (local Z), vertical range y0..y1 (world)."""
    cx: float
    cz: float
    yaw: float
    sx: float
    sz: float
    y0: float
    y1: float

    def to_local(self, x, z):
        return world_to_local(self.cx, self.cz, self.yaw, x, z)

    def contains_xz(self, x, z, margin=0.0):
        lx, lz = self.to_local(x, z)
        return (np.abs(lx) <= self.sx / 2 + margin) & (np.abs(lz) <= self.sz / 2 + margin)

    def contains(self, pts):
        pts = np.asarray(pts, dtype=float)
        inside = self.contains_xz(pts[:, 0], pts[:, 2])
        return inside & (pts[:, 1] >= self.y0) & (pts[:, 1] <= self.y1)

    def corners(self):
        hx, hz = self.sx / 2, self.sz / 2
        return np.array([local_to_world(self.cx, self.cz, self.yaw, lx, lz)
                         for lx, lz in ((-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz))])

    def scaled(self, f):
        return replace(self, sx=self.sx * f, sz=self.sz * f)

    def with_y(self, y0, y1):
        return replace(self, y0=y0, y1=y1)


@dataclass(frozen=True)
class SceneLabel:
    """One Meta scene anchor already converted to Unity stage space.

    ``labels`` is the Meta CSV (e.g. ``"COUCH"``); ``obb`` its 3D bounding box
    (bbox3d, or bbox2d extruded to the floor) as a yaw-only box.
    """
    uuid: str
    labels: str
    obb: Obb
