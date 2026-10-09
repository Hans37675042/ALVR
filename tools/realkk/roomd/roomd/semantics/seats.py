"""Seat points (port of RealKK.Core SeatGenerator) and per-seat occupancy checks."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .mapview import HeightMap, heightmap_cells
from .params import SemanticsParams
from .basics import Obb, SeatState, Source, local_to_world, yaw_quat

SEAT_SPACING = 0.6  # SeatGenerator.SeatSpacing


def pose_json(x, y, z, yaw):
    qx, qy, qz, qw = yaw_quat(yaw)
    return {"Position": {"X": _r(x), "Y": _r(y), "Z": _r(z)},
            "Rotation": {"X": _r(qx), "Y": _r(qy), "Z": _r(qz), "W": _r(qw)}}


def _r(v, nd=6):
    return round(float(v), nd)


def seat_layout(size: Sequence[float]) -> List[Tuple[float, float]]:
    """Seat centres (local x, local z) on a box with ``size`` = (X, Y, Z), exactly as
    SeatGenerator.ForObject: one seat per full 0.6 m along the longer horizontal
    side (local Z only when Z > X), each centred in its equal segment."""
    sx, _, sz = size
    along_z = sz > sx
    length = abs(sz if along_z else sx)
    count = max(1, int(math.floor(length / SEAT_SPACING + 1e-6)))
    out = []
    for i in range(count):
        offset = -length / 2.0 + (i + 0.5) * length / count
        out.append((0.0, offset) if along_z else (offset, 0.0))
    return out


def generate_seats(object_id: str, position: Sequence[float], yaw: float, size: Sequence[float],
                   seat_h: float, source: str = Source.FUSED,
                   states: Optional[Sequence[str]] = None) -> List[dict]:
    """Seat dicts for a sittable object whose bottom-face centre is ``position``."""
    px, py, pz = position
    seats = []
    for i, (lx, lz) in enumerate(seat_layout(size)):
        wx, wz = local_to_world(px, pz, yaw, lx, lz)
        seats.append({
            "Id": f"{object_id or 'obj'}#{i}",
            "ObjectId": object_id,
            "Pose": pose_json(wx, py + seat_h, wz, yaw),
            "Height": _r(seat_h),
            "State": states[i] if states else SeatState.AVAILABLE,
            "Source": source,
            "Enabled": True,
        })
    return seats


def seat_areas(cx, cz, yaw, size, p: SemanticsParams) -> List[Obb]:
    """Footprint of each seat's sitting area (front part of the seat, backrest excluded)."""
    sx, _, sz = size
    layout = seat_layout(size)
    along_z = sz > sx
    seg = (sz if along_z else sx) / len(layout)
    depth = sx if along_z else sz
    out = []
    for lx, lz in layout:
        wx, wz = local_to_world(cx, cz, yaw, lx, lz)
        if along_z:
            out.append(Obb(wx, wz, yaw, depth * p.seat_area_depth_frac, seg * p.seat_area_len_frac, 0, 0))
        else:
            out.append(Obb(wx, wz, yaw, seg * p.seat_area_len_frac, depth * p.seat_area_depth_frac, 0, 0))
    return out


def seat_occupancy(areas: Sequence[Obb], seat_h: float, hm: Optional[HeightMap],
                   user_head: Optional[Sequence[float]], floor_y: float,
                   p: SemanticsParams) -> List[str]:
    """Available / Blocked (something > blocked_rise above the seat over enough of its
    area) / OccupiedByUser (head at seated height above the seat)."""
    states = []
    for a in areas:
        if user_head is not None:
            hx, hy, hz = user_head
            if (p.head_h[0] <= hy - floor_y <= p.head_h[1]
                    and math.hypot(hx - a.cx, hz - a.cz) <= p.head_seat_radius):
                states.append(SeatState.OCCUPIED_BY_USER)
                continue
        if hm is None:
            states.append(SeatState.AVAILABLE)
            continue
        rel, known = heightmap_cells(hm, a)
        if known.sum() == 0:
            states.append(SeatState.AVAILABLE)
            continue
        with np.errstate(invalid="ignore"):
            high = (rel[known] > seat_h + p.blocked_rise).mean()
        states.append(SeatState.BLOCKED if high > p.blocked_fraction else SeatState.AVAILABLE)
    return states
