"""Read-only view of the fused map that the semantics layer consumes.

The fusion layer implements :class:`MapView`; :class:`~.synthetic.SyntheticRoom`
is a pure-numpy stand-in for tests. Unity left-handed stage space, metres.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Protocol, Tuple

import numpy as np

from .basics import Obb

# NAV_HEIGHTMAP flag bits (CONTRACT-roomd.md)
FLAG_KNOWN = 1
FLAG_OBSTACLE = 2
FLAG_WALKABLE = 4


class HeightMap(NamedTuple):
    """2.5D map. Arrays are ``[h, w]``: row ``iz``, column ``ix``; cell (iz, ix) has
    its centre at ``(origin_x + (ix + 0.5) * cell, origin_z + (iz + 0.5) * cell)``.
    Unknown cells have ``flags & FLAG_KNOWN == 0`` (heights may be NaN)."""
    cell: float
    origin_x: float
    origin_z: float
    floor_y: np.ndarray
    top_y: np.ndarray
    flags: np.ndarray


class MapView(Protocol):
    def floor_plane(self) -> Tuple[float, float]:
        """(floor y, plane-fit rms) in metres."""

    def heightmap(self) -> HeightMap:
        """Current height map or None (a 6-tuple or same-named attributes are accepted)."""

    def surface_points(self, min_y: float, max_y: float,
                       region: Optional[Obb] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Surface samples (N x 3) and unit normals (N x 3) with min_y <= y <= max_y,
        optionally restricted to the footprint of ``region``."""

    def free_fraction(self, obb: Obb) -> float:
        """Fraction of the box volume observed as free space (0..1)."""

    def occupied_fraction(self, obb: Obb) -> float:
        """Fraction of the box volume observed as occupied (0..1)."""

    def visible_fraction(self, obb: Obb) -> float:
        """Fraction of the box volume that is observed at all (free + occupied)."""


def as_heightmap(hm) -> Optional[HeightMap]:
    """HeightMap from a HeightMap, a 6-tuple, or any object with the same attributes
    (e.g. roomd.fusion Heightmap); None stays None (no map yet)."""
    if hm is None or isinstance(hm, HeightMap):
        return hm
    if hasattr(hm, "top_y"):
        return HeightMap(hm.cell, hm.origin_x, hm.origin_z, np.asarray(hm.floor_y),
                         np.asarray(hm.top_y), np.asarray(hm.flags))
    return HeightMap(*hm)


def heightmap_cells(hm: HeightMap, obb: Obb):
    """(rel_top, known) for cells whose centres lie in the footprint of ``obb``.
    ``rel_top`` = top_y - floor_y."""
    corners = obb.corners()
    x0, z0 = corners.min(axis=0)
    x1, z1 = corners.max(axis=0)
    h, w = hm.top_y.shape
    ix0 = max(0, int(np.floor((x0 - hm.origin_x) / hm.cell)))
    ix1 = min(w - 1, int(np.floor((x1 - hm.origin_x) / hm.cell)))
    iz0 = max(0, int(np.floor((z0 - hm.origin_z) / hm.cell)))
    iz1 = min(h - 1, int(np.floor((z1 - hm.origin_z) / hm.cell)))
    if ix1 < ix0 or iz1 < iz0:
        return np.empty(0), np.empty(0, dtype=bool)
    iz, ix = np.mgrid[iz0:iz1 + 1, ix0:ix1 + 1]
    x = hm.origin_x + (ix + 0.5) * hm.cell
    z = hm.origin_z + (iz + 0.5) * hm.cell
    inside = obb.contains_xz(x, z)
    iz, ix = iz[inside], ix[inside]
    known = (hm.flags[iz, ix] & FLAG_KNOWN) != 0
    rel = hm.top_y[iz, ix].astype(float) - hm.floor_y[iz, ix].astype(float)
    return rel, known
