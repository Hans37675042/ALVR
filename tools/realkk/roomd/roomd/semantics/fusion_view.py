"""MapView over roomd.fusion's TsdfFusion query API.

TsdfFusion boxes are ``Obb(center, half_extents, yaw)`` with yaw in radians (Unity:
positive turns +Z towards +X), i.e. the same rotation as ours in degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .basics import Obb
from .mapview import as_heightmap


@dataclass
class FusionObb:
    """Fallback with roomd.fusion.tsdf.Obb's fields when the fusion package is absent."""
    center: tuple
    half_extents: tuple
    yaw: float = 0.0


def _default_obb_cls():
    try:
        from roomd.fusion.tsdf import Obb as TsdfObb
        return TsdfObb
    except ImportError:
        return FusionObb


class FusionMapView:
    """Adapts a TsdfFusion-like object to :class:`~.mapview.MapView`."""

    def __init__(self, fusion, obb_cls=None, up_normal_min=None):
        self.fusion = fusion
        self._obb_cls = obb_cls
        # TsdfFusion filters surface points by config.upward_min_normal_y (0.85); its
        # gradient normals are noisy, so semantics queries with a looser value.
        self.up_normal_min = up_normal_min

    def to_fusion_obb(self, o: Obb):
        if self._obb_cls is None:
            self._obb_cls = _default_obb_cls()
        return self._obb_cls(((o.cx, (o.y0 + o.y1) / 2, o.cz)),
                             (o.sx / 2, (o.y1 - o.y0) / 2, o.sz / 2), math.radians(o.yaw))

    def floor_plane(self):
        fp = self.fusion.floor_plane()
        if fp is None:
            return None
        if isinstance(fp, tuple):
            return fp
        return float(fp.y), float(fp.rms)

    def heightmap(self):
        return as_heightmap(self.fusion.heightmap())

    def surface_points(self, min_y, max_y, region=None):
        box = None if region is None else self.to_fusion_obb(region)
        cfg = getattr(self.fusion, "config", None)
        if self.up_normal_min is None or not hasattr(cfg, "upward_min_normal_y"):
            return self.fusion.surface_points(min_y, max_y, box)
        saved = cfg.upward_min_normal_y
        cfg.upward_min_normal_y = self.up_normal_min
        try:
            return self.fusion.surface_points(min_y, max_y, box)
        finally:
            cfg.upward_min_normal_y = saved

    def free_fraction(self, obb: Obb) -> float:
        return float(self.fusion.free_fraction(self.to_fusion_obb(obb)))

    def occupied_fraction(self, obb: Obb) -> float:
        return float(self.fusion.occupied_fraction(self.to_fusion_obb(obb)))

    def visible_fraction(self, obb: Obb) -> float:
        return float(self.fusion.visible_fraction(self.to_fusion_obb(obb)))
