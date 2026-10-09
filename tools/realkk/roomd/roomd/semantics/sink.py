"""FusionSink that adds semantics on top of the fusion sink:
``roomd --fusion roomd.semantics:create_sink``.

The wrapped sink keeps doing depth integration, mesh chunks and the nav heightmap;
``snapshot_outputs`` additionally fills ``objects`` / ``seats`` / ``walls`` with
roomd.model records. Tracked Scene API objects (``scene:<uuid>``, pose following the
fused geometry) are published too; roomd's service prefers them over its own scene
copy with the same id. ``publish_scene_objects=False`` leaves scene objects to roomd.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from roomd import model as M
from roomd.coords import flip_position
from roomd.sink import FusionOutputs, FusionSink

from .basics import Obb, SceneLabel, quat_yaw
from .fusion_view import FusionMapView
from .params import SemanticsParams
from .tracker import RoomSemantics


def scene_labels_from_objects(objects):
    """roomd.model SceneApi objects (Unity stage, bottom-face pose) -> SceneLabel."""
    out = []
    for o in objects or []:
        if o is None or o.Source != M.SOURCE_SCENE or not (o.Id or "").startswith("scene:"):
            continue
        p = o.Pose.Position
        yaw = quat_yaw(o.Pose.Rotation.tuple())
        kind = o.Kind.value if hasattr(o.Kind, "value") else str(o.Kind)
        out.append(SceneLabel(o.Id[len("scene:"):], o.Label or kind.upper(),
                              Obb(p.X, p.Z, yaw, o.Size.X, o.Size.Z, p.Y, p.Y + o.Size.Y)))
    return out


CAP_MARGIN = 0.05  # m below the fusion heightmap cap that counts as "at the cap"


class SemanticsSink(FusionSink):
    name = "semantics"

    def __init__(self, inner, params: Optional[SemanticsParams] = None, clock=time.time,
                 obb_cls=None, publish_scene_objects: bool = True):
        self.inner = inner
        if params is None:
            params = SemanticsParams()
            cap = getattr(getattr(inner.fusion, "config", None), "obstacle_max_height", None)
            if cap is not None:
                # the heightmap stops at the cap: walls and tall cabinets both read as cap,
                # so the structure cut never sits above it
                params.structure_min_h = min(params.structure_min_h, float(cap) - CAP_MARGIN)
        self.params = params
        self.clock = clock
        self.publish_scene_objects = publish_scene_objects
        self.view = FusionMapView(inner.fusion, obb_cls, up_normal_min=params.up_normal_min)
        self.semantics = RoomSemantics(params)
        self._labels = []
        self._prior_walls = None
        self._head = None

    def integrate(self, frame):
        self.inner.integrate(frame)
        try:
            # head ~ midpoint of the two depth view poses (OpenXR -> Unity)
            pos = np.array([v.pose_xr[:3] for v in frame.views], dtype=float)
            self._head = tuple(float(x) for x in flip_position(pos.mean(axis=0)))
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            pass

    def snapshot_outputs(self):
        out = self.inner.snapshot_outputs() or FusionOutputs()
        if out.floor_y is None or self.view.floor_plane() is None:
            return out
        frag = self.semantics.update(self.view, self.clock(), self._labels, self._head)
        room = M.room_from_dict(dict(frag, Version=M.CURRENT_VERSION))
        # Geometry-only "Other" objects are not published: obstacles reach the plugin through
        # NAV_HEIGHTMAP and occlusion through MESH_CHUNK; they stay tracked here so they
        # still absorb clutter. Scene API "Other" objects (labelled) are published.
        objects = [o for o in room.Objects
                   if (self.publish_scene_objects or o.Source != M.SOURCE_SCENE)
                   and not (o.Source == M.SOURCE_FUSED and o.Kind == M.Kind.Other)]
        ids = {o.Id for o in objects}
        out.objects = objects
        out.seats = [s for s in room.Seats if s.ObjectId in ids]
        out.walls = room.Walls or (list(self._prior_walls) if self._prior_walls else None)
        return out

    def set_scene_prior(self, prior):
        self.inner.set_scene_prior(prior)
        self._labels = scene_labels_from_objects(prior.objects)
        self._prior_walls = list(prior.walls or [])

    def on_playspace_changed(self, recenter_pose):
        self.inner.on_playspace_changed(recenter_pose)
        # the map is rebuilt in a new frame: start tracking over
        self.semantics = RoomSemantics(self.params)
        self._head = None

    def close(self):
        self.inner.close()


def create_sink():
    """Fusion sink + semantics (imports roomd.fusion lazily: it needs warp)."""
    from roomd.fusion import create_sink as fusion_sink

    return SemanticsSink(fusion_sink())
