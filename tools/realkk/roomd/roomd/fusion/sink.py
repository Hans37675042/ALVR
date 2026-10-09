"""roomd FusionSink backed by TsdfFusion: `roomd --fusion roomd.fusion:create_sink`.

roomd.sink / roomd.protocol belong to the I/O slice; this adapter only maps between their
records and TsdfFusion. Objects, seats and walls come from the semantics layer, so they
stay None here ("not produced").
"""

import math

import numpy as np

from roomd import protocol
from roomd.sink import FusionOutputs, FusionSink

from .config import FusionConfig
from .decode import make_frame
from .outputs import to_protocol_heightmap, to_protocol_mesh_chunk
from .tsdf import TsdfFusion

RECENTER_EPS_M = 1e-3
RECENTER_EPS_RAD = math.radians(0.1)


def _pose_changed(a, b):
    """a, b: (px, py, pz, qx, qy, qz, qw)."""
    if math.dist(a[:3], b[:3]) > RECENTER_EPS_M:
        return True
    dot = abs(sum(x * y for x, y in zip(a[3:], b[3:])))
    return 2 * math.acos(min(1.0, dot)) > RECENTER_EPS_RAD


class TsdfFusionSink(FusionSink):
    name = "tsdf"

    def __init__(self, config=None):
        self.fusion = TsdfFusion(config or FusionConfig())
        self._recenter_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._nav = None
        self._nav_revision = 0

    def integrate(self, frame):
        """protocol.DepthFrame (OpenXR; roomd already dropped fake frames) -> TSDF."""
        flip = self.fusion.config.flip_rows
        depths = [frame.view_z(i)[::-1] if flip else frame.view_z(i) for i in range(2)]
        views = frame.views
        self.fusion.integrate(make_frame(depths, [v.pose_xr for v in views], [v.intrinsics for v in views],
                                         frame.near, frame.far, frame.header.get("client_timestamp_ns", 0)))

    def snapshot_outputs(self):
        snap = self.fusion.snapshot_outputs()
        if snap.heightmap is not None:
            hm = snap.heightmap
            self._nav = to_protocol_heightmap(hm)
            self._nav_revision += 1
        chunks = [to_protocol_mesh_chunk(c) for c in snap.mesh_chunks]
        return FusionOutputs(map_revision=snap.stats["mapRevision"],
                             floor_y=None if snap.floor is None else snap.floor.y,
                             floor_rms=None if snap.floor is None else snap.floor.rms,
                             nav_heightmap=self._nav, nav_revision=self._nav_revision, mesh_chunks=chunks)

    def set_scene_prior(self, prior):
        """Scene API floor sets the volume's floor (before the volume exists); GLOBAL_MESH
        goes in with the low prior weight."""
        if self.fusion.bounds() is None and prior.floor_y is not None:
            self.fusion.config.floor_y = float(prior.floor_y)
        if prior.mesh_vertices is not None and prior.mesh_triangles is not None and len(prior.mesh_triangles):
            self.fusion.integrate_prior_mesh(prior.mesh_vertices, prior.mesh_triangles)

    def on_playspace_changed(self, recenter_pose):
        """A new recenter pose moves STAGE under the map: drop it and start over."""
        pose = tuple(float(x) for x in recenter_pose)
        changed = _pose_changed(pose, self._recenter_pose)
        self._recenter_pose = pose
        if not changed:
            return
        self.fusion.reset()
        if self._nav is not None:
            # everything we published is in the old frame: publish "nothing known"
            self._nav = protocol.NavHeightmap(self._nav.cell, self._nav.origin_x, self._nav.origin_z,
                                              np.zeros_like(self._nav.floor_y), np.zeros_like(self._nav.top_y),
                                              np.zeros_like(self._nav.flags))
            self._nav_revision += 1


def create_sink():
    return TsdfFusionSink()
