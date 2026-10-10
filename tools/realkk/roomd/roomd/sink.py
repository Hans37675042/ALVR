"""Pluggable depth consumer (the fusion slice plugs in here).

roomd's main loop is single-threaded: it calls integrate() with the newest real depth
frame (fake all-0x80 frames are already dropped, older frames already skipped) and
snapshot_outputs() about once per second, both from the same thread. An implementation
may integrate on its own worker thread; integrate() should then return quickly.

Coordinates: frames arrive in OpenXR stage space (right-handed, +Y up, metres, already
recentered by ALVR; see protocol.DepthFrame). Everything in FusionOutputs is Unity
left-handed stage space (coords.flip_position / flip_pose / flip_quat convert).
"""

from dataclasses import dataclass, field


@dataclass
class ScenePrior:
    """What the Quest Scene API snapshot says about the room (from scene.prior_from_snapshot)."""

    snapshot_id: int
    floor_y: float  # Unity stage
    objects: list  # model.RoomObject, Source=SceneApi, Unity stage
    walls: list  # model.Wall
    mesh_vertices: object = None  # (N, 3) float32 Unity stage, GLOBAL_MESH; None if absent
    mesh_triangles: object = None  # (M, 3) uint32, Unity winding


@dataclass
class FusionOutputs:
    """State published by the sink. None means "not produced / keep what roomd has"."""

    map_revision: int = 0
    floor_y: float = None
    floor_rms: float = None
    # Source="Fused" objects with stable ids "fused:<n>", and their seats (SeatGenerator ids
    # "<objectId>#<i>"); both None when the sink does not produce objects.
    objects: list = None
    seats: list = None
    walls: list = None
    nav_heightmap: object = None  # protocol.NavHeightmap, latest
    nav_revision: int = 0  # bump when nav_heightmap changed; roomd forwards on change
    # protocol.MeshChunk changed since the previous snapshot_outputs() call (drained)
    mesh_chunks: list = field(default_factory=list)


class FusionSink:
    """Interface and no-op implementation used until the fusion package is ready."""

    name = "noop"

    def integrate(self, frame):
        """Consume one protocol.DepthFrame."""

    def snapshot_outputs(self):
        return FusionOutputs()

    def set_scene_prior(self, prior):
        """A new Scene API snapshot was converted (ScenePrior)."""

    def on_playspace_changed(self, recenter_pose):
        """ALVR recentered; recenter_pose is (px,py,pz,qx,qy,qz,qw), OpenXR."""

    def reclassify(self, ids):
        """Plugin ROOM_RECLASSIFY: let the classifier decide the kind of these object ids
        again (None = every automatic object). Returns the ids actually unlocked."""
        return []

    def close(self):
        pass


NullSink = FusionSink


def load_sink(spec):
    """'noop' or 'module:factory' (factory() -> FusionSink)."""
    if not spec or spec == "noop":
        return NullSink()
    import importlib

    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr or "create_sink")()
