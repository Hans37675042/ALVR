"""Stand-ins for roomd.sink / roomd.protocol (owned by the roomd-io slice) so the fusion sink
can be tested on a branch that does not have them yet. Field names mirror the io slice;
when the real modules are importable they are used instead."""

import importlib
import sys
import types
from dataclasses import dataclass, field

import numpy as np


def ensure_io_modules():
    try:
        importlib.import_module("roomd.sink")
        importlib.import_module("roomd.protocol")
        return False
    except ImportError:
        pass
    sink = types.ModuleType("roomd.sink")
    protocol = types.ModuleType("roomd.protocol")

    @dataclass
    class ScenePrior:
        snapshot_id: int
        floor_y: float
        objects: list
        walls: list
        mesh_vertices: object = None
        mesh_triangles: object = None

    @dataclass
    class FusionOutputs:
        map_revision: int = 0
        floor_y: float = None
        floor_rms: float = None
        objects: list = None
        seats: list = None
        walls: list = None
        nav_heightmap: object = None
        nav_revision: int = 0
        mesh_chunks: list = field(default_factory=list)

    class FusionSink:
        name = "noop"

        def integrate(self, frame):
            pass

        def snapshot_outputs(self):
            return FusionOutputs()

        def set_scene_prior(self, prior):
            pass

        def on_playspace_changed(self, recenter_pose):
            pass

        def close(self):
            pass

    @dataclass
    class NavHeightmap:
        cell: float
        origin_x: float
        origin_z: float
        floor_y: np.ndarray
        top_y: np.ndarray
        flags: np.ndarray

    @dataclass
    class MeshChunk:
        ix: int
        iy: int
        iz: int
        revision: int
        chunk_size: float
        vertices: np.ndarray
        indices: np.ndarray

    @dataclass
    class DepthFrame:
        header: dict
        data: bytes
        recv_ns: int = 0

        def d16(self):
            from roomd.fusion.decode import unpack_d16
            return unpack_d16(self.header, self.data)

    def decode_depth_v2(payload, recv_ns=0):
        from roomd.fusion.decode import parse_header
        header = parse_header(payload)
        return DepthFrame(header, payload[header["header_size"]:], recv_ns)

    sink.ScenePrior, sink.FusionOutputs, sink.FusionSink = ScenePrior, FusionOutputs, FusionSink
    protocol.NavHeightmap, protocol.MeshChunk, protocol.DepthFrame = NavHeightmap, MeshChunk, DepthFrame
    protocol.decode_depth_v2 = decode_depth_v2
    sys.modules["roomd.sink"] = sink
    sys.modules["roomd.protocol"] = protocol
    return True
