"""Fusion output records (internal) and their mapping onto roomd.protocol / 9945 payloads.

Payloads exclude the `u32 type, u32 len` frame; roomd.protocol owns the wire format.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

MSG_NAV_HEIGHTMAP = 2
MSG_MESH_CHUNK = 3
FLAG_KNOWN = 1
FLAG_OBSTACLE = 2
FLAG_WALKABLE = 4


@dataclass
class MeshChunk:
    """One 1 m chunk of the fused mesh. Vertices are absolute Unity stage coordinates; the
    chunk covers [i*chunk_size, (i+1)*chunk_size) per axis plus a one-voxel seam overlap.
    Triangles use Unity's front-face convention (clockwise seen from free space).
    Empty vertices/indices = the chunk was removed."""
    ix: int
    iy: int
    iz: int
    revision: int
    chunk_size: float
    vertices: np.ndarray  # (N, 3) float32
    indices: np.ndarray   # (M,) uint32, M % 3 == 0


@dataclass
class Heightmap:
    """2.5D map. Cell (x, z) covers [origin_x + x*cell, +cell) x [origin_z + z*cell, +cell);
    arrays are indexed [z, x] (row-major, row = z), matching the payload order."""
    cell: float
    origin_x: float
    origin_z: float
    width: int
    height: int
    floor_y: np.ndarray  # (height, width) float32
    top_y: np.ndarray    # (height, width) float32; == floor_y where nothing stands
    flags: np.ndarray    # (height, width) uint8: bit0 known, bit1 obstacle, bit2 walkable

    def index(self, x, z) -> Optional[Tuple[int, int]]:
        cx = int(np.floor((x - self.origin_x) / self.cell))
        cz = int(np.floor((z - self.origin_z) / self.cell))
        if 0 <= cx < self.width and 0 <= cz < self.height:
            return cz, cx
        return None

    def sample(self, x, z):
        """(floor_y, top_y, flags) of the cell containing (x, z); flags 0 outside the map."""
        ij = self.index(x, z)
        if ij is None:
            return float("nan"), float("nan"), 0
        return float(self.floor_y[ij]), float(self.top_y[ij]), int(self.flags[ij])


@dataclass
class FloorPlane:
    """Fitted floor: y at the inlier centroid, unit normal (Unity), residual RMS, inliers."""
    y: float
    normal: Tuple[float, float, float]
    rms: float
    count: int


@dataclass
class FusionOutputs:
    mesh_chunks: List[MeshChunk] = field(default_factory=list)
    heightmap: Optional[Heightmap] = None
    floor: Optional[FloorPlane] = None
    stats: dict = field(default_factory=dict)


def encode_mesh_chunk(chunk: MeshChunk) -> bytes:
    """Contract payload; thin wrapper over roomd.protocol (owns the wire format)."""
    from roomd import protocol

    return protocol.encode_mesh_chunk(chunk)


def decode_mesh_chunk(payload: bytes) -> MeshChunk:
    from roomd import protocol

    c = protocol.decode_mesh_chunk(payload)
    return MeshChunk(c.ix, c.iy, c.iz, c.revision, c.chunk_size, c.vertices, c.indices)


def encode_nav_heightmap(hm: Heightmap) -> bytes:
    from roomd import protocol

    return protocol.encode_nav_heightmap(to_protocol_heightmap(hm))


def decode_nav_heightmap(payload: bytes) -> Heightmap:
    from roomd import protocol

    p = protocol.decode_nav_heightmap(payload)
    return Heightmap(p.cell, p.origin_x, p.origin_z, p.width, p.height, p.floor_y.astype(np.float32),
                     p.top_y.astype(np.float32), p.flags)


def to_protocol_heightmap(hm: Heightmap):
    from roomd import protocol

    return protocol.NavHeightmap(hm.cell, hm.origin_x, hm.origin_z, np.asarray(hm.floor_y, np.float16),
                                 np.asarray(hm.top_y, np.float16), np.asarray(hm.flags, np.uint8))


def to_protocol_mesh_chunk(c: MeshChunk):
    from roomd import protocol

    return protocol.MeshChunk(c.ix, c.iy, c.iz, c.revision, c.chunk_size, c.vertices, c.indices)
