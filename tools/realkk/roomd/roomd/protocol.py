"""Wire formats of CONTRACT-roomd.md (9944 relay and 9945 plugin sockets).

Every frame on both sockets is little-endian `u32 type, u32 len, payload[len]`.
The depth header and room snapshot parsers and the .rktap format live in tools/realkk
(depth_listener.py, rktap.py; stdlib-only) and are reused here, not copied.
"""

import json
import math
import struct
import uuid as _uuid
from dataclasses import dataclass, field

import numpy as np

from ._legacy import depth_listener, rktap, tap_inspect

# --- 9944 relay (ALVR server <-> roomd) ---
MSG_CAMERA_FRAME = depth_listener.MSG_CAMERA_FRAME  # 2
MSG_DEPTH_FRAME_V2 = depth_listener.MSG_DEPTH_FRAME_V2  # 3
MSG_ROOM_SNAPSHOT = depth_listener.MSG_ROOM_SNAPSHOT  # 4
MSG_PLAYSPACE_CHANGED = depth_listener.MSG_PLAYSPACE_CHANGED  # 5
MSG_STREAM_CONTROL = depth_listener.MSG_STREAM_CONTROL  # 100, roomd -> ALVR
MSG_ROOM_REQUEST = depth_listener.MSG_ROOM_REQUEST  # 101, roomd -> ALVR
MSG_MARKER = rktap.MSG_MARKER  # recorder only, never on the wire
STREAM_DEPTH = depth_listener.STREAM_DEPTH
STREAM_CAMERA = depth_listener.STREAM_CAMERA

FORMAT_RAW_D16 = 0
FORMAT_H264 = 1
FORMAT_LZ4_D16 = 2
DEPTH_HEADER_SIZE = 176

# --- 9945 plugin (roomd <-> KKS plugin) ---
ROOM_MODEL = 1
NAV_HEIGHTMAP = 2
MESH_CHUNK = 3
STATUS = 4
PLUGIN_HELLO = 101  # plugin -> roomd

NAV_FLAG_KNOWN = 1
NAV_FLAG_OBSTACLE = 2
NAV_FLAG_WALKABLE = 4

_FRAME = struct.Struct("<II")


def pack_frame(msg_type, payload):
    return _FRAME.pack(msg_type, len(payload)) + payload


def unpack_frame(buf):
    """(type, payload, rest) for the first complete frame in buf, or None if incomplete."""
    if len(buf) < _FRAME.size:
        return None
    msg_type, length = _FRAME.unpack_from(buf, 0)
    end = _FRAME.size + length
    if len(buf) < end:
        return None
    return msg_type, bytes(buf[_FRAME.size:end]), bytes(buf[end:])


def recv_exact(sock, n):
    return depth_listener.recv_exact(sock, n)


def read_frame(sock):
    """Blocking read of one frame; raises ConnectionError on EOF."""
    msg_type, length = _FRAME.unpack(recv_exact(sock, _FRAME.size))
    return msg_type, recv_exact(sock, length) if length else b""


# --- stream control / room request ---

def encode_stream_control(stream_id, enabled):
    return bytes([stream_id, 1 if enabled else 0])


def decode_stream_control(payload):
    return payload[0], payload[1] != 0


def encode_room_request(recapture):
    return bytes([1 if recapture else 0])


def decode_room_request(payload):
    return bool(payload[0])


# --- depth V2 ---

def fov_to_intrinsics(fov, width, height):
    """alvr_packets::fov_to_pinhole_intrinsics (image y down); fov = (l, r, u, d) radians."""
    l, r, u, d = (math.tan(a) for a in fov)
    fx = width / (r - l)
    fy = height / (u - d)
    return (fx, fy, -l * fx, u * fy)


def d16_to_z(d16, near, far):
    """Linear view depth (metres) from D16 (standard GL projection, d in [0, 1]).
    d = 1 maps to +inf (no measurement)."""
    d = np.asarray(d16, dtype=np.float64) / 65535.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if math.isinf(far):
            z = near / (1.0 - d)
        else:
            z = 2 * near * far / ((far + near) - (2 * d - 1) * (far - near))
    return np.where(d >= 1.0, np.inf, z)


def encode_d16_from_z(z, near, far):
    """Inverse of d16_to_z; z <= near clamps to 0, inf/nan maps to 65535."""
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if math.isinf(far):
            d = 1.0 - near / z
        else:
            d = ((far + near) - 2 * near * far / z) / (far - near) * 0.5 + 0.5
    d = np.where(np.isfinite(z), d, 1.0)
    return np.clip(np.round(d * 65535.0), 0, 65535).astype(np.uint16)


@dataclass
class DepthView:
    pose_xr: tuple  # (px,py,pz,qx,qy,qz,qw), OpenXR stage, recentered
    fov: tuple  # (l, r, u, d) radians
    intrinsics: tuple  # (fx, fy, cx, cy) pixels, per-view image


@dataclass
class DepthFrame:
    """A decoded MSG_DEPTH_FRAME_V2 header. Pixels stay compressed until d16() is called."""

    header: dict
    data: bytes
    recv_ns: int = 0
    _d16: np.ndarray = field(default=None, repr=False)

    @property
    def views(self):
        out = []
        for i in range(2):
            qx, qy, qz, qw, px, py, pz = self.header["view_poses"][i]
            out.append(DepthView((px, py, pz, qx, qy, qz, qw), tuple(self.header["fov_angles"][i]),
                                 tuple(self.header["intrinsics"][i])))
        return out

    @property
    def width(self):
        return self.header["width"]

    @property
    def view_height(self):
        return self.header["height"] // 2

    @property
    def near(self):
        return self.header["near_z"]

    @property
    def far(self):
        return self.header["far_z"]

    def d16(self):
        """Stacked (height, width) uint16 image; view 0 on top."""
        if self._d16 is None:
            self._d16 = depth_listener.decode_d16(self.header, self.data)
        return self._d16

    def view_d16(self, i):
        h = self.view_height
        return self.d16()[i * h:(i + 1) * h]

    def view_z(self, i):
        """Linear depth (metres along the view axis) of view i; inf where unknown."""
        return d16_to_z(self.view_d16(i), self.near, self.far)

    def is_fake(self):
        """Failed GPU readback on the headset: every decoded byte is 0x80."""
        return tap_inspect.is_fake_frame(self.header, self.data)


def decode_depth_v2(payload, recv_ns=0):
    header, data = depth_listener.parse_depth_v2(payload)
    return DepthFrame(header, data, recv_ns)


def encode_depth_v2(d16, near, far, view_poses, fov_angles, fmt=FORMAT_LZ4_D16, intrinsics=None,
                    client_timestamp_ns=0, server_timestamp_unix_ns=0, clock_offset_ns=0,
                    server_receive_unix_ns=0):
    """Build a MSG_DEPTH_FRAME_V2 payload. d16: stacked (2*h, w) uint16; view_poses are
    in the wire order (qx,qy,qz,qw,px,py,pz)."""
    d16 = np.ascontiguousarray(d16, dtype="<u2")
    height, width = d16.shape
    if intrinsics is None:
        intrinsics = [fov_to_intrinsics(f, width, height // 2) for f in fov_angles]
    raw = d16.tobytes()
    if fmt == FORMAT_LZ4_D16:
        import lz4.block

        data = lz4.block.compress(raw, store_size=True)  # = lz4_flex::compress_prepend_size
    elif fmt == FORMAT_RAW_D16:
        data = raw
    else:
        raise ValueError("unsupported depth format %d" % fmt)
    buf = bytearray(DEPTH_HEADER_SIZE)
    struct.pack_into("<IQ", buf, 0, DEPTH_HEADER_SIZE, client_timestamp_ns)
    struct.pack_into("<7f", buf, 12, *view_poses[0])
    struct.pack_into("<7f", buf, 40, *view_poses[1])
    struct.pack_into("<IIffI", buf, 68, width, height, near, far, fmt)
    struct.pack_into("<4f", buf, 88, *fov_angles[0])
    struct.pack_into("<4f", buf, 104, *fov_angles[1])
    struct.pack_into("<4f", buf, 120, *intrinsics[0])
    struct.pack_into("<4f", buf, 136, *intrinsics[1])
    struct.pack_into("<QqQ", buf, 152, server_timestamp_unix_ns, clock_offset_ns, server_receive_unix_ns)
    return bytes(buf) + data


# --- room snapshot (type 4) / playspace changed (type 5) ---

SNAPSHOT_VERSION = 1
# header_size, version, snapshot_id, recenter_pose; header_size = offset of json_len (40)
_SNAP_HEAD = struct.Struct("<III7f")


def uuid_str(raw16):
    return str(_uuid.UUID(bytes=bytes(raw16)))


def uuid_bytes(text):
    return _uuid.UUID(text).bytes


@dataclass
class SnapshotMesh:
    anchor_uuid: bytes  # 16 raw bytes
    pose: tuple  # mesh anchor pose (px,py,pz,qx,qy,qz,qw), OpenXR stage
    vertices: np.ndarray  # (n, 3) float32, anchor-local
    indices: np.ndarray  # (m,) uint32

    @property
    def uuid(self):
        return uuid_str(self.anchor_uuid)


@dataclass
class RoomSnapshot:
    snapshot_id: int
    recenter_pose: tuple
    scene: dict  # {"rooms": [...], "anchors": [...]}
    meshes: list = field(default_factory=list)
    version: int = SNAPSHOT_VERSION


def encode_room_snapshot(snap):
    js = json.dumps(snap.scene, ensure_ascii=False).encode("utf-8")
    parts = [_SNAP_HEAD.pack(_SNAP_HEAD.size, snap.version, snap.snapshot_id, *snap.recenter_pose),
             struct.pack("<I", len(js)), js, struct.pack("<I", len(snap.meshes))]
    for m in snap.meshes:
        v = np.ascontiguousarray(m.vertices, dtype="<f4").reshape(-1, 3)
        i = np.ascontiguousarray(m.indices, dtype="<u4").reshape(-1)
        parts += [bytes(m.anchor_uuid), struct.pack("<7fII", *m.pose, len(v), len(i)),
                  v.tobytes(), i.tobytes()]
    return b"".join(parts)


def decode_room_snapshot(payload):
    """Layout parsing is depth_listener.parse_room_snapshot (stdlib-only, shared with the
    listener); this wraps the mesh data in numpy arrays."""
    raw = depth_listener.parse_room_snapshot(payload, unpack_mesh=False)
    if raw["version"] != SNAPSHOT_VERSION:
        raise ValueError("unsupported room snapshot version %d" % raw["version"])
    meshes = [SnapshotMesh(uuid_bytes(m["anchor_uuid"]), m["pose"],
                           np.frombuffer(m["vertex_data"], dtype="<f4").reshape(m["vcount"], 3),
                           np.frombuffer(m["index_data"], dtype="<u4"))
              for m in raw["meshes"]]
    return RoomSnapshot(raw["snapshot_id"], raw["recenter_pose"], raw["scene"], meshes, raw["version"])


def encode_playspace_changed(recenter_pose):
    return struct.pack("<I7f", 1, *recenter_pose)


def decode_playspace_changed(payload):
    return depth_listener.parse_playspace_changed(payload)["recenter_pose"]


# --- 9945 plugin messages ---

def encode_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_json(payload):
    return json.loads(payload.decode("utf-8"))


@dataclass
class NavHeightmap:
    """Row-major grids indexed [iz, ix]; cell (ix, iz) covers x in [origin_x + ix*cell, +cell)
    and z in [origin_z + iz*cell, +cell), Unity stage space."""

    cell: float
    origin_x: float
    origin_z: float
    floor_y: np.ndarray  # (h, w) float16
    top_y: np.ndarray  # (h, w) float16
    flags: np.ndarray  # (h, w) uint8, NAV_FLAG_*

    @property
    def width(self):
        return self.flags.shape[1]

    @property
    def height(self):
        return self.flags.shape[0]


def encode_nav_heightmap(hm):
    h, w = hm.flags.shape
    return (struct.pack("<IfffII", 1, hm.cell, hm.origin_x, hm.origin_z, w, h)
            + np.ascontiguousarray(hm.floor_y, "<f2").tobytes()
            + np.ascontiguousarray(hm.top_y, "<f2").tobytes()
            + np.ascontiguousarray(hm.flags, np.uint8).tobytes())


def decode_nav_heightmap(payload):
    _, cell, ox, oz, w, h = struct.unpack_from("<IfffII", payload, 0)
    off = 24
    n = w * h
    floor = np.frombuffer(payload, "<f2", n, off).reshape(h, w)
    top = np.frombuffer(payload, "<f2", n, off + 2 * n).reshape(h, w)
    flags = np.frombuffer(payload, np.uint8, n, off + 4 * n).reshape(h, w)
    return NavHeightmap(cell, ox, oz, floor, top, flags)


@dataclass
class MeshChunk:
    ix: int
    iy: int
    iz: int
    revision: int
    chunk_size: float
    vertices: np.ndarray  # (n, 3) float32, Unity stage
    indices: np.ndarray  # (m,) uint32, Unity winding

    @property
    def key(self):
        return (self.ix, self.iy, self.iz)

    @property
    def is_removed(self):
        return len(self.vertices) == 0

    @classmethod
    def removed(cls, ix, iy, iz, revision, chunk_size):
        return cls(ix, iy, iz, revision, chunk_size, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32))


def encode_mesh_chunk(c):
    v = np.ascontiguousarray(c.vertices, "<f4").reshape(-1, 3)
    i = np.ascontiguousarray(c.indices, "<u4").reshape(-1)
    return (struct.pack("<IiiiIfII", 1, c.ix, c.iy, c.iz, c.revision, c.chunk_size, len(v), len(i))
            + v.tobytes() + i.tobytes())


def decode_mesh_chunk(payload):
    _, ix, iy, iz, rev, size, vc, ic = struct.unpack_from("<IiiiIfII", payload, 0)
    off = 32
    v = np.frombuffer(payload, "<f4", vc * 3, off).reshape(vc, 3)
    i = np.frombuffer(payload, "<u4", ic, off + vc * 12)
    return MeshChunk(ix, iy, iz, rev, size, v, i)
