"""MSG_DEPTH_FRAME_V2 -> per-view metric depth, pinhole intrinsics and stage poses.

Payload layout: alvr-realkk REALKK.md. View poses arrive in OpenXR STAGE space
(right-handed, +Y up, qx qy qz qw px py pz); CONTRACT-roomd.md maps them to Unity
left-handed stage space with position (x, y, -z) and quaternion (-qx, -qy, qz, qw).

Depth is stored as GL-style D16: d in [0, 1], view depth z (metres along -Z of the view).
Samples 0 and 0xFFFF carry no measurement and decode to 0 ("invalid").
"""

import math
import struct
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

MSG_DEPTH_FRAME_V2 = 3
FORMAT_RAW_D16 = 0
FORMAT_H264 = 1
FORMAT_LZ4_D16 = 2
MIN_HEADER_SIZE = 152
FAKE_SAMPLE = 0x8080  # failed GPU readback fills the whole frame with 0x80 bytes


def xr_to_unity_position(p):
    return np.array([p[0], p[1], -p[2]], dtype=np.float64)


def xr_to_unity_quat(q):
    return np.array([-q[0], -q[1], q[2], q[3]], dtype=np.float64)


def quat_to_matrix(q):
    x, y, z, w = (float(c) for c in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def d16_to_metric(raw, near, far):
    """uint16 D16 -> view depth in metres (float32); 0 and 0xFFFF -> 0 (invalid)."""
    raw = np.asarray(raw)
    d = raw.astype(np.float64) / 65535.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if math.isinf(far):
            z = near / (1.0 - d)
        else:
            z = 2.0 * near * far / ((far + near) - (2.0 * d - 1.0) * (far - near))
    z[(raw == 0) | (raw == 0xFFFF) | ~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def metric_to_d16(z, near, far):
    """Inverse of d16_to_metric; non-finite or non-positive z -> 0xFFFF (no return)."""
    z = np.asarray(z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if math.isinf(far):
            d = 1.0 - near / z
        else:
            d = ((far + near) - 2.0 * near * far / z) / (far - near) * 0.5 + 0.5
    raw = np.clip(np.round(d * 65535.0), 1, 65534)
    raw[~np.isfinite(z) | (z <= 0) | ~np.isfinite(d) | (d >= 1.0)] = 0xFFFF
    raw[np.isfinite(z) & (z > 0) & (z < near)] = 0
    return raw.astype(np.uint16)


@dataclass
class DepthView:
    depth: np.ndarray           # (H, W) float32 view depth in metres, 0 = invalid
    fx: float
    fy: float
    cx: float
    cy: float
    position_xr: np.ndarray     # OpenXR stage
    rotation_xr: np.ndarray     # xyzw, OpenXR stage
    position_unity: np.ndarray  # Unity left-handed stage
    rotation_unity: np.ndarray  # xyzw, Unity left-handed stage

    @property
    def width(self):
        return self.depth.shape[1]

    @property
    def height(self):
        return self.depth.shape[0]


@dataclass
class DepthFrame:
    views: List[DepthView]
    client_ts_ns: int
    server_ts_ns: Optional[int]
    near: float
    far: float

    @property
    def head_position_unity(self):
        return np.mean([v.position_unity for v in self.views], axis=0)


def parse_header(payload) -> dict:
    """MSG_DEPTH_FRAME_V2 header as a dict with depth_listener.parse_depth_v2's keys."""
    if len(payload) < MIN_HEADER_SIZE:
        raise ValueError("depth payload too short (%d bytes)" % len(payload))
    header_size = struct.unpack_from("<I", payload, 0)[0]
    if header_size < MIN_HEADER_SIZE or header_size > len(payload):
        raise ValueError("bad depth header_size %d" % header_size)
    h = {"header_size": header_size}
    h["client_timestamp_ns"] = struct.unpack_from("<Q", payload, 4)[0]
    h["view_poses"] = [struct.unpack_from("<7f", payload, off) for off in (12, 40)]
    h["width"], h["height"] = struct.unpack_from("<II", payload, 68)
    h["near_z"], h["far_z"] = struct.unpack_from("<ff", payload, 76)
    h["format"] = struct.unpack_from("<I", payload, 84)[0]
    h["fov_angles"] = [struct.unpack_from("<4f", payload, off) for off in (88, 104)]
    h["intrinsics"] = [struct.unpack_from("<4f", payload, off) for off in (120, 136)]
    if header_size >= 160:
        h["server_timestamp_unix_ns"] = struct.unpack_from("<Q", payload, 152)[0]
    return h


def unpack_d16(header, data):
    """Pixel data after the header -> stacked (height, width) uint16 D16 image."""
    width, height, fmt = header["width"], header["height"], header["format"]
    if width == 0 or height == 0 or height % 2:
        raise ValueError("bad stacked depth size %dx%d" % (width, height))
    if fmt == FORMAT_LZ4_D16:
        import lz4.block

        if len(data) < 4:
            raise ValueError("truncated LZ4 depth data")
        size = struct.unpack_from("<I", data, 0)[0]
        if size != width * height * 2:
            raise ValueError("LZ4 size %d does not match %dx%d" % (size, width, height))
        try:
            raw = lz4.block.decompress(bytes(data[4:]), uncompressed_size=size)
        except Exception as e:  # lz4 raises LZ4BlockError
            raise ValueError("bad LZ4 depth data: %s" % e) from e
    elif fmt == FORMAT_RAW_D16:
        raw = data
        if len(raw) != width * height * 2:
            raise ValueError("raw depth size %d does not match %dx%d" % (len(raw), width, height))
    else:
        raise ValueError("unsupported depth format %d" % fmt)
    return np.frombuffer(raw, dtype="<u2").reshape(height, width)


def frame_from_header(header, raw, flip_rows=False) -> Optional[DepthFrame]:
    """Header dict (parse_header / depth_listener.parse_depth_v2) + stacked D16 image.

    Returns None for a fake (all-0x80) frame. flip_rows: the image rows are bottom-up;
    each view image is flipped vertically before the pinhole model (u right, v down).
    """
    raw = np.asarray(raw)
    if np.all(raw == FAKE_SAMPLE):
        return None
    near, far = header["near_z"], header["far_z"]
    if not (near > 0) or math.isnan(far) or far <= near:
        raise ValueError("bad near/far %r/%r" % (near, far))
    half = raw.shape[0] // 2
    views = []
    for i in range(2):
        img = raw[i * half:(i + 1) * half]
        if flip_rows:
            img = img[::-1]
        pose = header["view_poses"][i]
        q = pose[:4]
        p = pose[4:]
        fx, fy, cx, cy = header["intrinsics"][i]
        views.append(DepthView(
            depth=np.ascontiguousarray(d16_to_metric(img, near, far)),
            fx=fx, fy=fy, cx=cx, cy=cy,
            position_xr=np.array(p, dtype=np.float64), rotation_xr=np.array(q, dtype=np.float64),
            position_unity=xr_to_unity_position(p), rotation_unity=xr_to_unity_quat(q)))
    return DepthFrame(views=views, client_ts_ns=header["client_timestamp_ns"],
                      server_ts_ns=header.get("server_timestamp_unix_ns"), near=near, far=far)


def decode_depth_frame(payload, flip_rows=False) -> Optional[DepthFrame]:
    """Decode a MSG_DEPTH_FRAME_V2 payload. Returns None for a fake (all-0x80) frame."""
    header = parse_header(payload)
    return frame_from_header(header, unpack_d16(header, payload[header["header_size"]:]), flip_rows)


def view_rays(view):
    """Per-pixel OpenXR camera-space ray (x, y, -1) scaled so that the view depth is 1."""
    v, u = np.mgrid[0:view.height, 0:view.width].astype(np.float64)
    x = (u + 0.5 - view.cx) / view.fx
    y = -(v + 0.5 - view.cy) / view.fy
    return np.stack([x, y, -np.ones_like(x)], axis=-1)


def backproject_unity(view, depth=None):
    """Valid pixels of a view -> (N, 3) points in Unity stage space."""
    depth = view.depth if depth is None else depth
    rays = view_rays(view)
    valid = depth > 0
    cam = rays[valid] * depth[valid][:, None]
    world_xr = cam @ quat_to_matrix(view.rotation_xr).T + view.position_xr
    world_xr[:, 2] *= -1.0
    return world_xr
