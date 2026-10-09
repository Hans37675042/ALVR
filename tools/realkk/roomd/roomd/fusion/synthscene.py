"""Analytic synthetic depth scenes (floor + axis-aligned boxes) rendered as MSG_DEPTH_FRAME_V2.

Scene geometry is given in Unity stage space; rendering happens in OpenXR stage space
exactly the way the headset would report it (view poses, D16, stacked views).
Used by the tests and by `replay --synthetic`.
"""

import math
import struct

import numpy as np

from .decode import FORMAT_LZ4_D16, metric_to_d16, quat_to_matrix

HEADER_SIZE = 176


def unity_to_xr(p):
    return np.array([p[0], p[1], -p[2]], dtype=np.float64)


def _matrix_to_quat(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def look_at_xr(eye, target, up=(0.0, 1.0, 0.0)):
    """Quaternion (xyzw) of an OpenXR view at eye whose -Z axis points at target."""
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, (0.0, 0.0, -1.0))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    return _matrix_to_quat(np.stack([right, true_up, -fwd], axis=1))


def encode_payload(raw_stacked, poses_xr, intrinsics, near, far, fmt=FORMAT_LZ4_D16,
                   client_ts=0, server_ts=0, clock_offset=0, server_recv=0):
    """Build a MSG_DEPTH_FRAME_V2 payload from a stacked (2H, W) uint16 D16 image."""
    raw_stacked = np.ascontiguousarray(raw_stacked, dtype="<u2")
    height, width = raw_stacked.shape
    raw = raw_stacked.tobytes()
    if fmt == FORMAT_LZ4_D16:
        import lz4.block

        data = lz4.block.compress(raw, store_size=True)  # = lz4_flex::compress_prepend_size
    else:
        data = raw
    buf = bytearray(HEADER_SIZE)
    struct.pack_into("<I", buf, 0, HEADER_SIZE)
    struct.pack_into("<Q", buf, 4, client_ts)
    for off, (p, q) in zip((12, 40), poses_xr):
        struct.pack_into("<7f", buf, off, *q, *p)
    struct.pack_into("<II", buf, 68, width, height)
    struct.pack_into("<ff", buf, 76, near, far)
    struct.pack_into("<I", buf, 84, fmt)
    for off, (fx, fy, cx, cy) in zip((120, 136), intrinsics):
        h = height // 2
        tl, tr = -cx / fx, (width - cx) / fx
        tu, td = cy / fy, (cy - h) / fy
        struct.pack_into("<4f", buf, off - 32, math.atan(tl), math.atan(tr), math.atan(tu), math.atan(td))
        struct.pack_into("<4f", buf, off, fx, fy, cx, cy)
    struct.pack_into("<QqQ", buf, 152, server_ts, clock_offset, server_recv)
    return bytes(buf) + data


def metric_payload(depths, poses_xr, intrinsics, near, far, fmt=FORMAT_LZ4_D16, flip_rows=False, **kw):
    """Two (H, W) view-depth images in metres -> payload (inf / <= 0 = no return)."""
    imgs = [metric_to_d16(d, near, far) for d in depths]
    if flip_rows:
        imgs = [i[::-1] for i in imgs]
    return encode_payload(np.vstack(imgs), poses_xr, intrinsics, near, far, fmt=fmt, **kw)


def box_mesh(lo, hi):
    """Closed box mesh in Unity space; cross(b-a, c-a) points outwards (Unity front face)."""
    lo = np.asarray(lo, np.float32)
    hi = np.asarray(hi, np.float32)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                 np.float32)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    center = (lo + hi) / 2
    for a, b, c, d in quads:
        for t in ((a, b, c), (a, c, d)):
            n = np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]])
            if np.dot(n, v[list(t)].mean(axis=0) - center) < 0:
                t = (t[0], t[2], t[1])
            tris.append(t)
    return v, np.array(tris, np.uint32).reshape(-1)


class SynthScene:
    """Infinite floor at floor_y plus solid axis-aligned boxes ((min), (max)) in Unity space."""

    def __init__(self, floor_y=0.0, boxes=()):
        self.floor_y = floor_y
        # boxes stored in OpenXR space: z range mirrored
        self.boxes_xr = []
        for lo, hi in boxes:
            self.boxes_xr.append((np.array([lo[0], lo[1], -hi[2]]), np.array([hi[0], hi[1], -lo[2]])))

    def depth(self, position_xr, rotation_xr, intr, width, height):
        """View depth (metres along -Z) per pixel; inf where nothing is hit."""
        fx, fy, cx, cy = intr
        v, u = np.mgrid[0:height, 0:width].astype(np.float64)
        rays = np.stack([(u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy, -np.ones_like(u)], axis=-1)
        d = rays.reshape(-1, 3) @ quat_to_matrix(rotation_xr).T
        o = np.asarray(position_xr, float)
        best = np.full(len(d), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.floor_y - o[1]) / d[:, 1]
            best = np.where((d[:, 1] < 0) & (t > 0), t, best)
            for lo, hi in self.boxes_xr:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tf >= tn) & (tn > 0)
                best = np.where(hit & (tn < best), tn, best)
        return best.reshape(height, width)

    def frame_payload(self, eye_unity, target_unity, *, width=256, height=256, fov_tan=1.0,
                      ipd=0.064, near=0.1, far=math.inf, noise=0.0, rng=None, flip_rows=False,
                      fmt=FORMAT_LZ4_D16, client_ts=0):
        """Render both views of a headset at eye_unity looking at target_unity.

        noise: depth noise sigma = noise * z^2 metres (Gaussian, needs rng).
        """
        eye = unity_to_xr(eye_unity)
        q = look_at_xr(eye, unity_to_xr(target_unity))
        right = quat_to_matrix(q)[:, 0]
        fx = width / (2 * fov_tan)
        fy = height / (2 * fov_tan)
        intr = (fx, fy, width / 2, height / 2)
        depths, poses = [], []
        for side in (-0.5, 0.5):
            p = eye + right * side * ipd
            z = self.depth(p, q, intr, width, height)
            if noise:
                rng = rng or np.random.default_rng()
                finite = np.isfinite(z)
                z[finite] += rng.normal(0, 1, finite.sum()) * noise * z[finite] ** 2
            depths.append(z)
            poses.append((p, q))
        return metric_payload(depths, poses, [intr, intr], near, far, fmt=fmt, flip_rows=flip_rows,
                              client_ts=client_ts)
