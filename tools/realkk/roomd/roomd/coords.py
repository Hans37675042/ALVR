"""Pose math and the OpenXR (right-handed) -> Unity (left-handed) stage flip.

Quaternions are (x, y, z, w). Poses in this module are (px, py, pz, qx, qy, qz, qw),
the CONTRACT order; depth V2 headers carry q-first poses and are reordered on decode.
"""

import math

import numpy as np


def flip_position(p):
    """OpenXR stage position -> Unity stage position: (x, y, -z)."""
    p = np.asarray(p, dtype=np.float64)
    out = p.copy()
    out[..., 2] = -out[..., 2]
    return out


def flip_quat(q):
    """OpenXR rotation -> Unity rotation: (-qx, -qy, qz, qw)."""
    x, y, z, w = q
    return (-x, -y, z, w)


def flip_pose(pose):
    px, py, pz, qx, qy, qz, qw = pose
    return (px, py, -pz) + flip_quat((qx, qy, qz, qw))


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_to_matrix(q):
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m):
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = quat_normalize(q)
    return tuple(float(v) for v in (q if q[3] >= 0 else -q))


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    return (float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), math.cos(angle / 2))


def yaw_quat(yaw):
    """Rotation about +Y. In Unity (left-handed) yaw maps local +Z to (sin yaw, 0, cos yaw)."""
    return (0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2))


def yaw_of_forward(forward):
    """Unity yaw whose local +Z points along the horizontal part of `forward`."""
    return math.atan2(forward[0], forward[2])


def transform_points(pose, pts):
    """Apply pose (px,py,pz,qx,qy,qz,qw) to points (N,3)."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ quat_to_matrix(pose[3:]).T + np.asarray(pose[:3], dtype=np.float64)
