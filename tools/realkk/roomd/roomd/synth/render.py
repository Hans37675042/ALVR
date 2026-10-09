"""Depth rendering of a SynthRoom with Open3D RaycastingScene (CPU).

Cameras follow MSG_DEPTH_FRAME_V2: OpenXR view pose (looks along local -Z, +Y up), FOV
angles (l, r, u, d) and pinhole intrinsics from alvr_packets::fov_to_pinhole_intrinsics
with the image y axis pointing down. Output is linear view depth z (metres), inf = no hit.
"""

import math
from dataclasses import dataclass

import numpy as np

from .. import coords
from .. import protocol as P
from .room import boxes_to_mesh


class SynthUnavailable(RuntimeError):
    pass


def require_open3d():
    try:
        import open3d as o3d  # noqa: F401
    except ImportError as e:
        raise SynthUnavailable(
            "roomd.synth needs Open3D for depth rendering (CPU wheel open3d==0.20.0): "
            "run `uv sync --extra synth` in tools/realkk/roomd (%s)" % e) from e
    return o3d


@dataclass
class DepthCamera:
    width: int = 320  # per view
    height: int = 320  # per view
    fov: tuple = ((-0.9, 0.8, 0.85, -0.95), (-0.8, 0.9, 0.85, -0.95))
    near: float = 0.1
    far: float = math.inf
    baseline: float = 0.1  # distance between the two depth views along head +X

    def intrinsics(self, i):
        return P.fov_to_intrinsics(self.fov[i], self.width, self.height)

    @classmethod
    def from_header(cls, header):
        """Mirror a real depth header (size, near/far, FOV, eye baseline)."""
        p0 = np.array(header["view_poses"][0][4:])
        p1 = np.array(header["view_poses"][1][4:])
        baseline = float(np.linalg.norm(p1 - p0)) or cls.baseline
        return cls(header["width"], header["height"] // 2, tuple(tuple(f) for f in header["fov_angles"]),
                   header["near_z"], header["far_z"], baseline)


def view_poses_xr(head_pose_unity, baseline):
    """Two depth view poses (px,py,pz,qx,qy,qz,qw) in OpenXR from a Unity head pose."""
    head_xr = coords.flip_pose(head_pose_unity)
    rot = coords.quat_to_matrix(head_xr[3:])
    out = []
    for side in (-0.5, 0.5):
        p = np.asarray(head_xr[:3]) + rot @ np.array([side * baseline, 0.0, 0.0])
        out.append(tuple(float(v) for v in p) + tuple(head_xr[3:]))
    return out


def pixel_rays_cam(width, height, intrinsics):
    """(h, w, 3) camera-space ray directions with z = -1, so the hit distance t equals z."""
    fx, fy, cx, cy = intrinsics
    u = (np.arange(width) + 0.5 - cx) / fx
    v = -(np.arange(height) + 0.5 - cy) / fy
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv, -np.ones_like(uu)], axis=-1)


class Renderer:
    def __init__(self, camera):
        self.o3d = require_open3d()
        self.camera = camera
        self._rays_cam = [pixel_rays_cam(camera.width, camera.height, camera.intrinsics(i)) for i in range(2)]
        self._scene = None
        self._scene_key = None

    def set_geometry(self, boxes_unity, key=None):
        """Rebuild the raycasting scene from 8-corner boxes (Unity) unless `key` is unchanged."""
        if key is not None and key == self._scene_key:
            return
        o3d = self.o3d
        verts_u, tris = boxes_to_mesh(boxes_unity)
        verts_xr = coords.flip_position(verts_u)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(verts_xr.astype(np.float32)),
                            o3d.core.Tensor(tris.astype(np.uint32)))
        self._scene = scene
        self._scene_key = key

    def render_view(self, i, view_pose_xr):
        rot = coords.quat_to_matrix(view_pose_xr[3:])
        dirs = self._rays_cam[i] @ rot.T
        origins = np.broadcast_to(np.asarray(view_pose_xr[:3]), dirs.shape)
        rays = np.concatenate([origins, dirs], axis=-1).astype(np.float32)
        t = self._scene.cast_rays(self.o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
        t[~np.isfinite(t)] = np.inf
        return t
