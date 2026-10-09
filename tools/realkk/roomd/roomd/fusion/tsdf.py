"""Dense projective TSDF fusion of the relayed Quest environment depth (Warp, GPU).

Design (R14): a fixed dense grid instead of a hashed block map, so voxels in front of every
measurement are carved back to free space and a moved chair disappears; weights are capped so
the running average turns into an exponential one and old geometry fades in a bounded number
of frames. A per-chunk change odometer drives incremental meshing.

Thread safety: every public method takes an internal lock, so integrate() may run on the
depth thread while snapshot_outputs() / queries run elsewhere.
"""

import dataclasses
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import warp as wp
import warp.geometry  # noqa: F401  (registers wp.geometry)

from . import kernels as K
from .config import FusionConfig
from .decode import DepthFrame, decode_depth_frame, quat_to_matrix
from .outputs import FloorPlane, FusionOutputs, Heightmap, MeshChunk


@dataclass
class Obb:
    """Oriented box in Unity stage space; yaw (radians) turns it about +Y (Unity convention)."""
    center: Sequence[float]
    half_extents: Sequence[float]
    yaw: float = 0.0

    def rotation(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        # Unity: positive yaw turns +Z towards +X
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass
class _ChunkState:
    revision: int = 0
    last_sent: float = -math.inf
    sent_nonempty: bool = False


def _device(cfg):
    if cfg.device:
        return cfg.device
    return "cuda:0" if wp.is_cuda_available() else "cpu"


class TsdfFusion:
    """FusionSink implementation: integrate(frame) + snapshot_outputs()."""

    def __init__(self, config: Optional[FusionConfig] = None):
        wp.init()
        self.config = config or FusionConfig()
        self._cv = self.config.chunk_voxels()
        self._device = _device(self.config)
        self._lock = threading.RLock()
        self._origin = None  # np.ndarray(3) voxel (0,0,0) centre, chunk aligned
        self._dims = None
        self._frame = 0
        self._fake = 0
        self._times = deque(maxlen=self.config.stats_window)
        self._chunks: Dict[Tuple[int, int, int], _ChunkState] = {}
        self._map_revision = 0
        self._hm_dirty = False
        self._hm_last = -math.inf
        self._floor_cache = None
        self._head = None
        self._pending_removals: List[Tuple[int, int, int]] = []

    # ------------------------------------------------------------------ volume

    def _allocate(self, center_xz):
        cfg = self.config
        cs = cfg.chunk_size
        cx, cz = cfg.center_xz if cfg.center_xz is not None else center_xz
        half = cfg.extent_xz / 2
        lo = np.array([math.floor((cx - half) / cs + 1e-9), math.floor((cfg.floor_y - cfg.floor_margin) / cs + 1e-9),
                       math.floor((cz - half) / cs + 1e-9)])
        hi = np.array([math.ceil((cx + half) / cs - 1e-9), math.ceil((cfg.floor_y + cfg.height_above_floor) / cs - 1e-9),
                       math.ceil((cz + half) / cs - 1e-9)])
        self._chunk_lo = lo.astype(int)
        self._chunk_dims = (hi - lo).astype(int)
        self._dims = tuple(int(n) * self._cv for n in self._chunk_dims)
        self._origin = lo * cs
        d = self._device
        self._tsdf = wp.ones(self._dims, dtype=float, device=d)
        self._weight = wp.zeros(self._dims, dtype=float, device=d)
        self._last_seen = wp.full(self._dims, -1, dtype=wp.int32, device=d)
        self._odo = wp.zeros(tuple(int(n) for n in self._chunk_dims), dtype=float, device=d)

    def bounds(self):
        """(lower, upper) corners of the voxel-centre grid, or None before the first frame."""
        with self._lock:
            if self._origin is None:
                return None
            v = self.config.voxel_size
            return tuple(self._origin), tuple(self._origin + (np.array(self._dims) - 1) * v)

    def voxel_count(self):
        return 0 if self._dims is None else int(np.prod(self._dims))

    def reset(self):
        """Drop the volume (e.g. after a recenter). The next frame allocates a new one.

        Chunk revisions survive and every chunk that was sent with geometry is reported as
        removed (vcount = 0) by the next snapshot_outputs(), so receivers never see a
        revision go backwards.
        """
        with self._lock:
            for key, st in self._chunks.items():
                if st.sent_nonempty:
                    self._pending_removals.append(key)
                    st.sent_nonempty = False
            self._origin = self._dims = None
            self._tsdf = self._weight = self._last_seen = self._odo = None
            self._floor_cache = None
            self._hm_dirty = False
            self._head = None

    def save_state(self, path):
        """Write the volume and counters to a compressed .npz (offline analysis, semantics
        experiments). Publishing state (chunk revisions) is not saved."""
        with self._lock:
            if self._origin is None:
                raise ValueError("nothing to save: no frame integrated yet")
            np.savez_compressed(
                path, tsdf=self._tsdf.numpy(), weight=self._weight.numpy(), last_seen=self._last_seen.numpy(),
                odometer=self._odo.numpy(), origin=self._origin, chunk_lo=self._chunk_lo,
                chunk_dims=self._chunk_dims, counters=np.array([self._frame, self._fake, self._map_revision]),
                head=np.asarray(self._head if self._head is not None else [np.nan] * 3, dtype=np.float64),
                config=json.dumps(dataclasses.asdict(self.config)))

    @classmethod
    def load_state(cls, path, device=None):
        """Inverse of save_state(); queries and snapshot_outputs() work on the result."""
        with np.load(path) as z:
            cfg = json.loads(str(z["config"]))
            if cfg.get("center_xz") is not None:
                cfg["center_xz"] = tuple(cfg["center_xz"])
            if device is not None:
                cfg["device"] = device
            fusion = cls(FusionConfig(**cfg))
            d = fusion._device
            fusion._origin = z["origin"]
            fusion._chunk_lo = z["chunk_lo"]
            fusion._chunk_dims = z["chunk_dims"]
            fusion._dims = tuple(int(n) for n in z["tsdf"].shape)
            fusion._tsdf = wp.array(z["tsdf"], dtype=float, device=d)
            fusion._weight = wp.array(z["weight"], dtype=float, device=d)
            fusion._last_seen = wp.array(z["last_seen"], dtype=wp.int32, device=d)
            fusion._odo = wp.array(z["odometer"], dtype=float, device=d)
            fusion._frame, fusion._fake, fusion._map_revision = (int(c) for c in z["counters"])
            head = z["head"]
            fusion._head = None if np.isnan(head).any() else head
            fusion._hm_dirty = True
        return fusion

    # ------------------------------------------------------------- integration

    def _view_arrays(self, frame: DepthFrame):
        d = self._device
        intr = [wp.vec4(v.fx, v.fy, v.cx, v.cy) for v in frame.views]
        rot_cw = [quat_to_matrix(v.rotation_xr) for v in frame.views]
        depth = np.stack([v.depth for v in frame.views]).astype(np.float32)
        return (wp.array(depth, dtype=float, device=d),
                wp.array(intr, dtype=wp.vec4, device=d),
                wp.array([wp.mat33(*r.flatten()) for r in rot_cw], dtype=wp.mat33, device=d),
                wp.array([wp.mat33(*r.T.flatten()) for r in rot_cw], dtype=wp.mat33, device=d),
                wp.array([wp.vec3(*v.position_xr) for v in frame.views], dtype=wp.vec3, device=d))

    def _mask(self, frame: DepthFrame, depth, intr, rot_cw, pos):
        cfg = self.config
        head = frame.head_position_unity
        n, h, w = depth.shape
        out = wp.empty_like(depth)
        wp.launch(K.mask_depth, dim=(n, h, w), device=self._device,
                  inputs=[depth, out, intr, rot_cw, pos, wp.vec3(*head),
                          int(round(cfg.border_crop * w)), int(round(cfg.border_crop * h)),
                          cfg.min_depth, cfg.max_depth, math.cos(math.radians(cfg.max_incidence_deg)),
                          cfg.body_radius ** 2, float(head[1] + cfg.body_top_offset),
                          float(cfg.floor_y - cfg.below_floor_tolerance)])
        return out

    def _decode(self, frame):
        if isinstance(frame, DepthFrame):
            return frame
        return decode_depth_frame(bytes(frame), flip_rows=self.config.flip_rows)

    def masked_depth(self, frame) -> List[np.ndarray]:
        """Debug/test helper: per-view depth after the validity mask (0 = dropped)."""
        frame = self._decode(frame)
        depth, intr, rot_cw, _, pos = self._view_arrays(frame)
        return list(self._mask(frame, depth, intr, rot_cw, pos).numpy())

    def integrate(self, frame) -> bool:
        """Fuse one depth frame (raw MSG_DEPTH_FRAME_V2 payload or DepthFrame).

        Returns False when the frame is a fake readback frame (counted, ignored).
        Raises ValueError on a malformed payload.
        """
        t0 = time.perf_counter()
        frame = self._decode(frame)
        with self._lock:
            if frame is None:
                self._fake += 1
                return False
            head = frame.head_position_unity
            if self._origin is None:
                self._allocate((head[0], head[2]))
            self._head = head
            cfg = self.config
            depth, intr, rot_cw, rot_wc, pos = self._view_arrays(frame)
            masked = self._mask(frame, depth, intr, rot_cw, pos)
            nx, ny, nz = self._dims
            wp.launch(K.integrate_depth, dim=(nx, ny // self._cv, nz), device=self._device,
                      inputs=[self._tsdf, self._weight, self._last_seen, self._odo, masked, intr, rot_wc,
                              pos, len(frame.views), wp.vec3(*self._origin), cfg.voxel_size, self._cv,
                              cfg.trunc, cfg.max_weight, self._frame, cfg.odometer_eps, cfg.conflict_threshold,
                              cfg.conflict_decay])
            wp.synchronize_device(self._device)
            self._frame += 1
            self._hm_dirty = True
            self._floor_cache = None
            self._times.append((time.perf_counter() - t0) * 1e3)
            return True

    def integrate_prior_mesh(self, vertices, indices, weight: Optional[float] = None, center=None):
        """Write a scene global mesh (Unity stage space) into the TSDF with a low weight (S7).

        Triangles must face free space with Unity's convention: cross(b - a, c - a) points
        out of the solid. Only the truncation band around the mesh is written; observed
        depth outweighs the prior after a few frames.
        center: (x, y, z) used to place the volume if no depth frame arrived yet
        (default: the mesh's bounding-box centre).
        """
        cfg = self.config
        verts = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
        idx = np.ascontiguousarray(indices, dtype=np.int32).reshape(-1)
        if len(idx) == 0 or len(idx) % 3:
            raise ValueError("indices must be a non-empty multiple of 3")
        with self._lock:
            if self._origin is None:
                c = center if center is not None else (verts.min(0) + verts.max(0)) / 2
                self._allocate((c[0], c[2]))
            d = self._device
            mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device=d),
                           indices=wp.array(idx, dtype=wp.int32, device=d))
            nx, ny, nz = self._dims
            wp.launch(K.integrate_mesh_prior, dim=(nx, ny // self._cv, nz), device=d,
                      inputs=[self._tsdf, self._weight, self._odo, mesh.id, wp.vec3(*self._origin),
                              cfg.voxel_size, self._cv, cfg.trunc,
                              cfg.prior_weight if weight is None else float(weight), cfg.max_weight])
            wp.synchronize_device(d)
            self._hm_dirty = True
            self._floor_cache = None

    # ------------------------------------------------------------------ queries

    def debug_arrays(self):
        """(tsdf, weight) as numpy (nx, ny, nz); large copy, tests and tools only."""
        with self._lock:
            return self._tsdf.numpy(), self._weight.numpy()

    def _j_index(self, y):
        return int(math.floor((y - self._origin[1]) / self.config.voxel_size))

    def surface_points(self, min_y, max_y, region: Optional[Obb] = None):
        """Upward-facing surface points with min_y <= y < max_y: (points (N,3), normals (N,3)).

        Points sit on the interpolated zero crossing of each voxel column; normals are the
        normalised TSDF gradient (y >= config.upward_min_normal_y). region filters by Obb.
        """
        with self._lock:
            if self._origin is None:
                return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
            nx, ny, nz = self._dims
            j_lo = max(self._j_index(min_y) - 1, 0)
            j_hi = min(self._j_index(max_y) + 1, ny - 1)
            cap = nx * nz * 2
            d = self._device
            count = wp.zeros(1, dtype=wp.int32, device=d)
            pts = wp.empty(cap, dtype=wp.vec3, device=d)
            nrm = wp.empty(cap, dtype=wp.vec3, device=d)
            wp.launch(K.upward_surface_points, dim=(nx, nz), device=d,
                      inputs=[self._tsdf, self._weight, wp.vec3(*self._origin), self.config.voxel_size,
                              j_lo, j_hi, self.config.upward_min_normal_y, count, pts, nrm])
            n = min(int(count.numpy()[0]), cap)
            p = pts.numpy()[:n]
            q = nrm.numpy()[:n]
        # the kernel appends through an atomic counter: fix the order so results (and the
        # floor fit summed over them) repeat exactly
        order = np.lexsort((p[:, 1], p[:, 2], p[:, 0]))
        p = p[order]
        q = q[order]
        keep = (p[:, 1] >= min_y) & (p[:, 1] < max_y)
        if region is not None:
            local = (p - np.asarray(region.center)) @ region.rotation()
            keep &= np.all(np.abs(local) <= np.asarray(region.half_extents), axis=1)
        return p[keep], q[keep]

    def floor_plane(self) -> Optional[FloorPlane]:
        """Plane fitted to upward surfaces near config.floor_y (robust least squares)."""
        with self._lock:
            if self._floor_cache is not None:
                return self._floor_cache
            cfg = self.config
            p, _ = self.surface_points(cfg.floor_y - 0.15, cfg.floor_y + 0.15)
            if len(p) < 50:
                return None
            # Low furniture (rugs, platforms, bed frames) also lies in the search band and can
            # outnumber the visible floor: seed with the well-populated 1 cm height bin closest
            # to the STAGE floor instead of fitting everything.
            hist, edges = np.histogram(p[:, 1], bins=np.arange(cfg.floor_y - 0.15, cfg.floor_y + 0.155, 0.01))
            smooth = np.convolve(hist, np.ones(3), mode="same")
            centers = (edges[:-1] + edges[1:]) / 2
            dense = smooth >= 0.5 * smooth.max()
            seed_y = centers[dense][np.argmin(np.abs(centers[dense] - cfg.floor_y))]
            keep = np.abs(p[:, 1] - seed_y) < 0.03
            for _ in range(3):
                a = np.c_[p[keep, 0], p[keep, 2], np.ones(keep.sum())]
                coef, *_ = np.linalg.lstsq(a, p[keep, 1], rcond=None)
                res = p[:, 1] - (coef[0] * p[:, 0] + coef[1] * p[:, 2] + coef[2])
                sigma = max(np.std(res[keep]), 1e-4)
                keep = np.abs(res) < min(3 * sigma, 0.03)
            normal = np.array([-coef[0], 1.0, -coef[1]])
            normal /= np.linalg.norm(normal)
            centroid = p[keep].mean(axis=0)
            y = float(coef[0] * centroid[0] + coef[1] * centroid[2] + coef[2])
            self._floor_cache = FloorPlane(y=y, normal=tuple(normal), rms=float(np.sqrt(np.mean(res[keep] ** 2))),
                                           count=int(keep.sum()))
            return self._floor_cache

    def _floor_y(self):
        fp = self.floor_plane()
        return fp.y if fp is not None else self.config.floor_y

    def heightmap(self) -> Optional[Heightmap]:
        """2.5D map on a config.heightmap_cell grid covering the volume footprint."""
        with self._lock:
            if self._origin is None:
                return None
            cfg = self.config
            cell = cfg.heightmap_cell
            # footprint of the volume, cell-aligned to the chunk grid
            ox = float(self._origin[0])
            oz = float(self._origin[2])
            w = int(round(self._chunk_dims[0] * cfg.chunk_size / cell))
            h = int(round(self._chunk_dims[2] * cfg.chunk_size / cell))
            d = self._device
            floor = wp.empty((h, w), dtype=float, device=d)
            top = wp.empty((h, w), dtype=float, device=d)
            flags = wp.empty((h, w), dtype=wp.uint8, device=d)
            v = cfg.voxel_size
            wp.launch(K.heightmap_cells, dim=(h, w), device=d,
                      inputs=[self._tsdf, self._weight, wp.vec3(*self._origin), v, ox, oz, cell,
                              self._floor_y(), cfg.floor_tolerance, cfg.obstacle_min_height,
                              cfg.obstacle_max_height, cfg.obstacle_min_voxels, floor, top, flags])
            return Heightmap(cell=cell, origin_x=ox, origin_z=oz, width=w, height=h,
                             floor_y=floor.numpy(), top_y=top.numpy(), flags=flags.numpy())

    def region_stats(self, obb: Obb, max_age_frames: int = 10) -> dict:
        """Voxel counts inside obb: total, free, occupied, visible (seen in the last N frames)."""
        with self._lock:
            if self._origin is None:
                return {"total": 0, "free": 0, "occupied": 0, "visible": 0}
            v = self.config.voxel_size
            rot = obb.rotation()
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            pts = corners * np.asarray(obb.half_extents) @ rot.T + np.asarray(obb.center)
            lo = np.floor((pts.min(0) - self._origin) / v).astype(int)
            hi = np.ceil((pts.max(0) - self._origin) / v).astype(int)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, np.array(self._dims) - 1)
            if np.any(hi < lo):
                return {"total": 0, "free": 0, "occupied": 0, "visible": 0}
            d = self._device
            counts = wp.zeros(4, dtype=wp.int32, device=d)
            wp.launch(K.obb_counts, dim=tuple(int(n) for n in (hi - lo + 1)), device=d,
                      inputs=[self._tsdf, self._weight, self._last_seen, wp.vec3(*self._origin), v,
                              wp.vec3i(*lo.tolist()), wp.vec3(*obb.center), wp.mat33(*rot.T.flatten()),
                              wp.vec3(*obb.half_extents), self._frame - max_age_frames, counts])
            total, free, occ, vis = (int(c) for c in counts.numpy())
            return {"total": total, "free": free, "occupied": occ, "visible": vis}

    def _fraction(self, obb, key, max_age_frames=10):
        s = self.region_stats(obb, max_age_frames)
        return s[key] / s["total"] if s["total"] else 0.0

    def free_fraction(self, obb: Obb) -> float:
        """Share of the box observed as free space (tsdf > 0, weight > 0)."""
        return self._fraction(obb, "free")

    def occupied_fraction(self, obb: Obb) -> float:
        """Share of the box observed on or behind a surface (tsdf <= 0 within the band)."""
        return self._fraction(obb, "occupied")

    def visible_fraction(self, obb: Obb, max_age_frames: int = 10) -> float:
        """Share of the box updated by one of the last max_age_frames depth frames."""
        return self._fraction(obb, "visible", max_age_frames)

    # ------------------------------------------------------------------ meshing

    def _chunk_mesh(self, ci, cj, ck, floor_y):
        """Marching cubes on one chunk (+1 voxel overlap); returns (verts, indices) in Unity space.

        Triangles are kept only in cells flagged by kernels.mesh_cells, which removes the
        spurious faces at observed/unobserved borders and at the back of truncation bands.
        """
        cfg = self.config
        v = cfg.voxel_size
        n = self._cv
        i0, j0, k0 = ci * n, cj * n, ck * n
        sx = min(n + 1, self._dims[0] - i0)
        sy = min(n + 1, self._dims[1] - j0)
        sz = min(n + 1, self._dims[2] - k0)
        d = self._device
        empty = (np.zeros((0, 3), np.float32), np.zeros(0, np.uint32))
        field_t = wp.empty((sx, sy, sz), dtype=float, device=d)
        ok = wp.empty((sx - 1, sy - 1, sz - 1), dtype=wp.uint8, device=d)
        count = wp.zeros(1, dtype=wp.int32, device=d)
        wp.launch(K.mesh_cells, dim=(sx, sy, sz), device=d,
                  inputs=[self._tsdf, self._weight, field_t, ok, i0, j0, k0, count])
        if int(count.numpy()[0]) == 0:
            return empty
        lower = self._origin + np.array([i0, j0, k0]) * v
        upper = lower + (np.array([sx, sy, sz]) - 1) * v
        verts, idx = wp.geometry.IsoSurfaceMarchingCubes.extract(
            field_t, 0.0, lower=tuple(float(x) for x in lower), upper=tuple(float(x) for x in upper))
        verts = verts.numpy()
        tris = idx.numpy().reshape(-1, 3)
        if len(tris) == 0:
            return empty
        cell_ok = ok.numpy().astype(bool)
        cen = verts[tris].mean(axis=1)
        cell = np.floor((cen - lower) / v).astype(int)
        cell = np.clip(cell, 0, np.array(cell_ok.shape) - 1)
        keep = cell_ok[cell[:, 0], cell[:, 1], cell[:, 2]] & (cen[:, 1] >= floor_y + cfg.floor_exclude)
        tris = tris[keep]
        if len(tris) == 0:
            return empty
        used, inverse = np.unique(tris.reshape(-1), return_inverse=True)
        return verts[used].astype(np.float32), inverse.astype(np.uint32)

    def extract_mesh(self):
        """Whole-volume mesh (all chunks merged): (verts (N,3) float32, indices (M,) uint32)."""
        with self._lock:
            if self._origin is None:
                return np.zeros((0, 3), np.float32), np.zeros(0, np.uint32)
            floor_y = self._floor_y()
            vs, ids, base = [], [], 0
            for ci, cj, ck in np.ndindex(*self._chunk_dims):
                v, i = self._chunk_mesh(ci, cj, ck, floor_y)
                if len(i):
                    vs.append(v)
                    ids.append(i + base)
                    base += len(v)
            if not vs:
                return np.zeros((0, 3), np.float32), np.zeros(0, np.uint32)
            return np.concatenate(vs), np.concatenate(ids)

    # ------------------------------------------------------------------ outputs

    def stats(self) -> dict:
        with self._lock:
            times = list(self._times)
            return {
                "frames": self._frame,
                "fakeFrames": self._fake,
                "integrateMsP95": float(np.percentile(times, 95)) if times else 0.0,
                "integrateMsP50": float(np.percentile(times, 50)) if times else 0.0,
                "mapRevision": self._map_revision,
                "voxels": self.voxel_count(),
            }

    def snapshot_outputs(self, now: Optional[float] = None, force: bool = False) -> FusionOutputs:
        """Collect what is due for sending.

        mesh_chunks: chunks whose change odometer passed config.dirty_threshold and that were
        not sent within config.mesh_min_interval, most changed first, at most
        config.max_chunks_per_snapshot per call (~10 ms each on a 5090); vcount = 0 marks a
        chunk that became empty.
        heightmap: at most every config.heightmap_min_interval and only after new data.
        force: ignore rate limits and thresholds (every observed chunk, fresh heightmap).
        """
        now = time.monotonic() if now is None else now
        cfg = self.config
        with self._lock:
            out = FusionOutputs(mesh_chunks=[], heightmap=None, floor=None, stats={})
            for key in self._pending_removals:
                st = self._chunks[key]
                st.revision += 1
                st.last_sent = now
                out.mesh_chunks.append(MeshChunk(ix=key[0], iy=key[1], iz=key[2], revision=st.revision,
                                                 chunk_size=cfg.chunk_size,
                                                 vertices=np.zeros((0, 3), np.float32),
                                                 indices=np.zeros(0, np.uint32)))
            if self._pending_removals:
                self._pending_removals = []
                self._map_revision += 1
            if self._origin is None:
                out.stats = self.stats()
                return out
            odo = self._odo.numpy()
            floor_y = self._floor_y()
            changed = False
            due = []
            for ci, cj, ck in np.ndindex(*self._chunk_dims):
                key = tuple(int(x) for x in (self._chunk_lo + (ci, cj, ck)))
                st = self._chunks.setdefault(key, _ChunkState())
                o = odo[ci, cj, ck]
                if force:
                    if o > 0 or st.sent_nonempty:
                        due.append((o, (ci, cj, ck), key, st))
                elif o > cfg.dirty_threshold and now - st.last_sent >= cfg.mesh_min_interval:
                    due.append((o, (ci, cj, ck), key, st))
            due.sort(key=lambda d: -d[0])
            if not force:
                due = due[:cfg.max_chunks_per_snapshot]  # the rest stays dirty for the next call
            for _, (ci, cj, ck), key, st in due:
                verts, idx = self._chunk_mesh(ci, cj, ck, floor_y)
                odo[ci, cj, ck] = 0.0
                if len(idx) == 0 and not st.sent_nonempty:
                    continue
                st.revision += 1
                st.last_sent = now
                st.sent_nonempty = len(idx) > 0
                out.mesh_chunks.append(MeshChunk(ix=key[0], iy=key[1], iz=key[2], revision=st.revision,
                                                 chunk_size=cfg.chunk_size, vertices=verts, indices=idx))
                changed = True
            if force:
                odo[:] = 0.0
            self._odo.assign(odo)
            if force or (self._hm_dirty and now - self._hm_last >= cfg.heightmap_min_interval):
                out.heightmap = self.heightmap()
                self._hm_dirty = False
                self._hm_last = now
                changed = True
            if changed:
                self._map_revision += 1
            out.floor = self.floor_plane()
            out.stats = self.stats()
            return out
