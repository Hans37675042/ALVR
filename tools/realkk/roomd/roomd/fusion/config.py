"""Tunables for the dense TSDF fusion. Distances in metres, Unity stage space."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FusionConfig:
    # volume
    voxel_size: float = 0.02
    chunk_size: float = 1.0           # MESH_CHUNK / change-odometer granularity; multiple of voxel_size
    extent_xz: float = 8.0            # x and z extent, centred on center_xz (or the first head pose)
    height_above_floor: float = 3.0
    floor_margin: float = 0.3         # volume reaches this far below floor_y
    floor_y: float = 0.0              # STAGE floor; Quest guardian floor is y = 0
    center_xz: Optional[Tuple[float, float]] = None

    # integration
    trunc: float = 0.08
    max_weight: float = 16.0          # running average 1/(n+1) until the cap, then a fixed-rate EMA
    conflict_threshold: float = 0.5  # |tsdf| above which an opposite-sign sample counts as a change
    conflict_decay: float = 0.25       # weight multiplier on such a sample (1 = plain running average)
    prior_weight: float = 2.0         # weight of a scene global-mesh prior (S7)
    min_depth: float = 0.5
    max_depth: float = 4.5
    border_crop: float = 0.10         # fraction of each depth image edge ignored
    max_incidence_deg: float = 75.0   # drop pixels whose surface is seen more edge-on than this
    body_radius: float = 0.4          # user's body: vertical cylinder below the head
    body_top_offset: float = 0.1      # cylinder top relative to the head height
    below_floor_tolerance: float = 0.05  # drop samples this far below floor_y (glossy/textureless floor)
    flip_rows: bool = False           # set if the depth rows turn out to be bottom-up
    device: Optional[str] = None      # Warp device; None = cuda:0 when available

    # change tracking and outputs
    odometer_eps: float = 0.01        # per-voxel |delta tsdf| below this is noise
    dirty_threshold: float = 2.0      # summed |delta tsdf| that makes a chunk re-mesh
    mesh_min_interval: float = 0.5    # seconds between MESH_CHUNKs of the same chunk (<= 2 Hz)
    heightmap_min_interval: float = 1.0
    max_chunks_per_snapshot: int = 16  # bounds how long snapshot_outputs holds the volume lock
    floor_exclude: float = 0.03       # mesh triangles this close above the floor are dropped
    heightmap_cell: float = 0.05
    floor_tolerance: float = 0.05     # an upward surface this close to floor_y counts as floor
    obstacle_min_height: float = 0.05
    obstacle_max_height: float = 1.7  # character height (1.5-1.6 m) + margin: loft underside blocks, beams do not
    obstacle_min_voxels: int = 2
    upward_min_normal_y: float = 0.85  # surface_points keeps normals at most ~32 deg from +Y
    stats_window: int = 200

    def chunk_voxels(self) -> int:
        n = round(self.chunk_size / self.voxel_size)
        if n < 2 or abs(n * self.voxel_size - self.chunk_size) > 1e-6:
            raise ValueError("chunk_size must be a multiple of voxel_size")
        return n
