"""All tunable thresholds in one place (defaults from R15/R16; tune on device in S9)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SemanticsParams:
    # --- 2.5D raster / segmentation
    raster_source: str = "heightmap"   # "heightmap" (fusion 5 cm map) or "points" (upward points)
    raster_cell: float = 0.02          # m, raster cell when raster_source = "points"
    up_normal_min: float = 0.65        # normal.y for "horizontal surface" points (TSDF normals are noisy)
    blob_min_h: float = 0.05           # m above floor; lower = floor
    blob_max_h: float = 2.0            # m above floor; higher = structure, ignored
    structure_min_h: Optional[float] = None  # m; capped heightmap: cells this high = walls/unknown, ignored
    split_dh: float = 0.10             # m, neighbour step that cuts a blob in two
    overhang_min_h: float = 1.4        # m; heightmap tops this high may be an overhang (loft bed)
    overhang_free_min: float = 0.5     # free share under a tall piece that makes it an overhang
    split_patch_dh: float = 0.03       # m, two large flat patches this far apart split a blob
    split_patch_fill: float = 0.6      # ... if each fills this much of its min-area rectangle
    patch_tol: float = 0.02            # m, neighbour step inside one horizontal patch
    patch_max_range: float = 0.05      # m, 5-95 percentile height range a patch may span
    patch_min_area: float = 0.03       # m^2
    min_piece_area: float = 0.005      # m^2, smaller blob pieces are noise
    min_object_area: float = 0.04      # m^2
    thin_max: float = 0.30             # m, short side of a piece that may be a backrest/arm
    attach_min_rise: float = 0.05      # m, thin piece must be this much higher to attach
    attach_reach: int = 2              # cells; TSDF leaves a gap at seat/backrest edges

    # --- classification (R15 table)
    surface_band: float = 0.10         # m, height window of the main (seat/table) surface
    flat_min_ratio: float = 0.35       # share of base cells inside that window (cushion + armrests)
    seat_h: Tuple[float, float] = (0.35, 0.60)
    back_min_rise: float = 0.25        # m above seat
    back_min_cover: float = 0.45       # backrest length / seat width (office chair top edge is narrow)
    back_max_ratio: float = 0.8        # backrest cells / seat cells; more = side of other furniture
    seat_min_side: float = 0.30        # m, chair/couch footprint narrower than this is a ledge
    seat_max_depth: float = 1.1        # m, front-to-back depth no chair or couch exceeds
    couch_min_len: float = 1.2
    table_h: Tuple[float, float] = (0.65, 0.80)
    bed_min_area: float = 1.5          # m^2
    bed_h: Tuple[float, float] = (0.25, 0.70)
    bed_sittable: bool = False
    perch_h: Tuple[float, float] = (0.30, 0.60)
    perch_table_min_area: float = 0.25  # backless perch at least this big = low table
    conf_chair: float = 0.9
    conf_couch: float = 0.9
    conf_table: float = 0.85
    conf_bed: float = 0.8
    conf_perch: float = 0.4
    conf_other: float = 0.5
    hist_bin: float = 0.05             # m, template height histogram
    hist_max: float = 2.0

    # --- walls
    wall_max_ny: float = 0.3
    wall_min_len: float = 1.0
    wall_min_span: float = 1.2         # m of vertical extent
    wall_bin: float = 0.03
    wall_gap: float = 0.3
    wall_angle_tol: float = 10.0       # deg from a Manhattan axis

    # --- scene label fusion
    label_iou: float = 0.3
    label_center_dist: float = 0.3
    label_conf: float = 0.9
    label_only_conf: float = 0.6
    label_conflict_scale: float = 0.6
    default_seat_h: float = 0.45
    stale_known_min: float = 0.5       # label footprint must be this observed ...
    stale_occupied_max: float = 0.1    # ... and this empty to count as stale

    # --- tracking
    gate_dist: float = 3.0
    gate_surface_dh: float = 0.05
    cost_w_size: float = 1.0
    cost_w_surface: float = 2.0
    cost_w_hist: float = 0.5
    move_dist: float = 0.05
    move_yaw: float = 25.0             # deg; real chair yaw jitters about +-20
    move_confirm: int = 3              # consistent deviating observations to republish
    moving_after: int = 2              # deviating observations before State=Moving
    ema_alpha: float = 0.3
    new_confirm: int = 3               # consecutive observations before a new object
    new_match_dist: float = 0.3
    absorb_iou: float = 0.3            # unmatched candidate over a live track = same thing
    revive_dist: float = 1.0           # Removed tombstone revived within this distance
    missing_visible_min: float = 0.5
    missing_free_min: float = 0.6
    missing_confirm: int = 2           # consecutive free-space observations -> Missing
    remove_after_s: float = 5.0
    probe_below: float = 0.04          # probe slab around the surface for free-space evidence
    probe_above: float = 0.01
    probe_shrink: float = 0.7

    # --- seat states
    blocked_rise: float = 0.08
    blocked_fraction: float = 0.4
    seat_area_len_frac: float = 0.8
    seat_area_depth_frac: float = 0.5
    head_h: Tuple[float, float] = (1.1, 1.3)
    head_seat_radius: float = 0.35

    # --- publishing
    floor_change: float = 0.01
    wall_change: float = 0.05
