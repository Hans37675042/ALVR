import math

import numpy as np

from roomd.semantics import Obb, SyntheticRoom
from roomd.semantics.geometry import (
    detect_walls, fit_floor, footprint_iou, label_components, min_area_rect, rasterize_max)


def _rect_points(cx, cz, deg, w, d, step=0.02):
    us = np.arange(-w / 2 + step / 2, w / 2, step)
    vs = np.arange(-d / 2 + step / 2, d / 2, step)
    u, v = np.meshgrid(us, vs)
    a = math.radians(deg)
    x = cx + u * math.cos(a) - v * math.sin(a)
    z = cz + u * math.sin(a) + v * math.cos(a)
    return np.stack([x.ravel(), z.ravel()], axis=1)


def test_min_area_rect_recovers_rotated_rectangle():
    pts = _rect_points(1.0, -0.5, 30.0, 2.0, 1.0)
    center, u, v, ext_u, ext_v = min_area_rect(pts)
    assert np.allclose(center, [1.0, -0.5], atol=0.02)
    long_ext, short_ext = max(ext_u, ext_v), min(ext_u, ext_v)
    assert abs(long_ext - 1.98) < 0.03 and abs(short_ext - 0.98) < 0.03
    ang = math.degrees(math.atan2(u[1], u[0])) % 90.0
    assert min(abs(ang - 30.0), abs(ang - 30.0 + 90.0), abs(ang - 30.0 - 90.0)) < 1.0


def test_footprint_iou():
    a = Obb(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert abs(footprint_iou(a, a) - 1.0) < 1e-9
    b = Obb(0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert abs(footprint_iou(a, b) - 1.0 / 3.0) < 1e-6
    c = Obb(0.0, 0.0, 90.0, 1.0, 1.0, 0.0, 1.0)
    assert abs(footprint_iou(a, c) - 1.0) < 1e-6
    far = Obb(5.0, 5.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert footprint_iou(a, far) == 0.0


def test_label_components_cuts_on_height_steps():
    top = np.full((4, 6), np.nan)
    top[:, 0:3] = 0.45
    top[:, 3:6] = 0.90  # adjacent but 45 cm higher
    mask = np.isfinite(top)
    labels, n = label_components(mask, top, 0.10)
    assert n == 2
    assert labels[0, 0] != labels[0, 5]
    labels, n = label_components(mask, top, 1.0)
    assert n == 1


def test_fit_floor_from_points():
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(-2, 2, 4000), 0.012 + rng.normal(0, 0.004, 4000),
                             rng.uniform(-2, 2, 4000)])
    table = np.column_stack([rng.uniform(0, 1, 800), np.full(800, 0.75), rng.uniform(0, 1, 800)])
    pts = np.vstack([floor, table])
    nrm = np.tile([0.0, 1.0, 0.0], (len(pts), 1))
    y, rms = fit_floor(pts, nrm)
    assert abs(y - 0.012) < 0.003
    assert rms < 0.008


def test_detect_walls_finds_room_box():
    room = SyntheticRoom(half_x=2.5, half_z=2.0, noise=0.003, seed=2)
    pts, nrm = room.surface_points(0.03, 3.0)
    walls = detect_walls(pts, nrm, 0.0)
    assert len(walls) == 4
    for (ax, az), (bx, bz) in walls:
        length = math.hypot(bx - ax, bz - az)
        assert length > 3.5
        on_x = abs(abs(ax) - 2.5) < 0.05 and abs(abs(bx) - 2.5) < 0.05
        on_z = abs(abs(az) - 2.0) < 0.05 and abs(abs(bz) - 2.0) < 0.05
        assert on_x or on_z


def test_raster_of_voxel_lattice_points_has_no_holes():
    # TSDF surface points sit exactly on the 2 cm voxel grid (multiples of the cell size)
    i, k = np.meshgrid(np.arange(20, 41), np.arange(13, 31))
    pts = np.column_stack([i.ravel() * 0.02, np.full(i.size, 0.45), k.ravel() * 0.02])
    r = rasterize_max(pts, 0.02)
    mask = np.isfinite(r.top)
    assert mask.sum() == i.size
    labels, n = label_components(mask, r.top, 0.1)
    assert n == 1


def test_raster_fills_isolated_dropouts():
    rng = np.random.default_rng(1)
    i, k = np.meshgrid(np.arange(0, 25), np.arange(0, 20))
    pts = np.column_stack([i.ravel() * 0.02 + 0.003, np.full(i.size, 0.45), k.ravel() * 0.02 + 0.004])
    keep = rng.uniform(size=len(pts)) > 0.1
    r = rasterize_max(pts[keep], 0.02)
    mask = np.isfinite(r.top)
    labels, n = label_components(mask, r.top, 0.1)
    assert n == 1
    assert mask.sum() >= 0.97 * i.size
