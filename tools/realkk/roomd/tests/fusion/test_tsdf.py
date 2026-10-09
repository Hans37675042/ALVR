import numpy as np
import pytest

from roomd.fusion import FusionConfig, Obb, TsdfFusion
from roomd.fusion.synthscene import SynthScene, box_mesh

from fusionkit import (BOX_CENTER, BOX_MAX, BOX_MIN, HEAD_START, box_scene, first_frame, scan,
                      scan_poses)

# Box volume minus the floor's own truncation band.
BOX_BODY = Obb(center=(0.5, 0.255, 1.5), half_extents=(0.25, 0.195, 0.25))


def test_volume_follows_first_head_and_floor():
    fusion = TsdfFusion(FusionConfig())
    first_frame(fusion, box_scene())
    lo, hi = fusion.bounds()
    assert lo[0] <= HEAD_START[0] - 3.9 and hi[0] >= HEAD_START[0] + 3.9
    assert lo[2] <= HEAD_START[2] - 3.9 and hi[2] >= HEAD_START[2] + 3.9
    assert lo[1] <= -0.25 and hi[1] >= 2.9
    # chunk aligned so chunk indices are global
    assert np.allclose(np.asarray(lo) / fusion.config.chunk_size,
                       np.round(np.asarray(lo) / fusion.config.chunk_size))


def test_box_top_height_within_1cm(scanned):
    fusion, _ = scanned
    pts, nrm = fusion.surface_points(0.3, 0.6)
    on = (np.abs(pts[:, 0] - 0.5) < 0.18) & (np.abs(pts[:, 2] - 1.5) < 0.18)
    assert on.sum() > 100
    assert np.abs(pts[on, 1] - 0.45).max() <= 0.01
    assert np.all(nrm[on, 1] > 0.9)
    hm = fusion.heightmap()
    floor_y, top_y, flags = hm.sample(0.5, 1.5)
    assert abs(top_y - 0.45) <= 0.01
    assert flags & 0b011 == 0b011  # known + obstacle


def test_floor_plane(scanned):
    fusion, _ = scanned
    fp = fusion.floor_plane()
    assert abs(fp.y) <= 0.005
    assert fp.normal[1] > 0.999
    assert fp.rms < 0.01 and fp.count > 1000


def test_region_fractions_on_present_box(scanned):
    fusion, _ = scanned
    assert fusion.occupied_fraction(BOX_BODY) > 0.1
    assert fusion.visible_fraction(BOX_BODY, max_age_frames=30) > 0.1
    air = Obb(center=(0.5, 1.0, 1.5), half_extents=(0.25, 0.2, 0.25))
    stats = fusion.region_stats(air)
    assert stats["free"] > 0.8 * stats["total"]  # the rest is outside every view: unknown
    assert stats["occupied"] == 0


def test_obb_yaw_selects_rotated_region(scanned):
    fusion, _ = scanned
    # a thin slab rotated 45 degrees still lies inside the box top region
    slab = Obb(center=(0.5, 0.4, 1.5), half_extents=(0.15, 0.04, 0.02), yaw=np.pi / 4)
    assert fusion.occupied_fraction(slab) > 0.5


def test_removed_box_is_carved_within_n_frames():
    fusion = TsdfFusion(FusionConfig())
    first_frame(fusion, box_scene())
    poses = scan_poses()
    scan(fusion, box_scene(), poses)
    assert fusion.occupied_fraction(BOX_BODY) > 0.1
    empty = box_scene(with_box=False)
    cleared_at = None
    for n, (eye, target) in enumerate(poses * 2, start=1):
        fusion.integrate(empty.frame_payload(eye, target))
        if fusion.occupied_fraction(BOX_BODY) < 0.01:
            cleared_at = n
            break
    print("box carved after %s frames" % cleared_at)
    assert cleared_at is not None and cleared_at <= 15
    assert fusion.free_fraction(BOX_BODY) > 0.9
    scan(fusion, empty, poses[:6])
    _, _, flags = fusion.heightmap().sample(0.5, 1.5)
    assert flags & 0b110 == 0b100  # walkable, not obstacle


def test_weight_is_capped():
    cfg = FusionConfig(max_weight=16.0)
    fusion = TsdfFusion(cfg)
    scene = box_scene()
    first_frame(fusion, scene)
    for _ in range(30):
        fusion.integrate(scene.frame_payload((0.5, 1.6, 0.2), (0.5, 0.0, 1.5)))
    _, weight = fusion.debug_arrays()
    assert weight.max() == pytest.approx(16.0)


def test_fake_frame_counted_and_ignored():
    fusion = TsdfFusion(FusionConfig())
    from roomd.fusion.synthscene import encode_payload
    raw = np.full((64, 32), 0x8080, np.uint16)
    pose = ((0, 1.6, 0), (0, 0, 0, 1))
    assert fusion.integrate(encode_payload(raw, [pose, pose], [(16.0, 16.0, 16.0, 16.0)] * 2,
                                           0.1, float("inf"))) is False
    assert fusion.stats()["fakeFrames"] == 1
    assert fusion.stats()["frames"] == 0
    assert fusion.bounds() is None  # a fake frame must not anchor the volume


def test_body_cylinder_is_not_integrated():
    # a "torso" right below the head and an identical control column 1.2 m away
    torso = ((-0.15, 0.0, -0.15), (0.15, 1.3, 0.15))
    control = ((1.05, 0.0, 0.65), (1.35, 1.3, 0.95))
    scene = SynthScene(floor_y=0.0, boxes=[torso, control])
    fusion = TsdfFusion(FusionConfig())
    for target in [(0.0, 0.0, 0.6), (0.6, 0.0, 0.8), (1.2, 0.5, 0.8), (0.2, 0.0, 0.4)]:
        assert fusion.integrate(scene.frame_payload(HEAD_START, target))
    torso_obb = Obb(center=(0.0, 0.9, 0.0), half_extents=(0.15, 0.4, 0.15))
    control_obb = Obb(center=(1.2, 0.9, 0.8), half_extents=(0.15, 0.4, 0.15))
    assert fusion.occupied_fraction(control_obb) > 0.05
    assert fusion.occupied_fraction(torso_obb) == 0.0


def test_mask_border_range_and_grazing():
    fusion = TsdfFusion(FusionConfig(border_crop=0.1, min_depth=0.5, max_depth=4.5))
    wall = SynthScene(floor_y=-100.0, boxes=[((-5, -5, 2.0), (5, 5, 2.5))])
    masked = fusion.masked_depth(wall.frame_payload((0, 1.6, 0), (0, 1.6, 2.0), width=100, height=100))
    for m in masked:
        assert np.all(m[:10, :] == 0) and np.all(m[-10:, :] == 0)
        assert np.all(m[:, :10] == 0) and np.all(m[:, -10:] == 0)
        assert np.all(m[12:-12, 12:-12] > 0)
    far_wall = SynthScene(floor_y=-100.0, boxes=[((-50, -50, 5.0), (50, 50, 6.0))])
    masked = fusion.masked_depth(far_wall.frame_payload((0, 1.6, 0), (0, 1.6, 5.0), width=100, height=100))
    assert all(np.all(m == 0) for m in masked)
    # floor seen almost edge-on from 0.3 m height: the far rows are grazing and dropped
    floor = SynthScene(floor_y=0.0)
    masked = fusion.masked_depth(floor.frame_payload((0, 0.3, 0), (0, 0.3, 3.0), width=100, height=100))
    for m in masked:
        assert np.count_nonzero(m[52:58, 20:80]) == 0
        # rows below ~66 hit the floor closer than min_depth; 68..74 are 70 deg off-normal
        assert np.count_nonzero(m[68:74, 20:80]) > 0


def test_prior_mesh_low_weight_and_overridden_by_depth():
    fusion = TsdfFusion(FusionConfig(prior_weight=2.0))
    verts, idx = box_mesh(BOX_MIN, BOX_MAX)
    fusion.integrate_prior_mesh(verts, idx, center=HEAD_START)
    assert fusion.occupied_fraction(BOX_BODY) > 0.1
    _, weight = fusion.debug_arrays()
    assert weight.max() == pytest.approx(2.0)
    hm = fusion.heightmap()
    _, top_y, flags = hm.sample(*BOX_CENTER[::2])
    assert flags & 0b010 and abs(top_y - 0.45) <= 0.02
    empty = box_scene(with_box=False)
    cleared_at = None
    for n, (eye, target) in enumerate(scan_poses(), start=1):
        fusion.integrate(empty.frame_payload(eye, target))
        if fusion.occupied_fraction(BOX_BODY) < 0.01:
            cleared_at = n
            break
    assert cleared_at is not None and cleared_at <= 4


def test_noisy_depth_still_meets_box_top_tolerance():
    fusion = TsdfFusion(FusionConfig())
    scene = box_scene()
    rng = np.random.default_rng(1)
    first_frame(fusion, scene)
    scan(fusion, scene, scan_poses(40), noise=0.004, rng=rng)
    _, top_y, _ = fusion.heightmap().sample(0.5, 1.5)
    assert abs(top_y - 0.45) <= 0.01


def test_floor_plane_ignores_low_platform_within_search_band():
    # a 10 cm high platform covering as much area as the visible floor (real tap: low
    # furniture inside the +-15 cm floor search band dragged the fit to rms 6.8 cm)
    scene = SynthScene(floor_y=0.0, boxes=[((-0.8, 0.0, 1.2), (1.8, 0.10, 2.6))])
    fusion = TsdfFusion(FusionConfig())
    first_frame(fusion, scene)
    scan(fusion, scene, scan_poses())
    fp = fusion.floor_plane()
    assert abs(fp.y) <= 0.005
    assert fp.rms < 0.01


def _corrupted_floor_payload(scene, eye, target, rng, frac=0.4, scale=1.3, size=128):
    """Real room tap: looking steeply down, ~40% of floor samples come back 5-30 cm beyond
    the floor. Reproduce by stretching a random subset of the rendered depth."""
    from roomd.fusion.decode import quat_to_matrix
    from roomd.fusion.synthscene import look_at_xr, metric_payload, unity_to_xr

    e = unity_to_xr(eye)
    q = look_at_xr(e, unity_to_xr(target))
    right = quat_to_matrix(q)[:, 0]
    intr = (size / 2, size / 2, size / 2, size / 2)
    depths, poses = [], []
    for side in (-0.032, 0.032):
        p = e + right * side
        z = scene.depth(p, q, intr, size, size)
        bad = rng.random(z.shape) < frac
        z[bad] *= scale
        depths.append(z)
        poses.append((p, q))
    return metric_payload(depths, poses, [intr, intr], 0.1, float("inf"))


def test_measurements_beyond_the_floor_do_not_carve_it():
    scene = SynthScene(floor_y=0.0)
    fusion = TsdfFusion(FusionConfig())
    rng = np.random.default_rng(3)
    for k in range(20):
        a = 2 * np.pi * k / 20
        target = (0.6 * np.sin(a), 0.0, 1.0 + 0.4 * np.cos(a))
        fusion.integrate(_corrupted_floor_payload(scene, (0.0, 1.15, 0.0), target, rng))
    below = fusion.region_stats(Obb(center=(0.0, -0.04, 1.0), half_extents=(0.4, 0.015, 0.3)))
    assert below["free"] == 0  # nothing may be carved under the stage floor
    fp = fusion.floor_plane()
    assert fp is not None and abs(fp.y) <= 0.01


def test_new_object_appears_within_n_frames():
    # real room tap: a chair moved into well-carved free space took ~6 s of viewing to show up,
    # while the old position cleared in < 1 s (R14 target: moved chair published p95 <= 2 s)
    fusion = TsdfFusion(FusionConfig())
    poses = scan_poses()
    empty = box_scene(with_box=False)
    first_frame(fusion, empty)
    scan(fusion, empty, poses)
    scan(fusion, empty, poses)  # free space at full weight
    assert fusion.occupied_fraction(BOX_BODY) == 0.0
    reference = TsdfFusion(FusionConfig())
    first_frame(reference, box_scene())
    scan(reference, box_scene(), poses)
    target = 0.8 * reference.occupied_fraction(BOX_BODY)
    appeared_at = None
    for n, (eye, target_pt) in enumerate(poses, start=1):
        fusion.integrate(box_scene().frame_payload(eye, target_pt))
        if fusion.occupied_fraction(BOX_BODY) >= target:
            appeared_at = n
            break
    print("box appeared after %s frames" % appeared_at)
    assert appeared_at is not None and appeared_at <= 8
