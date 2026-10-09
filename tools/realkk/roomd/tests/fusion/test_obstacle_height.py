"""Heightmap obstacle band: a loft bed underside blocks walking, a ceiling beam does not."""
from roomd.fusion import FusionConfig, TsdfFusion
from roomd.fusion.synthscene import SynthScene

from fusionkit import first_frame

# Real room2: loft underside ~1.55 m above the floor, ceiling beam ~1.9 m; characters are
# 1.5-1.6 m tall, so the band stops at 1.7 m.
LOFT = ((0.2, 1.50, 1.0), (1.0, 1.60, 1.8))
BEAM = ((-1.2, 1.85, 1.0), (-0.4, 1.95, 1.8))


def _scanned(config):
    fusion = TsdfFusion(config)
    scene = SynthScene(floor_y=0.0, boxes=[LOFT, BEAM])
    first_frame(fusion, scene)
    for i in range(16):
        x = -1.4 + 2.6 * i / 15
        for target in ((0.6, 1.5, 1.4), (-0.8, 1.85, 1.4), (x, 0.0, 1.4)):
            fusion.integrate(scene.frame_payload((x, 1.1, 0.4), target))
    return fusion


def _obstacle(hm, x, z):
    _, _, flags = hm.sample(x, z)
    assert flags & 1, "cell not observed"
    return bool(flags & 2)


def test_default_obstacle_band_stops_at_1_7_m():
    assert FusionConfig().obstacle_max_height == 1.7


def test_loft_underside_blocks_but_ceiling_beam_does_not():
    hm = _scanned(FusionConfig()).heightmap()
    assert _obstacle(hm, 0.6, 1.4)
    assert not _obstacle(hm, -0.8, 1.4)
