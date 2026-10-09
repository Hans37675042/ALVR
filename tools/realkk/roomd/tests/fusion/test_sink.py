import numpy as np
import pytest

from roomd_io_stubs import ensure_io_modules

ensure_io_modules()

from roomd import protocol  # noqa: E402
from roomd.sink import FusionSink, ScenePrior  # noqa: E402
from roomd.fusion import create_sink  # noqa: E402
from roomd.fusion.sink import TsdfFusionSink  # noqa: E402
from roomd.fusion.synthscene import box_mesh  # noqa: E402

from conftest import BOX_MAX, BOX_MIN, HEAD_START, box_scene, scan_poses  # noqa: E402

IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def feed(sink, scene, poses):
    for eye, target in poses:
        sink.integrate(protocol.decode_depth_v2(scene.frame_payload(eye, target)))


def _hm_flags(hm, x, z):
    ix = int(np.floor((x - hm.origin_x) / hm.cell))
    iz = int(np.floor((z - hm.origin_z) / hm.cell))
    return int(hm.flags[iz, ix]), float(hm.top_y[iz, ix])


@pytest.fixture(scope="module")
def fed():
    sink = create_sink()
    scene = box_scene()
    feed(sink, scene, [(HEAD_START, (0.0, 0.0, 1.2))] + scan_poses())
    first = sink.snapshot_outputs()
    second = sink.snapshot_outputs()
    return sink, first, second


def test_factory_returns_fusion_sink():
    sink = create_sink()
    assert isinstance(sink, FusionSink) and isinstance(sink, TsdfFusionSink)
    assert sink.name == "tsdf"


def test_snapshot_maps_to_io_outputs(fed):
    _, out, _ = fed
    assert abs(out.floor_y) < 0.005 and out.floor_rms < 0.01
    assert out.objects is None and out.seats is None and out.walls is None
    assert isinstance(out.nav_heightmap, protocol.NavHeightmap) and out.nav_revision == 1
    assert out.nav_heightmap.floor_y.dtype == np.float16
    flags, top = _hm_flags(out.nav_heightmap, 0.5, 1.5)
    assert flags & 0b011 == 0b011 and abs(top - 0.45) <= 0.01
    assert out.mesh_chunks and all(isinstance(c, protocol.MeshChunk) for c in out.mesh_chunks)
    assert out.map_revision >= 1


def test_second_snapshot_has_nothing_new(fed):
    _, first, second = fed
    assert second.mesh_chunks == []
    assert second.nav_revision == first.nav_revision
    assert second.nav_heightmap is first.nav_heightmap  # latest map, unchanged


def test_scene_prior_mesh_and_floor():
    sink = create_sink()
    verts, idx = box_mesh(BOX_MIN, BOX_MAX)
    sink.set_scene_prior(ScenePrior(snapshot_id=7, floor_y=0.01, objects=[], walls=[],
                                    mesh_vertices=verts, mesh_triangles=idx.reshape(-1, 3)))
    assert sink.fusion.config.floor_y == pytest.approx(0.01)
    out = sink.snapshot_outputs()
    flags, top = _hm_flags(out.nav_heightmap, 0.5, 1.5)
    assert flags & 0b010 and abs(top - 0.45) <= 0.02
    assert out.mesh_chunks


def test_scene_prior_without_mesh_only_sets_floor():
    sink = create_sink()
    sink.set_scene_prior(ScenePrior(snapshot_id=1, floor_y=-0.02, objects=[], walls=[]))
    assert sink.fusion.config.floor_y == pytest.approx(-0.02)
    assert sink.fusion.bounds() is None


def test_unchanged_playspace_keeps_map(fed):
    sink, _, _ = fed
    sink.on_playspace_changed(IDENTITY)
    assert sink.fusion.bounds() is not None


def test_recenter_clears_map_and_removes_sent_chunks():
    sink = create_sink()
    feed(sink, box_scene(), [(HEAD_START, (0.0, 0.0, 1.2))] + scan_poses(8))
    before = sink.snapshot_outputs()
    sent = {(c.ix, c.iy, c.iz): c.revision for c in before.mesh_chunks if len(c.vertices)}
    assert sent
    sink.on_playspace_changed((0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert sink.fusion.bounds() is None
    after = sink.snapshot_outputs()
    removed = {(c.ix, c.iy, c.iz): c for c in after.mesh_chunks}
    assert set(removed) == set(sent)
    for key, c in removed.items():
        assert len(c.vertices) == 0 and c.revision > sent[key]
    assert after.nav_revision == before.nav_revision + 1
    assert not np.any(after.nav_heightmap.flags)
    assert after.floor_y is None
    # mapping restarts from the next frame
    feed(sink, box_scene(), [(HEAD_START, (0.0, 0.0, 1.2))])
    assert sink.fusion.bounds() is not None
