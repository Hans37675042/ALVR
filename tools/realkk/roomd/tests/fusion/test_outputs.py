import struct

import numpy as np
import pytest

from roomd.fusion import FusionConfig, TsdfFusion
from roomd.fusion.outputs import (decode_mesh_chunk, decode_nav_heightmap, encode_mesh_chunk,
                                  encode_nav_heightmap)

from fusionkit import box_scene, first_frame, scan, scan_poses

BOX_CHUNK = (0, 0, 1)  # chunk holding the box: x 0..1, y 0..1, z 1..2


@pytest.fixture(scope="module")
def session():
    """Scan with the box, take outputs, remove the box, take outputs again."""
    fusion = TsdfFusion(FusionConfig())
    first_frame(fusion, box_scene())
    poses = scan_poses()
    scan(fusion, box_scene(), poses)
    first = fusion.snapshot_outputs(now=100.0)
    immediate = fusion.snapshot_outputs(now=100.1)
    scan(fusion, box_scene(with_box=False), poses)
    after = fusion.snapshot_outputs(now=102.0)
    return fusion, first, immediate, after


def _chunk(outputs, key):
    for c in outputs.mesh_chunks:
        if (c.ix, c.iy, c.iz) == key:
            return c
    return None


def test_mesh_chunks_cover_box_and_skip_floor(session):
    fusion, first, _, _ = session
    assert first.mesh_chunks
    c = _chunk(first, BOX_CHUNK)
    assert c is not None and len(c.vertices) > 0
    v = c.vertices
    top = (np.abs(v[:, 1] - 0.45) < 0.01) & (np.abs(v[:, 0] - 0.5) < 0.2) & (np.abs(v[:, 2] - 1.5) < 0.2)
    assert top.sum() > 50
    floor_y = fusion.floor_plane().y
    for ch in first.mesh_chunks:
        if len(ch.indices) == 0:
            continue
        tri = ch.vertices[ch.indices.reshape(-1, 3)]
        assert tri[:, :, 1].mean(axis=1).min() >= floor_y + 0.03
        lo = np.array([ch.ix, ch.iy, ch.iz]) * ch.chunk_size - fusion.config.voxel_size
        hi = lo + ch.chunk_size + 2 * fusion.config.voxel_size
        assert np.all(ch.vertices >= lo) and np.all(ch.vertices <= hi)
        assert ch.indices.max() < len(ch.vertices)


def test_mesh_winding_faces_free_space(session):
    _, first, _, _ = session
    c = _chunk(first, BOX_CHUNK)
    tri = c.vertices[c.indices.reshape(-1, 3)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    top = np.abs(tri[:, :, 1].mean(axis=1) - 0.45) < 0.005
    assert top.sum() > 10
    assert np.all(n[top, 1] > 0)  # Unity clockwise front face = cross(b-a, c-a) towards the viewer


def test_rate_limit_and_change_detection(session):
    _, first, immediate, _ = session
    assert first.heightmap is not None
    assert immediate.mesh_chunks == [] and immediate.heightmap is None


def test_removed_box_empties_its_chunk(session):
    _, first, _, after = session
    c = _chunk(after, BOX_CHUNK)
    assert c is not None and len(c.vertices) == 0 and len(c.indices) == 0
    assert c.revision > _chunk(first, BOX_CHUNK).revision
    assert after.stats["mapRevision"] > first.stats["mapRevision"]


def test_stats_keys(session):
    _, first, _, _ = session
    for key in ("integrateMsP95", "fakeFrames", "frames", "mapRevision"):
        assert key in first.stats
    assert first.stats["frames"] == 25


def test_mesh_chunk_payload_layout(session):
    _, first, _, _ = session
    c = _chunk(first, BOX_CHUNK)
    payload = encode_mesh_chunk(c)
    version, ix, iy, iz, rev, size, vcount, icount = struct.unpack_from("<IiiiIfII", payload, 0)
    assert (version, ix, iy, iz, rev) == (1, *BOX_CHUNK, c.revision)
    assert size == pytest.approx(1.0) and vcount == len(c.vertices) and icount == len(c.indices)
    assert len(payload) == 32 + 12 * vcount + 4 * icount
    back = decode_mesh_chunk(payload)
    np.testing.assert_array_equal(back.vertices, c.vertices.astype(np.float32))
    np.testing.assert_array_equal(back.indices, c.indices)


def test_heightmap_payload_and_flags(session):
    _, first, _, _ = session
    hm = first.heightmap
    payload = encode_nav_heightmap(hm)
    version, cell, ox, oz, w, h = struct.unpack_from("<IfffII", payload, 0)
    assert version == 1 and cell == pytest.approx(0.05) and (w, h) == (hm.width, hm.height)
    assert len(payload) == 24 + w * h * 5
    back = decode_nav_heightmap(payload)
    np.testing.assert_allclose(back.top_y, hm.top_y, atol=2e-3)
    _, top_y, flags = back.sample(0.5, 1.5)
    assert flags & 0b011 == 0b011 and abs(top_y - 0.45) <= 0.01
    _, _, flags = back.sample(-0.6, 1.5)  # floor next to the box
    assert flags & 0b111 == 0b101
    _, _, flags = back.sample(3.8, -3.8)  # never seen
    assert flags == 0


def test_force_snapshot_returns_everything(session):
    fusion, _, _, _ = session
    out = fusion.snapshot_outputs(now=200.0, force=True)
    assert out.heightmap is not None
    verts, _ = fusion.extract_mesh()
    assert len(verts) == 0  # box removed, floor excluded: nothing left


def test_extract_mesh_merges_chunks(scanned):
    fusion, _ = scanned
    verts, idx = fusion.extract_mesh()
    assert len(verts) > 0 and len(idx) % 3 == 0 and idx.max() < len(verts)
    assert np.abs(verts[:, 1] - 0.45).min() < 0.01
