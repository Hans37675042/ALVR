"""9945 SCENE_MESH (type 5): the Quest GLOBAL_MESH forwarded to the plugin in parts."""

import struct

import numpy as np
import pytest

from roomd import protocol as P
from roomd import scene
from roomd.io.plugin_server import PluginServer
from roomd.service import RoomService
from roomd.sink import FusionSink
from roomd.synth.room import default_room
from roomd.synth.snapshot import build_snapshot


def _grid_mesh(nx, nz, cell=0.1):
    """Flat grid of 2 * nx * nz triangles in the xz plane."""
    xs, zs = np.meshgrid(np.arange(nx + 1) * cell, np.arange(nz + 1) * cell, indexing="ij")
    v = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()]).astype(np.float32)
    tris = []
    for i in range(nx):
        for k in range(nz):
            a = i * (nz + 1) + k
            b, c, d = a + 1, a + nz + 1, a + nz + 2
            tris += [(a, b, d), (a, d, c)]
    return v, np.array(tris, np.uint32)


def _triangle_set(v, t):
    return {tuple(sorted(tuple(np.round(v[i], 5)) for i in tri)) for tri in t}


def test_scene_mesh_type_and_header_layout():
    assert P.SCENE_MESH == 5
    part = P.SceneMeshPart(revision=3, snapshot_id=9, part=1, part_count=2,
                           vertices=np.arange(9, dtype=np.float32).reshape(3, 3),
                           indices=np.array([0, 2, 1], np.uint32))
    raw = P.encode_scene_mesh_part(part)
    assert struct.unpack_from("<7I", raw, 0) == (1, 3, 9, 1, 2, 3, 3)
    assert len(raw) == 28 + 3 * 12 + 3 * 4
    back = P.decode_scene_mesh_part(raw)
    assert (back.revision, back.snapshot_id, back.part, back.part_count) == (3, 9, 1, 2)
    assert np.array_equal(back.vertices, part.vertices) and np.array_equal(back.indices, [0, 2, 1])


def test_split_scene_mesh_caps_triangles_reindexes_and_keeps_every_triangle():
    v, t = _grid_mesh(30, 20)  # 1200 triangles
    parts = P.split_scene_mesh(v, t, revision=4, snapshot_id=7, max_triangles=500)
    assert len(parts) == 3
    assert [p.part for p in parts] == [0, 1, 2]
    assert all(p.part_count == 3 and p.revision == 4 and p.snapshot_id == 7 for p in parts)
    got = set()
    for p in parts:
        assert len(p.indices) % 3 == 0 and 0 < len(p.indices) // 3 <= 500
        assert p.indices.max() < len(p.vertices)
        # only the vertices the part uses are sent
        assert len(np.unique(p.indices)) == len(p.vertices)
        got |= _triangle_set(p.vertices, p.indices.reshape(-1, 3))
    assert got == _triangle_set(v, t)


def test_split_scene_mesh_parts_are_spatially_compact():
    v, t = _grid_mesh(40, 40, cell=0.1)  # 4 m x 4 m
    parts = P.split_scene_mesh(v, t, 1, 1, max_triangles=800)
    # sorted into 1 m cells before splitting: no part spans the whole room in x
    spans = [float(np.ptp(p.vertices[:, 0])) for p in parts]
    assert max(spans) < 3.0


def test_split_scene_mesh_empty_is_one_clear_part():
    parts = P.split_scene_mesh(None, None, revision=2, snapshot_id=5)
    assert len(parts) == 1
    p = parts[0]
    assert (p.part, p.part_count, len(p.vertices), len(p.indices)) == (0, 0, 0, 0)
    raw = P.encode_scene_mesh_part(p)
    assert struct.unpack_from("<7I", raw, 0) == (1, 2, 5, 0, 0, 0, 0) and len(raw) == 28


class _Publisher:
    """PluginServer stand-in that records publish/forget calls and the remembered state."""

    def __init__(self):
        self.sent = []
        self.remembered = {}
        self.client_count = 0

    def publish(self, msg_type, payload, remember=None):
        self.sent.append((msg_type, payload))
        if remember is not None and remember is not False:
            self.remembered[msg_type if remember is True else remember] = (msg_type, payload)

    def forget(self, key):
        self.remembered.pop(key, None)

    def scene_parts(self, sent=None):
        return [P.decode_scene_mesh_part(p) for (t, p) in (sent if sent is not None else self.sent)
                if t == P.SCENE_MESH]

    def remembered_scene_parts(self):
        return self.scene_parts(list(self.remembered.values()))


def _snapshot_msg(room, snapshot_id):
    return P.encode_room_snapshot(build_snapshot(room, snapshot_id=snapshot_id))


def test_service_publishes_global_mesh_in_unity_space():
    pub = _Publisher()
    svc = RoomService(FusionSink(), pub, log=lambda *a: None)
    room = default_room()
    snap = build_snapshot(room, snapshot_id=11)
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, P.encode_room_snapshot(snap))
    parts = pub.scene_parts()
    assert parts and all(p.snapshot_id == 11 and p.revision == 1 and p.part_count == len(parts) for p in parts)
    v_u, t_u = scene.scene_mesh_unity(snap)
    got = set()
    for p in parts:
        got |= _triangle_set(p.vertices, p.indices.reshape(-1, 3))
    assert got == _triangle_set(v_u, t_u)
    allv = np.concatenate([p.vertices for p in parts])
    assert allv[:, 2].min() == pytest.approx(room.z_min - 0.1, abs=1e-3)  # Unity z, not mirrored
    assert allv[:, 2].max() == pytest.approx(room.z_max + 0.1, abs=1e-3)
    # Unity winding (reversed against the OpenXR mesh) is kept: same orientation as scene_mesh_unity
    p0 = parts[0]
    tri = p0.vertices[p0.indices[:3]]
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    ref = {tuple(sorted(tuple(np.round(v_u[i], 5)) for i in tr)): tr for tr in t_u}
    rt = v_u[ref[tuple(sorted(tuple(np.round(x, 5)) for x in tri))]]
    assert np.dot(n, np.cross(rt[1] - rt[0], rt[2] - rt[0])) > 0
    assert len(pub.remembered_scene_parts()) == len(parts)


def test_service_skips_identical_mesh_and_replaces_parts_on_change():
    pub = _Publisher()
    svc = RoomService(FusionSink(), pub, log=lambda *a: None)
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, _snapshot_msg(default_room(), 1))
    n1 = len(pub.scene_parts())
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, _snapshot_msg(default_room(), 2))  # re-sent, same mesh
    assert len(pub.scene_parts()) == n1
    moved = default_room()
    moved.move("table", dx=1.0)
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, _snapshot_msg(moved, 3))
    new = pub.scene_parts()[n1:]
    assert new and all(p.revision == 2 and p.snapshot_id == 3 for p in new)
    kept = pub.remembered_scene_parts()
    assert kept and all(p.revision == 2 for p in kept) and len(kept) == new[0].part_count


def test_service_clears_scene_mesh_when_snapshot_has_none():
    pub = _Publisher()
    svc = RoomService(FusionSink(), pub, log=lambda *a: None)
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, _snapshot_msg(default_room(), 1))
    snap = build_snapshot(default_room(), snapshot_id=2)
    snap.meshes = []
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, P.encode_room_snapshot(snap))
    last = pub.scene_parts()[-1]
    assert (last.part_count, len(last.vertices), last.revision) == (0, 0, 2)
    kept = pub.remembered_scene_parts()
    assert len(kept) == 1 and kept[0].part_count == 0


def test_late_plugin_client_gets_current_scene_mesh_only():
    from test_io import PluginClient

    server = PluginServer(0, log=lambda *a: None)
    server.start()
    try:
        svc = RoomService(FusionSink(), server, log=lambda *a: None)
        svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, _snapshot_msg(default_room(), 1))
        moved = default_room()
        moved.move("table", dx=1.0)
        svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, _snapshot_msg(moved, 2))
        c = PluginClient(server.port)
        assert c.wait_for(lambda c: c.of_type(P.SCENE_MESH)
                          and len(c.of_type(P.SCENE_MESH)) == P.decode_scene_mesh_part(c.of_type(P.SCENE_MESH)[0]).part_count)
        parts = [P.decode_scene_mesh_part(p) for p in c.of_type(P.SCENE_MESH)]
        assert {p.revision for p in parts} == {2} and sorted(p.part for p in parts) == list(range(len(parts)))
        c.close()
    finally:
        server.close()
