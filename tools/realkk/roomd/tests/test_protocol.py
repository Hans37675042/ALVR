import json
import math
import struct

import numpy as np
import pytest

from roomd import protocol as P


def test_frame_header_round_trip():
    buf = P.pack_frame(P.ROOM_MODEL, b"{}")
    assert buf[:8] == struct.pack("<II", 1, 2)
    t, payload, rest = P.unpack_frame(buf + b"xx")
    assert (t, payload, rest) == (1, b"{}", b"xx")
    assert P.unpack_frame(buf[:5]) is None


def test_stream_control_and_room_request():
    assert P.encode_stream_control(P.STREAM_DEPTH, True) == bytes([0, 1])
    assert P.decode_stream_control(bytes([1, 0])) == (P.STREAM_CAMERA, False)
    assert P.encode_room_request(True) == b"\x01"
    assert P.decode_room_request(b"\x00") is False


def test_depth_round_trip_lz4_and_raw():
    d16 = (np.arange(4 * 6, dtype=np.uint16) * 991).reshape(6, 4)
    poses = ((0.0, 0.0, 0.0, 1.0, 0.1, 1.6, 0.2), (0.0, 0.0, 0.0, 1.0, 0.16, 1.6, 0.2))
    fov = ((-0.9, 0.8, 0.85, -0.95), (-0.8, 0.9, 0.85, -0.95))
    for fmt in (P.FORMAT_RAW_D16, P.FORMAT_LZ4_D16):
        payload = P.encode_depth_v2(d16, near=0.1, far=math.inf, fmt=fmt, view_poses=poses,
                                    fov_angles=fov, client_timestamp_ns=7, server_timestamp_unix_ns=9)
        frame = P.decode_depth_v2(payload)
        assert frame.header["width"] == 4 and frame.header["height"] == 6
        assert frame.header["client_timestamp_ns"] == 7
        assert np.array_equal(frame.d16(), d16)
        assert frame.views[1].pose_xr == pytest.approx((0.16, 1.6, 0.2, 0.0, 0.0, 0.0, 1.0))  # p-first
        assert frame.views[0].fov == pytest.approx(fov[0])
        # intrinsics follow fov_to_pinhole_intrinsics for a 4x3 view
        fx = 4 / (math.tan(0.8) - math.tan(-0.9))
        assert frame.views[0].intrinsics[0] == pytest.approx(fx, rel=1e-5)
        assert frame.view_d16(1).shape == (3, 4)


def test_depth_decode_matches_legacy_listener():
    import synth  # tools/realkk/tests/synth.py

    frame = P.decode_depth_v2(synth.depth_payload(width=4, height=4))
    assert frame.header["intrinsics"][1] == pytest.approx((102.0, 103.0, 52.0, 53.0))
    assert frame.views[0].pose_xr == pytest.approx((0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert not frame.is_fake()
    assert P.decode_depth_v2(synth.fake_depth_payload()).is_fake()


def test_d16_linear_depth_infinite_far():
    z = np.array([[0.5, 1.0, 4.0]])
    d16 = P.encode_d16_from_z(z, near=0.1, far=math.inf)
    back = P.d16_to_z(d16, near=0.1, far=math.inf)
    assert back == pytest.approx(z, rel=2e-3)
    finite = P.d16_to_z(P.encode_d16_from_z(z, near=0.1, far=10.0), near=0.1, far=10.0)
    assert finite == pytest.approx(z, rel=5e-3)


def test_room_snapshot_round_trip():
    meshes = [P.SnapshotMesh(anchor_uuid=bytes(range(16)),
                             pose=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                             vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                             indices=np.array([0, 1, 2], np.uint32))]
    doc = {"rooms": [], "anchors": [{"uuid": "a", "labels": "TABLE", "pose": None,
                                      "bbox2d": None, "boundary2d": None, "bbox3d": None}]}
    snap = P.RoomSnapshot(snapshot_id=42, recenter_pose=(0, 0, 0, 0, 0, 0, 1), scene=doc, meshes=meshes)
    payload = P.encode_room_snapshot(snap)
    assert struct.unpack_from("<III", payload, 0) == (40, 1, 42)
    back = P.decode_room_snapshot(payload)
    assert back.snapshot_id == 42 and back.scene == doc
    assert back.meshes[0].anchor_uuid == bytes(range(16))
    assert np.array_equal(back.meshes[0].indices, [0, 1, 2])
    assert back.meshes[0].vertices.shape == (3, 3)
    assert P.uuid_str(bytes(range(16))) == "00010203-0405-0607-0809-0a0b0c0d0e0f"


def test_room_snapshot_skips_extended_header():
    snap = P.RoomSnapshot(snapshot_id=1, recenter_pose=(0, 0, 0, 0, 0, 0, 1), scene={"anchors": []})
    payload = bytearray(P.encode_room_snapshot(snap))
    struct.pack_into("<I", payload, 0, 44)
    payload[40:40] = b"\xAA\xBB\xCC\xDD"  # a future header field
    assert P.decode_room_snapshot(bytes(payload)).scene == {"anchors": []}


def test_playspace_changed_round_trip():
    pose = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    assert P.decode_playspace_changed(P.encode_playspace_changed(pose)) == pytest.approx(pose)


def test_plugin_messages_round_trip():
    hm = P.NavHeightmap(cell=0.05, origin_x=-1.0, origin_z=-2.0,
                        floor_y=np.zeros((2, 3), np.float16), top_y=np.full((2, 3), 0.7, np.float16),
                        flags=np.array([[1, 3, 5], [0, 1, 1]], np.uint8))
    raw = P.encode_nav_heightmap(hm)
    assert struct.unpack_from("<I", raw, 0)[0] == 1
    assert len(raw) == 4 + 4 * 3 + 8 + 6 * 2 * 2 + 6
    back = P.decode_nav_heightmap(raw)
    assert (back.width, back.height) == (3, 2)
    assert np.array_equal(back.flags, hm.flags) and back.top_y[1, 2] == np.float16(0.7)

    chunk = P.MeshChunk(ix=1, iy=0, iz=-2, revision=5, chunk_size=1.0,
                        vertices=np.ones((3, 3), np.float32), indices=np.array([0, 2, 1], np.uint32))
    c2 = P.decode_mesh_chunk(P.encode_mesh_chunk(chunk))
    assert (c2.ix, c2.iy, c2.iz, c2.revision) == (1, 0, -2, 5)
    assert np.array_equal(c2.indices, [0, 2, 1])
    removed = P.decode_mesh_chunk(P.encode_mesh_chunk(P.MeshChunk.removed(1, 0, -2, 6, 1.0)))
    assert removed.is_removed

    status = {"utc": "x", "depthFps": 9.5, "fakeFrames": 1, "lastSnapshotId": 3,
              "integrateMsP95": 4.0, "mapRevision": 2, "clients": 1}
    assert P.decode_json(P.encode_json(status)) == status
    assert P.decode_json(P.encode_json({"plugin": "0.3.0", "roomId": "living"}))["roomId"] == "living"


def test_message_names_cover_contract():
    assert (P.MSG_DEPTH_FRAME_V2, P.MSG_ROOM_SNAPSHOT, P.MSG_PLAYSPACE_CHANGED,
            P.MSG_STREAM_CONTROL, P.MSG_ROOM_REQUEST) == (3, 4, 5, 100, 101)
    assert (P.ROOM_MODEL, P.NAV_HEIGHTMAP, P.MESH_CHUNK, P.STATUS, P.PLUGIN_HELLO) == (1, 2, 3, 4, 101)
    assert P.MSG_MARKER == 0xFFFF0001
