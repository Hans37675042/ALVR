import json
import struct
import subprocess
import sys
import time

import pytest

import depth_listener
import rktap
from synth import playspace_payload, room_snapshot_payload
from test_listener_tap import LISTENER, _connect, _free_port

SCENE = {
    "rooms": [{"uuid": "r1", "floor": "a1", "ceiling": None, "walls": ["a2", "a3"]}],
    "anchors": [
        {"uuid": "a1", "labels": "FLOOR", "pose": [0, 0, 0, 0, 0, 0, 1],
         "bbox2d": [-2, -2, 4, 4], "boundary2d": [[-2, -2], [2, -2], [2, 2]], "bbox3d": None},
        {"uuid": "a2", "labels": "WALL_FACE", "pose": [0, 1, -2, 0, 0, 0, 1],
         "bbox2d": [-2, -1, 4, 2], "boundary2d": None, "bbox3d": None},
        {"uuid": "a3", "labels": "WALL_FACE", "pose": [2, 1, 0, 0, 0.7, 0, 0.7],
         "bbox2d": [-2, -1, 4, 2], "boundary2d": None, "bbox3d": None},
        {"uuid": "a4", "labels": "TABLE", "pose": [0.5, 0.7, -1, 0, 0, 0, 1],
         "bbox2d": None, "boundary2d": None, "bbox3d": [-0.5, -0.3, -0.7, 1, 0.6, 0.7]},
        {"uuid": "a5", "labels": "GLOBAL_MESH", "pose": None,
         "bbox2d": None, "boundary2d": None, "bbox3d": None},
    ],
}
MESHES = [
    (bytes(range(16)), (0, 0, 0.5, 0, 0, 0, 1),
     [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [0, 1, 2, 2, 1, 3]),
]
RECENTER = (0.25, 0.0, -0.5, 0.0, 0.38268343, 0.0, 0.9238795)


def test_parse_room_snapshot():
    payload = room_snapshot_payload(9, RECENTER, SCENE, MESHES)
    snap = depth_listener.parse_room_snapshot(payload)
    assert snap["version"] == 1
    assert snap["snapshot_id"] == 9
    assert snap["recenter_pose"] == pytest.approx(RECENTER)
    assert snap["scene"] == SCENE
    assert len(snap["meshes"]) == 1
    mesh = snap["meshes"][0]
    assert mesh["anchor_uuid"] == "00010203-0405-0607-0809-0a0b0c0d0e0f"
    assert mesh["pose"] == pytest.approx(MESHES[0][1])
    assert (mesh["vcount"], mesh["icount"]) == (4, 6)
    assert mesh["vertices"][3] == pytest.approx((1, 1, 0))
    assert mesh["indices"][5] == 3


def test_parse_room_snapshot_skips_unknown_header_fields():
    # A future header with 8 extra bytes must still parse (viewers jump to header_size)
    payload = room_snapshot_payload(1, RECENTER, SCENE, MESHES, extra_header=b"\xff" * 8)
    snap = depth_listener.parse_room_snapshot(payload)
    assert snap["scene"] == SCENE
    assert snap["meshes"][0]["icount"] == 6


def test_room_snapshot_summary():
    snap = depth_listener.parse_room_snapshot(room_snapshot_payload(3, RECENTER, SCENE, MESHES))
    s = depth_listener.summarize_room_snapshot(snap)
    assert s["snapshot_id"] == 3
    assert s["rooms"] == 1
    assert s["anchors"] == 5
    assert s["located"] == 4
    assert s["labels"] == {"FLOOR": 1, "WALL_FACE": 2, "TABLE": 1, "GLOBAL_MESH": 1}
    assert s["meshes"] == 1
    assert s["triangles"] == 2


def test_empty_room_snapshot_summary():
    snap = depth_listener.parse_room_snapshot(
        room_snapshot_payload(1, RECENTER, {"rooms": [], "anchors": []}, []))
    s = depth_listener.summarize_room_snapshot(snap)
    assert (s["rooms"], s["anchors"], s["meshes"], s["triangles"]) == (0, 0, 0, 0)
    assert s["labels"] == {}


def test_parse_playspace_changed():
    p = depth_listener.parse_playspace_changed(playspace_payload(RECENTER))
    assert p["version"] == 1
    assert p["recenter_pose"] == pytest.approx(RECENTER)


def test_room_request_message():
    assert depth_listener.room_request_message(True) == struct.pack("<II", 101, 1) + b"\x01"
    assert depth_listener.room_request_message(False) == struct.pack("<II", 101, 1) + b"\x00"


def test_listener_prints_and_records_room_snapshot_and_sends_request(tmp_path):
    port = _free_port()
    tap = tmp_path / "room.rktap"
    proc = subprocess.Popen([sys.executable, str(LISTENER), "--port", str(port), "--seconds", "20",
                             "--tap", str(tap), "--request-room"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        sock = _connect(port)
        # two stream controls (depth on, camera off) then MSG_ROOM_REQUEST recapture=0
        expected = 2 * 10 + 9
        ctrl = b""
        while len(ctrl) < expected:
            ctrl += sock.recv(64)
        assert struct.unpack_from("<II", ctrl, 20) == (101, 1)
        assert ctrl[28] == 0
        payload = room_snapshot_payload(5, RECENTER, SCENE, MESHES)
        sock.sendall(struct.pack("<II", 4, len(payload)) + payload)
        play = playspace_payload(RECENTER)
        sock.sendall(struct.pack("<II", 5, len(play)) + play)
        time.sleep(0.3)
        sock.close()
        out, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    text = out.decode(errors="replace")
    assert proc.returncode == 0, text
    assert "room snapshot #5" in text
    assert "5 anchors" in text
    assert "WALL_FACE=2" in text
    assert "2 triangles" in text
    assert "playspace changed" in text
    recs = list(rktap.TapReader(tap))
    assert [(r.msg_type, r.payload) for r in recs] == [(4, payload), (5, play)]


def test_room_request_recapture_flag(tmp_path):
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(LISTENER), "--port", str(port), "--seconds", "20",
                             "--request-room", "recapture"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        sock = _connect(port)
        ctrl = b""
        while len(ctrl) < 29:
            ctrl += sock.recv(64)
        assert struct.unpack_from("<II", ctrl, 20) == (101, 1)
        assert ctrl[28] == 1
        sock.close()
        out, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, out.decode(errors="replace")


def test_scene_json_is_parsed_not_reencoded():
    # JSON text is taken verbatim from the payload
    payload = room_snapshot_payload(2, RECENTER, SCENE, [])
    snap = depth_listener.parse_room_snapshot(payload)
    assert snap["scene"] == json.loads(json.dumps(SCENE))
    assert snap["meshes"] == []
