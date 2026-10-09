import json
import struct
import threading
import time

import pytest

import rktap
from synth import depth_payload

MSG_DEPTH = 3
MSG_CAMERA = 2


def _records():
    return [
        (1_700_000_000_000_000_000, MSG_DEPTH, depth_payload()),
        (1_700_000_000_100_000_000, MSG_CAMERA, b"\x00\x01camera"),
        (1_700_000_000_150_000_000, rktap.MSG_MARKER,
         rktap.marker_payload("chair moved", 1_700_000_000_150_000_000)),
        (1_700_000_000_200_000_000, MSG_DEPTH, b""),
    ]


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "a.rktap"
    meta = {"feeds": {"depth": True, "camera": False}}
    with rktap.TapWriter(path, meta) as w:
        for rec in _records():
            w.write(*rec)

    reader = rktap.TapReader(path)
    assert reader.version == 1
    assert reader.metadata["feeds"] == {"depth": True, "camera": False}
    got = [(r.recv_ns, r.msg_type, r.payload) for r in reader]
    assert got == _records()
    assert reader.truncated is False


def test_file_layout_matches_spec(tmp_path):
    path = tmp_path / "b.rktap"
    with rktap.TapWriter(path, {"k": "v"}) as w:
        w.write(42, 3, b"xyz")
    blob = path.read_bytes()
    assert blob[:5] == b"RKTAP"
    version, meta_len = struct.unpack_from("<II", blob, 5)
    assert version == 1
    meta = json.loads(blob[13:13 + meta_len].decode("utf-8"))
    assert meta["k"] == "v"
    rec = blob[13 + meta_len:]
    assert struct.unpack_from("<QII", rec, 0) == (42, 3, 3)
    assert rec[16:] == b"xyz"


def test_truncated_tail_is_tolerated(tmp_path):
    path = tmp_path / "c.rktap"
    with rktap.TapWriter(path, {}) as w:
        w.write(1, 3, b"abcd")
        w.write(2, 3, b"efgh")
    blob = path.read_bytes()
    path.write_bytes(blob[:-2])  # recorder killed mid-record
    reader = rktap.TapReader(path)
    got = list(reader)
    assert [r.payload for r in got] == [b"abcd"]
    assert reader.truncated is True


def test_bad_magic_rejected(tmp_path):
    path = tmp_path / "d.rktap"
    path.write_bytes(b"NOTAP" + b"\x00" * 16)
    with pytest.raises(ValueError):
        rktap.TapReader(path)


def test_marker_payload_round_trip():
    payload = rktap.marker_payload("椅子移 1 m", 1_700_000_000_123_456_789)
    m = rktap.parse_marker(payload)
    assert m["label"] == "椅子移 1 m"
    assert m["unix_ns"] == 1_700_000_000_123_456_789
    assert m["utc"].startswith("2023-11-14T22:13:20.123")


def test_threaded_recorder_keeps_order(tmp_path):
    path = tmp_path / "e.rktap"
    rec = rktap.TapRecorder(path, {"feeds": {"depth": True}})
    for i in range(500):
        rec.put(i, MSG_DEPTH, struct.pack("<I", i))
    rec.close()
    assert rec.written == 500
    got = list(rktap.TapReader(path))
    assert [r.recv_ns for r in got] == list(range(500))
    assert [struct.unpack("<I", r.payload)[0] for r in got] == list(range(500))


def test_recorder_put_does_not_block_on_slow_disk(tmp_path, monkeypatch):
    gate = threading.Event()
    real_write = rktap.TapWriter.write

    def slow_write(self, *a):
        gate.wait(5)
        return real_write(self, *a)

    monkeypatch.setattr(rktap.TapWriter, "write", slow_write)
    rec = rktap.TapRecorder(tmp_path / "f.rktap", {})
    t0 = time.monotonic()
    for i in range(100):
        rec.put(i, MSG_DEPTH, b"x" * 1000)
    assert time.monotonic() - t0 < 0.5
    gate.set()
    rec.close()
    assert rec.written == 100
