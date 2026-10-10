import json
import math
import socket
import struct
import threading
import time

import numpy as np
import pytest

import rktap
import tap_replay
from roomd import model as M
from roomd import protocol as P
from roomd.io.plugin_server import PluginServer
from roomd.io.relay import RelayInbox, RelayViewer
from roomd.io.tapsource import TapSource
from roomd.service import RoomService
from roomd.sink import FusionOutputs, FusionSink
from roomd.synth.room import default_room, synth_uuid
from roomd.synth.snapshot import build_snapshot

POSES = ((0.0, 0.0, 0.0, 1.0, 0.0, 1.6, 0.0), (0.0, 0.0, 0.0, 1.0, 0.064, 1.6, 0.0))
FOV = ((-0.9, 0.8, 0.85, -0.95), (-0.8, 0.9, 0.85, -0.95))


def depth_msg(value=30000, w=8, h=8):
    return P.encode_depth_v2(np.full((h, w), value, np.uint16), 0.1, math.inf, POSES, FOV)


def fake_msg(w=8, h=8):
    return P.encode_depth_v2(np.full((h, w), 0x8080, np.uint16), 0.1, math.inf, POSES, FOV)


class RecordingSink(FusionSink):
    def __init__(self):
        self.frames = []
        self.priors = []

    def integrate(self, frame):
        self.frames.append(frame)

    def set_scene_prior(self, prior):
        self.priors.append(prior)

    def snapshot_outputs(self):
        hm = P.NavHeightmap(0.05, 0.0, 0.0, np.zeros((2, 2), np.float16), np.zeros((2, 2), np.float16),
                            np.ones((2, 2), np.uint8))
        chunks = [P.MeshChunk(0, 0, 0, len(self.frames), 1.0, np.zeros((3, 3), np.float32),
                              np.arange(3, dtype=np.uint32))] if self.frames else []
        return FusionOutputs(map_revision=len(self.frames), nav_heightmap=hm, nav_revision=1 if self.frames else 0,
                             mesh_chunks=chunks)


class PluginClient:
    """Fake KKS plugin: connects to 9945, sends PLUGIN_HELLO, collects frames."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.sendall(P.pack_frame(P.PLUGIN_HELLO, P.encode_json({"plugin": "0.0.1", "roomId": "test"})))
        self.frames = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                self.frames.append(P.read_frame(self.sock))
        except (ConnectionError, OSError):
            pass

    def of_type(self, t):
        return [p for (mt, p) in self.frames if mt == t]

    def wait_for(self, pred, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred(self):
                return True
            time.sleep(0.02)
        return False

    def close(self):
        self.sock.close()


def test_inbox_keeps_only_latest_depth_and_queues_the_rest():
    inbox = RelayInbox()
    for i in range(3):
        inbox.put(i, P.MSG_DEPTH_FRAME_V2, b"d%d" % i)
    inbox.put(5, P.MSG_ROOM_SNAPSHOT, b"s")
    inbox.put(6, P.MSG_CAMERA_FRAME, b"c")
    assert inbox.take_depth() == (2, b"d2")
    assert inbox.take_depth() is None
    assert inbox.skipped_depth == 2
    assert inbox.take_messages() == [(5, P.MSG_ROOM_SNAPSHOT, b"s")]  # camera frames are not kept


def test_relay_viewer_enables_depth_and_requests_snapshot():
    inbox = RelayInbox()
    viewer = RelayViewer(0, inbox, log=lambda *a: None)
    viewer.start()
    try:
        alvr = socket.create_connection(("127.0.0.1", viewer.port), timeout=5)
        got = [P.read_frame(alvr) for _ in range(3)]
        assert got[0] == (P.MSG_STREAM_CONTROL, bytes([P.STREAM_DEPTH, 1]))
        assert got[1] == (P.MSG_STREAM_CONTROL, bytes([P.STREAM_CAMERA, 0]))
        assert got[2] == (P.MSG_ROOM_REQUEST, b"\x00")
        alvr.sendall(P.pack_frame(P.MSG_DEPTH_FRAME_V2, b"x" * 10))
        assert inbox.wait(5)
        assert inbox.take_depth()[1] == b"x" * 10
        alvr.close()
    finally:
        viewer.close()


def test_relay_reader_drains_while_consumer_is_stalled():
    """ALVR writes with a 500 ms timeout: the reader must keep draining even if nobody consumes."""
    inbox = RelayInbox()
    viewer = RelayViewer(0, inbox, log=lambda *a: None)
    viewer.start()
    try:
        alvr = socket.create_connection(("127.0.0.1", viewer.port), timeout=5)
        alvr.settimeout(0.5)
        big = depth_msg(w=640, h=640)  # ~ uncompressed-size LZ4 of constant data is small; pad
        big = big + b"\0" * (800_000 - len(big))
        for _ in range(30):  # 24 MB with nobody calling take_depth()
            alvr.sendall(P.pack_frame(P.MSG_DEPTH_FRAME_V2, big))
        deadline = time.monotonic() + 5
        while inbox.received < 30 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert inbox.received == 30 and inbox.skipped_depth == 29
        alvr.close()
    finally:
        viewer.close()


def test_plugin_server_replays_state_to_late_clients_and_counts_them():
    server = PluginServer(0, log=lambda *a: None)
    server.start()
    try:
        server.publish(P.ROOM_MODEL, b'{"a":1}', remember=True)
        server.publish(P.MESH_CHUNK, b"c1", remember=("chunk", 1))
        server.publish(P.MESH_CHUNK, b"c1b", remember=("chunk", 1))
        server.publish(P.STATUS, b"{}")  # not remembered
        c = PluginClient(server.port)
        assert c.wait_for(lambda c: len(c.frames) >= 2)
        assert c.frames[:2] == [(P.ROOM_MODEL, b'{"a":1}'), (P.MESH_CHUNK, b"c1b")]
        assert c.wait_for(lambda c: server.client_count == 1)
        assert c.wait_for(lambda c: server.hellos and server.hellos[-1]["roomId"] == "test")
        c2 = PluginClient(server.port)
        assert c2.wait_for(lambda _: server.client_count == 2)
        server.publish(P.STATUS, b'{"x":1}')
        assert c2.wait_for(lambda c: (P.STATUS, b'{"x":1}') in c.frames)
        c.close()
        assert c2.wait_for(lambda _: server.client_count == 1)
        c2.close()
    finally:
        server.close()


def _write_tap(path, records):
    with rktap.TapWriter(path, {"feeds": {"depth": True, "camera": False}, "source": "test"}) as w:
        for rec in records:
            w.write(*rec)


def _scenario_records(t0=10**18):
    snap = P.encode_room_snapshot(build_snapshot(default_room(), snapshot_id=5))
    recs = [(t0, P.MSG_ROOM_SNAPSHOT, snap)]
    for i in range(8):
        recs.append((t0 + (i + 1) * 100_000_000, P.MSG_DEPTH_FRAME_V2, fake_msg() if i == 3 else depth_msg()))
    recs.append((t0 + 950_000_000, rktap.MSG_MARKER, rktap.marker_payload("end", t0)))
    return recs


def test_service_converts_snapshot_drops_fake_frames_and_forwards_fusion_outputs(tmp_path):
    tap = tmp_path / "s.rktap"
    _write_tap(tap, _scenario_records())
    sink = RecordingSink()
    server = PluginServer(0, log=lambda *a: None)
    server.start()
    client = PluginClient(server.port)
    svc = RoomService(sink, server, log=lambda *a: None)
    inbox = RelayInbox()
    src = TapSource(tap, inbox, speed=5.0)  # 20 ms apart: every frame reaches the loop
    src.start()
    stop = threading.Event()
    t = threading.Thread(target=svc.run, args=(inbox, stop), kwargs={"until": src.finished}, daemon=True)
    t.start()
    try:
        assert client.wait_for(lambda c: c.of_type(P.STATUS) and c.of_type(P.NAV_HEIGHTMAP))
        t.join(10)
        assert not t.is_alive()
        models = [json.loads(p) for p in client.of_type(P.ROOM_MODEL)]
        assert models and models[-1]["Version"] == 2 and models[-1]["FrameSource"] == "Stage"
        ids = {o["Id"] for o in models[-1]["Objects"]}
        assert "scene:" + synth_uuid("couch") in ids
        assert sink.priors and sink.priors[0].snapshot_id == 5 and sink.priors[0].mesh_vertices is not None
        assert svc.stats.fake_frames == 1
        assert len(sink.frames) == 7 and not any(f.is_fake() for f in sink.frames)
        status = json.loads(client.of_type(P.STATUS)[-1])
        assert set(status) == {"utc", "depthFps", "fakeFrames", "lastSnapshotId", "integrateMsP95",
                               "mapRevision", "clients"}
        assert status["lastSnapshotId"] == 5 and status["clients"] == 1
        assert client.of_type(P.MESH_CHUNK)
    finally:
        stop.set()
        client.close()
        server.close()


def test_model_revision_bumps_only_on_change():
    server = PluginServer(0, log=lambda *a: None)
    svc = RoomService(FusionSink(), server, log=lambda *a: None)
    snap = P.encode_room_snapshot(build_snapshot(default_room(), snapshot_id=1))
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, snap)
    r1 = svc.model.Revision
    created = svc.model.Objects[0].CreatedUtc
    time.sleep(0.01)
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, P.encode_room_snapshot(build_snapshot(default_room(), 2)))
    assert svc.model.Revision == r1 and svc.model.Objects[0].CreatedUtc == created
    moved = default_room()
    moved.move("table", dx=1.0)
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, P.encode_room_snapshot(build_snapshot(moved, 3)))
    assert svc.model.Revision == r1 + 1
    table = svc.model.find("scene:" + synth_uuid("table"))
    assert table.Revision == 1 and table.CreatedUtc == created
    server.close()


def test_end_to_end_tap_replay_to_plugin(tmp_path):
    """Fake ALVR (tap_replay) -> roomd relay viewer -> roomd -> fake plugin client."""
    from roomd.__main__ import Roomd

    tap = tmp_path / "e2e.rktap"
    _write_tap(tap, _scenario_records())
    app = Roomd(relay_port=0, plugin_port=0, sink=FusionSink(), log=lambda *a: None)
    app.start()
    stop = threading.Event()
    runner = threading.Thread(target=app.run, args=(stop,), daemon=True)
    runner.start()
    client = PluginClient(app.plugin.port)
    try:
        stats = tap_replay.replay(tap, port=app.relay.port, speed=4.0, connect_timeout=5, log=lambda *a: None)
        assert stats["sent"] == 9 and stats["markers"] == 1
        assert client.wait_for(lambda c: c.of_type(P.ROOM_MODEL) and c.of_type(P.STATUS)
                               and json.loads(c.of_type(P.STATUS)[-1])["lastSnapshotId"] == 5)
        model = json.loads(client.of_type(P.ROOM_MODEL)[-1])
        assert any(o["Label"] == "COUCH" for o in model["Objects"])
        assert client.wait_for(lambda c: json.loads(c.of_type(P.STATUS)[-1])["fakeFrames"] == 1)
    finally:
        stop.set()
        runner.join(5)
        client.close()
        app.close()


def test_record_option_writes_inbound_relay_messages(tmp_path):
    from roomd.__main__ import Roomd

    out = tmp_path / "rec.rktap"
    app = Roomd(relay_port=0, plugin_port=0, sink=FusionSink(), record=out, log=lambda *a: None)
    app.start()
    try:
        alvr = socket.create_connection(("127.0.0.1", app.relay.port), timeout=5)
        for _ in range(3):
            P.read_frame(alvr)
        alvr.sendall(P.pack_frame(P.MSG_DEPTH_FRAME_V2, depth_msg()))
        alvr.sendall(P.pack_frame(P.MSG_PLAYSPACE_CHANGED, P.encode_playspace_changed((0, 0, 0, 0, 0, 0, 1))))
        deadline = time.monotonic() + 5
        while app.inbox.received < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        alvr.close()
    finally:
        app.close()
    recs = list(rktap.TapReader(out))
    assert [r.msg_type for r in recs] == [P.MSG_DEPTH_FRAME_V2, P.MSG_PLAYSPACE_CHANGED]
    assert rktap.TapReader(out).metadata["listener_version"].startswith("roomd")


def test_plugin_server_reads_reclassify_requests_without_blocking_publish():
    logs = []
    server = PluginServer(0, log=logs.append)
    server.start()
    try:
        c = PluginClient(server.port)
        assert c.wait_for(lambda _: server.hellos)
        c.sock.sendall(P.pack_frame(P.ROOM_RECLASSIFY, P.encode_room_reclassify(["fused:1"]))
                       + P.pack_frame(P.ROOM_RECLASSIFY, b'{"ids":"bad"}')
                       + P.pack_frame(77, b"unknown type is skipped")
                       + P.pack_frame(P.ROOM_RECLASSIFY, b""))
        got = []
        assert c.wait_for(lambda _: got.extend(server.take_reclassify()) or len(got) >= 2)
        assert got == [["fused:1"], None]
        assert server.take_reclassify() == []
        assert any("reclassify" in str(m) for m in logs)
        server.publish(P.STATUS, b'{"y":2}')                 # the client is still served
        assert c.wait_for(lambda c: (P.STATUS, b'{"y":2}') in c.frames)
        assert server.client_count == 1
        c.close()
    finally:
        server.close()


class ReclassifySink(RecordingSink):
    def __init__(self):
        super().__init__()
        self.requests = []
        self.objects = [M.RoomObject(Id="fused:1", Kind=M.Kind.Chair, Source=M.SOURCE_FUSED, Locked=False)]

    def reclassify(self, ids):
        self.requests.append(ids)
        self.objects[0].Kind = M.Kind.Table
        return ["fused:1"]

    def snapshot_outputs(self):
        out = super().snapshot_outputs()
        out.floor_y, out.floor_rms, out.objects, out.seats = 0.0, 0.0, list(self.objects), []
        return out


class QueuePlugin:
    client_count = 0

    def __init__(self, requests):
        self.requests = list(requests)
        self.published = []

    def take_reclassify(self):
        out, self.requests = self.requests, []
        return out

    def publish(self, msg_type, payload, remember=None):
        self.published.append((msg_type, payload))

    def forget(self, key):
        pass


def test_service_forwards_reclassify_to_the_sink_and_publishes_the_model():
    logs = []
    sink = ReclassifySink()
    plugin = QueuePlugin([["fused:1", "fused:9"]])
    svc = RoomService(sink, plugin, log=logs.append)
    svc.tick()
    first = [p for t, p in plugin.published if t == P.ROOM_MODEL][-1]
    assert json.loads(first)["Objects"][0]["Kind"] == "Chair"
    n = len(plugin.published)
    assert svc.process(RelayInbox()) is True
    assert sink.requests == [["fused:1", "fused:9"]]
    models = [p for t, p in plugin.published[n:] if t == P.ROOM_MODEL]
    assert models and json.loads(models[-1])["Objects"][0]["Kind"] == "Table"
    assert any("reclassify" in str(m) for m in logs)
    assert svc.process(RelayInbox()) is False


def test_service_tolerates_a_sink_without_reclassify():
    plugin = QueuePlugin([None])
    svc = RoomService(FusionSink(), plugin, log=lambda *a: None)
    assert svc.process(RelayInbox()) is True
