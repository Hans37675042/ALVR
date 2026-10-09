import socket
import struct
import threading
import time

import rktap
import tap_replay

MSG_CAMERA = 2
MSG_DEPTH = 3
MSG_STREAM_CONTROL = 100


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _control(sock, stream_id, enabled):
    sock.sendall(struct.pack("<II", MSG_STREAM_CONTROL, 2) + bytes([stream_id, int(enabled)]))


class FakeViewer:
    """Listens like depth_listener does and records (arrival_time, type, payload)."""

    def __init__(self, on_connect=None, on_message=None):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.received = []
        self.on_connect = on_connect
        self.on_message = on_message
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.srv.settimeout(10)
        sock, _ = self.srv.accept()
        sock.settimeout(10)
        if self.on_connect:
            self.on_connect(sock)
        while True:
            hdr = _recv_exact(sock, 8)
            if hdr is None:
                break
            t, n = struct.unpack("<II", hdr)
            payload = _recv_exact(sock, n) if n else b""
            self.received.append((time.monotonic(), t, payload))
            if self.on_message:
                self.on_message(sock, self.received)
        sock.close()
        self.srv.close()


def _write_tap(path, records):
    with rktap.TapWriter(path, {"feeds": {"depth": True, "camera": True}}) as w:
        for rec in records:
            w.write(*rec)


def _run(path, port, **kw):
    kw.setdefault("connect_timeout", 5)
    return tap_replay.replay(path, port=port, **kw)


def test_only_enabled_feed_is_sent_and_markers_are_not(tmp_path):
    base = 10**18
    path = tmp_path / "r.rktap"
    _write_tap(path, [
        (base, MSG_DEPTH, b"d0"),
        (base + 10_000_000, MSG_CAMERA, b"c0"),
        (base + 20_000_000, rktap.MSG_MARKER, rktap.marker_payload("m", base)),
        (base + 30_000_000, MSG_DEPTH, b"d1"),
    ])
    viewer = FakeViewer(on_connect=lambda s: (_control(s, 0, True), _control(s, 1, False)))
    stats = _run(path, viewer.port)
    viewer.thread.join(5)
    assert [(t, p) for _, t, p in viewer.received] == [(MSG_DEPTH, b"d0"), (MSG_DEPTH, b"d1")]
    assert stats["sent"] == 2
    assert stats["markers"] == 1
    assert stats["skipped"] == 1


def test_nothing_sent_before_viewer_enables_a_feed(tmp_path):
    base = 10**18
    path = tmp_path / "w.rktap"
    _write_tap(path, [(base + i * 10_000_000, MSG_DEPTH, b"d%d" % i) for i in range(3)])

    def late_enable(sock):
        time.sleep(0.3)
        _control(sock, 0, True)

    viewer = FakeViewer(on_connect=late_enable)
    _run(path, viewer.port)
    viewer.thread.join(5)
    # replay waits for the first enable instead of dropping the head of the recording
    assert [p for _, _, p in viewer.received] == [b"d0", b"d1", b"d2"]


def test_timing_follows_recording_scaled_by_speed(tmp_path):
    base = 10**18
    path = tmp_path / "t.rktap"
    _write_tap(path, [(base + i * 300_000_000, MSG_DEPTH, b"d") for i in range(3)])  # 0, .3, .6 s
    viewer = FakeViewer(on_connect=lambda s: _control(s, 0, True))
    _run(path, viewer.port, speed=2.0)
    viewer.thread.join(5)
    times = [t for t, _, _ in viewer.received]
    assert len(times) == 3
    assert 0.12 < times[1] - times[0] < 0.25
    assert 0.25 < times[2] - times[0] < 0.45


def test_feed_toggled_mid_stream(tmp_path):
    base = 10**18
    path = tmp_path / "g.rktap"
    _write_tap(path, [(base + i * 200_000_000, MSG_DEPTH, b"d%d" % i) for i in range(5)])

    def disable_after_two(sock, received):
        if len(received) == 2:
            _control(sock, 0, False)

    viewer = FakeViewer(on_connect=lambda s: _control(s, 0, True), on_message=disable_after_two)
    stats = _run(path, viewer.port)
    viewer.thread.join(5)
    assert [p for _, _, p in viewer.received] == [b"d0", b"d1"]
    assert stats["skipped"] == 3


def test_loop_repeats(tmp_path):
    base = 10**18
    path = tmp_path / "l.rktap"
    _write_tap(path, [(base, MSG_DEPTH, b"a"), (base + 10_000_000, MSG_DEPTH, b"b")])
    viewer = FakeViewer(on_connect=lambda s: _control(s, 0, True))
    _run(path, viewer.port, loop=3)
    viewer.thread.join(5)
    assert [p for _, _, p in viewer.received] == [b"a", b"b"] * 3
