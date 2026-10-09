import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import rktap
from synth import depth_payload

LISTENER = Path(__file__).resolve().parent.parent / "depth_listener.py"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _connect(port, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=2)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def test_listener_records_every_message_verbatim(tmp_path):
    port = _free_port()
    tap = tmp_path / "x.rktap"
    proc = subprocess.Popen([sys.executable, str(LISTENER), "--port", str(port), "--seconds", "20",
                             "--tap", str(tap)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        sock = _connect(port)
        # listener enables depth and disables camera right after accept
        ctrl = b""
        while len(ctrl) < 20:
            ctrl += sock.recv(64)
        assert struct.unpack_from("<II", ctrl, 0) == (100, 2)
        payloads = [(3, depth_payload(client_ts=i)) for i in range(5)] + [(2, b"camera-frame")]
        t_before = time.time_ns()
        for msg_type, p in payloads:
            sock.sendall(struct.pack("<II", msg_type, len(p)) + p)
        time.sleep(0.3)
        sock.close()
        out, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, out.decode(errors="replace")
    reader = rktap.TapReader(tap)
    assert reader.metadata["feeds"] == {"depth": True, "camera": False}
    assert "recording_start_utc" in reader.metadata
    assert "listener_version" in reader.metadata
    recs = list(reader)
    assert [(r.msg_type, r.payload) for r in recs] == payloads
    assert all(r.recv_ns >= t_before for r in recs)
    assert reader.truncated is False
