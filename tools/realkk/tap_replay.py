#!/usr/bin/env python3
"""Fake ALVR streamer: replay a .rktap recording to a relay viewer.

Behaves like the streamer side of the realkk XR data relay: it connects *to* the viewer
(default 127.0.0.1:9944, retrying until the viewer listens), honours MSG_STREAM_CONTROL
from the viewer (all feeds start disabled; messages of a disabled feed are skipped, not
delayed), and drops the connection when a write blocks for more than 500 ms.

Messages go out verbatim, spaced by their original host receive times divided by
--speed. Playback starts when the viewer first enables a feed, so the head of the
recording is not lost to the enable handshake. Marker records are printed, never sent.

Standard library only.

Usage:
    python tap_replay.py FILE.rktap [--port 9944] [--speed 1.0] [--loop [N]]
"""

import argparse
import socket
import struct
import threading
import time
from pathlib import Path

import rktap
from depth_listener import MSG_CAMERA_FRAME, MSG_DEPTH_FRAME_V2, MSG_STREAM_CONTROL, STREAM_CAMERA, STREAM_DEPTH

WRITE_TIMEOUT_S = 0.5
FEED_OF_TYPE = {MSG_DEPTH_FRAME_V2: STREAM_DEPTH, MSG_CAMERA_FRAME: STREAM_CAMERA}


class ViewerControl:
    """Feed flags driven by the viewer's MSG_STREAM_CONTROL messages (read on its own thread)."""

    def __init__(self, sock, log):
        self.enabled = {STREAM_DEPTH: False, STREAM_CAMERA: False}
        self.any_enabled = threading.Event()
        self.closed = threading.Event()
        self._log = log
        self._sock = sock
        threading.Thread(target=self._run, name="viewer-control", daemon=True).start()

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("viewer closed the connection")
            buf += chunk
        return bytes(buf)

    def _run(self):
        try:
            while True:
                msg_type, length = struct.unpack("<II", self._recv_exact(8))
                payload = self._recv_exact(length) if length else b""
                if msg_type == MSG_STREAM_CONTROL and length >= 2 and payload[0] in self.enabled:
                    self.enabled[payload[0]] = payload[1] != 0
                    self._log("viewer stream control: depth=%s camera=%s"
                              % (self.enabled[STREAM_DEPTH], self.enabled[STREAM_CAMERA]))
                    if payload[1]:
                        self.any_enabled.set()
        except (ConnectionError, OSError):
            pass
        finally:
            self.closed.set()
            self.any_enabled.set()  # wake a waiting replay so it can notice the disconnect

    def allows(self, msg_type):
        feed = FEED_OF_TYPE.get(msg_type)
        return feed is None or self.enabled[feed]


def connect(host, port, timeout, retry_interval, log):
    deadline = None if timeout is None else time.monotonic() + timeout
    log("connecting to viewer at %s:%d ..." % (host, port))
    while True:
        try:
            return socket.create_connection((host, port), timeout=2)
        except OSError:
            if deadline is not None and time.monotonic() > deadline:
                raise
            time.sleep(retry_interval)


def replay(path, host="127.0.0.1", port=9944, speed=1.0, loop=1, connect_timeout=None,
           retry_interval=1.0, log=print):
    """Replay a tap to one viewer. loop: number of passes, 0 = forever. Returns counters."""
    reader = rktap.TapReader(path)
    stats = {"sent": 0, "skipped": 0, "markers": 0, "passes": 0, "disconnected": False}

    sock = connect(host, port, connect_timeout, retry_interval, log)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(WRITE_TIMEOUT_S)  # same 500 ms write timeout as the ALVR relay
    control_sock = sock.dup()
    control_sock.settimeout(None)
    control = ViewerControl(control_sock, log)
    log("connected; waiting for the viewer to enable a feed")
    try:
        control.any_enabled.wait()
        while not control.closed.is_set() and (loop == 0 or stats["passes"] < loop):
            first_ns = None
            start = time.monotonic()
            for rec in reader:
                if first_ns is None:
                    first_ns = rec.recv_ns
                delay = start + (rec.recv_ns - first_ns) / 1e9 / speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if rec.msg_type == rktap.MSG_MARKER:
                    m = rktap.parse_marker(rec.payload)
                    stats["markers"] += 1
                    log("marker t=%.2f s %s '%s'" % ((rec.recv_ns - first_ns) / 1e9, m.get("utc", ""),
                                                      m.get("label", "")))
                    continue
                if not control.allows(rec.msg_type):
                    stats["skipped"] += 1
                    continue
                try:
                    sock.sendall(struct.pack("<II", rec.msg_type, len(rec.payload)) + rec.payload)
                except OSError as e:
                    log("viewer write failed (%s), stopping" % e)
                    stats["disconnected"] = True
                    return stats
                stats["sent"] += 1
            stats["passes"] += 1
        if control.closed.is_set():
            stats["disconnected"] = True
        return stats
    finally:
        # FIN first and let the viewer close its side; closing outright while the control
        # thread is still in recv() makes Windows reset the connection instead.
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        control.closed.wait(2)
        sock.close()
        control_sock.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tap", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9944)
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    ap.add_argument("--loop", type=int, nargs="?", const=0, default=1,
                    help="repeat the recording N times; bare --loop repeats forever")
    args = ap.parse_args(argv)
    if args.speed <= 0:
        ap.error("--speed must be > 0")
    meta = rktap.TapReader(args.tap).metadata
    print("tap %s recorded %s (feeds %s)" % (args.tap, meta.get("recording_start_utc"), meta.get("feeds")))
    try:
        stats = replay(args.tap, args.host, args.port, args.speed, args.loop)
    except KeyboardInterrupt:
        print("stopped")
        return
    print("sent %(sent)d, skipped (feed off) %(skipped)d, markers %(markers)d, passes %(passes)d" % stats)


if __name__ == "__main__":
    main()
