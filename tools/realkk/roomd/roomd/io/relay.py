"""9944 relay viewer: roomd is the only viewer the ALVR server connects to.

The ALVR relay writes synchronously with a 500 ms timeout and drops the viewer when a write
blocks, so a dedicated reader thread drains the socket and never does more than hand the
message over: depth keeps only the newest frame (older ones are counted as skipped), camera
frames are discarded, everything else (room snapshots, playspace changes) is queued.
"""

import socket
import struct
import threading
import time
from collections import deque
from datetime import datetime, timezone

from .. import __version__
from .. import protocol as P
from .._legacy import rktap


class RelayInbox:
    """Hand-off between a producer thread (socket reader or tap source) and the main loop."""

    def __init__(self):
        self._cond = threading.Condition()
        self._depth = None
        self._messages = deque()
        self.received = 0
        self.skipped_depth = 0
        self.dropped_camera = 0

    def put(self, recv_ns, msg_type, payload):
        with self._cond:
            self.received += 1
            if msg_type == P.MSG_DEPTH_FRAME_V2:
                if self._depth is not None:
                    self.skipped_depth += 1
                self._depth = (recv_ns, payload)
            elif msg_type == P.MSG_CAMERA_FRAME:
                self.dropped_camera += 1
            else:
                self._messages.append((recv_ns, msg_type, payload))
            self._cond.notify_all()

    def wait(self, timeout):
        """True when a depth frame or a message is waiting."""
        with self._cond:
            return self._cond.wait_for(lambda: self._depth is not None or self._messages, timeout)

    def take_depth(self):
        with self._cond:
            d, self._depth = self._depth, None
            return d

    def take_messages(self):
        with self._cond:
            out = list(self._messages)
            self._messages.clear()
            return out


class RelayViewer:
    """TCP server on the relay port; ALVR is the client. One streamer connection at a time
    (a new connection replaces the old one). On connect: depth feed on, camera off, and a
    MSG_ROOM_REQUEST(recapture=0) so the server resends its cached scene snapshot."""

    def __init__(self, port, inbox, host="127.0.0.1", record=None, log=print):
        self.host = host
        self.inbox = inbox
        self.log = log
        self._requested_port = port
        self._srv = None
        self._conn = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self.connections = 0
        self.recorder = None
        if record is not None:
            self.recorder = rktap.TapRecorder(record, {
                "recording_start_utc": datetime.now(timezone.utc).isoformat(),
                "listener_version": "roomd %s" % __version__,
                "feeds": {"depth": True, "camera": False},
                "port": port,
                "marker_msg_type": rktap.MSG_MARKER,
            })

    @property
    def port(self):
        return self._srv.getsockname()[1]

    @property
    def connected(self):
        return self._conn is not None

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self._requested_port))
        self._srv.listen(1)
        self._srv.settimeout(0.25)
        self._spawn(self._accept_loop, "relay-accept")
        self.log("relay: waiting for ALVR on %s:%d" % (self.host, self.port))

    def _spawn(self, target, name, *args):
        t = threading.Thread(target=target, args=args, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(None)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            old, self._conn = self._conn, conn
            if old is not None:
                _close(old)
            self.connections += 1
            self.log("relay: ALVR connected from %s:%d" % addr)
            try:
                self._send_on(conn, P.MSG_STREAM_CONTROL, P.encode_stream_control(P.STREAM_DEPTH, True))
                self._send_on(conn, P.MSG_STREAM_CONTROL, P.encode_stream_control(P.STREAM_CAMERA, False))
                self._send_on(conn, P.MSG_ROOM_REQUEST, P.encode_room_request(False))
            except OSError as e:
                self.log("relay: handshake failed: %s" % e)
            self._spawn(self._read_loop, "relay-reader", conn)

    def _read_loop(self, conn):
        try:
            while True:
                msg_type, length = struct.unpack("<II", P.recv_exact(conn, 8))
                payload = P.recv_exact(conn, length) if length else b""
                recv_ns = time.time_ns()
                if self.recorder is not None:
                    self.recorder.put(recv_ns, msg_type, payload)
                self.inbox.put(recv_ns, msg_type, payload)
        except (ConnectionError, OSError) as e:
            if not self._stop.is_set():
                self.log("relay: ALVR disconnected (%s)" % e)
        finally:
            if self._conn is conn:
                self._conn = None
            _close(conn)

    def _send_on(self, conn, msg_type, payload):
        # Blocking on purpose: a timeout would also apply to the reader's recv() on the same
        # socket. Control messages are a few bytes and always fit the send buffer.
        with self._send_lock:
            conn.sendall(P.pack_frame(msg_type, payload))

    def send(self, msg_type, payload):
        """Send to the connected streamer; False when nobody is connected or the write failed."""
        conn = self._conn
        if conn is None:
            return False
        try:
            self._send_on(conn, msg_type, payload)
            return True
        except OSError:
            return False

    def request_room(self, recapture=False):
        return self.send(P.MSG_ROOM_REQUEST, P.encode_room_request(recapture))

    def close(self):
        self._stop.set()
        if self._srv is not None:
            _close(self._srv)
        if self._conn is not None:
            _close(self._conn)
        for t in self._threads:
            t.join(2)
        if self.recorder is not None:
            self.recorder.close()
            if self.recorder.error is not None:
                self.log("relay: tap write error: %s" % self.recorder.error)


def _close(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
