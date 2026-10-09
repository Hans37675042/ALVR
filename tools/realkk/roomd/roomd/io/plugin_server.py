"""9945 plugin server: roomd -> KKS realkk plugin (several clients, e.g. across KKS restarts).

publish() never blocks the caller: every client has its own writer thread and queue. A
client whose queue exceeds MAX_PENDING_BYTES (a stalled plugin) is dropped and will
reconnect. Messages published with `remember` are replayed to clients that connect later
(the latest ROOM_MODEL, NAV_HEIGHTMAP and every live MESH_CHUNK), so a new plugin instance
gets the full state before any later update.
"""

import socket
import struct
import threading
from collections import OrderedDict, deque

from .. import protocol as P

MAX_PENDING_BYTES = 64 * 1024 * 1024


class _Client:
    def __init__(self, server, sock, addr):
        self.server = server
        self.sock = sock
        self.addr = addr
        self.hello = None
        self._queue = deque()
        self._pending = 0
        self._cond = threading.Condition()
        self._closed = False
        self._threads = [threading.Thread(target=self._write_loop, name="plugin-writer", daemon=True),
                         threading.Thread(target=self._read_loop, name="plugin-reader", daemon=True)]

    def start(self):
        for t in self._threads:
            t.start()

    def enqueue(self, frame):
        with self._cond:
            if self._closed:
                return
            if self._pending + len(frame) > MAX_PENDING_BYTES:
                self.server.log("plugin: client %s:%d stalled, dropping it" % self.addr)
                self._close_locked()
                return
            self._queue.append(frame)
            self._pending += len(frame)
            self._cond.notify()

    def _write_loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._queue or self._closed)
                if self._closed:
                    return
                frame = self._queue.popleft()
                self._pending -= len(frame)
            try:
                self.sock.sendall(frame)
            except OSError:
                self.close()
                return

    def _read_loop(self):
        try:
            while True:
                msg_type, length = struct.unpack("<II", P.recv_exact(self.sock, 8))
                payload = P.recv_exact(self.sock, length) if length else b""
                if msg_type == P.PLUGIN_HELLO:
                    try:
                        self.hello = P.decode_json(payload)
                    except ValueError:
                        self.hello = {"raw": payload.decode("utf-8", "replace")}
                    self.server._on_hello(self, self.hello)
        except (ConnectionError, OSError):
            pass
        finally:
            self.close()

    def _close_locked(self):
        if self._closed:
            return
        self._closed = True
        self._queue.clear()
        self._cond.notify_all()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.server._on_closed(self)

    def close(self):
        with self._cond:
            self._close_locked()


class PluginServer:
    def __init__(self, port, host="127.0.0.1", log=print):
        self.host = host
        self.log = log
        self._requested_port = port
        self._srv = None
        self._lock = threading.RLock()
        self._clients = []
        self._remembered = OrderedDict()
        self._stop = threading.Event()
        self._accept_thread = None
        self.hellos = []

    @property
    def port(self):
        return self._srv.getsockname()[1]

    @property
    def client_count(self):
        with self._lock:
            return len(self._clients)

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self._requested_port))
        self._srv.listen(8)
        self._srv.settimeout(0.25)
        self._accept_thread = threading.Thread(target=self._accept_loop, name="plugin-accept", daemon=True)
        self._accept_thread.start()
        self.log("plugin: listening on %s:%d" % (self.host, self.port))

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                sock, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client = _Client(self, sock, addr)
            with self._lock:
                for msg_type, payload in self._remembered.values():
                    client.enqueue(P.pack_frame(msg_type, payload))
                self._clients.append(client)
            client.start()
            self.log("plugin: client connected from %s:%d" % addr)

    def publish(self, msg_type, payload, remember=None):
        """Send to every client. remember=True keeps it as the latest of its type for late
        clients; a hashable key keeps one per key (see forget())."""
        frame = P.pack_frame(msg_type, payload)
        with self._lock:
            if remember is not None and remember is not False:
                key = msg_type if remember is True else remember
                self._remembered.pop(key, None)
                self._remembered[key] = (msg_type, payload)
            clients = list(self._clients)
        for c in clients:
            c.enqueue(frame)

    def forget(self, key):
        with self._lock:
            self._remembered.pop(key, None)

    def _on_hello(self, client, hello):
        self.hellos.append(hello)
        self.log("plugin: hello from %s:%d %s" % (client.addr + (hello,)))

    def _on_closed(self, client):
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)
        if not self._stop.is_set():
            self.log("plugin: client %s:%d disconnected" % client.addr)

    def close(self):
        self._stop.set()
        if self._srv is not None:
            self._srv.close()
        if self._accept_thread is not None:
            self._accept_thread.join(2)
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            c.close()
