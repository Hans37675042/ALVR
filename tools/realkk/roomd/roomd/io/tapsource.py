"""Offline input: feed a .rktap recording into a RelayInbox instead of listening on 9944."""

import threading
import time

from .._legacy import rktap


class TapSource:
    """Plays records into the inbox on a thread. speed: 1.0 = recorded pacing, 0 = as fast
    as possible (the main loop then sees only the newest depth frame, like a slow consumer
    would). Markers are logged, not forwarded. `finished` is set after the last record."""

    def __init__(self, path, inbox, speed=1.0, loop=1, log=None):
        self.reader = rktap.TapReader(path)
        self.inbox = inbox
        self.speed = speed
        self.loop = loop
        self.log = log or (lambda *a: None)
        self.finished = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tap-source", daemon=True)
        self.sent = 0

    @property
    def metadata(self):
        return self.reader.metadata

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            passes = 0
            while not self._stop.is_set() and (self.loop == 0 or passes < self.loop):
                first = None
                start = time.monotonic()
                for rec in self.reader:
                    if self._stop.is_set():
                        return
                    if first is None:
                        first = rec.recv_ns
                    if self.speed > 0:
                        delay = start + (rec.recv_ns - first) / 1e9 / self.speed - time.monotonic()
                        if delay > 0:
                            self._stop.wait(delay)
                    if rec.msg_type == rktap.MSG_MARKER:
                        m = rktap.parse_marker(rec.payload)
                        self.log("tap marker t=%.2f s '%s'" % ((rec.recv_ns - first) / 1e9, m.get("label", "")))
                        continue
                    self.inbox.put(rec.recv_ns, rec.msg_type, rec.payload)
                    self.sent += 1
                passes += 1
        finally:
            self.finished.set()

    def close(self):
        self._stop.set()
        self._thread.join(2)
