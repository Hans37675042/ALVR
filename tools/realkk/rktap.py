"""Raw recording ("tap") of the realkk XR data relay stream.

A .rktap file keeps every relay message exactly as it arrived, so a recording can be
replayed later (tap_replay.py) to anything that speaks the relay protocol.

Layout (little-endian):
    b"RKTAP"                       magic, 5 bytes
    u32 version                    = 1
    u32 meta_len                   length of the JSON metadata that follows
    u8[meta_len]                   UTF-8 JSON (recording_start_utc, listener_version, feeds, ...)
    then repeated until EOF:
    u64 host_recv_unix_ns          host wall clock when the listener finished reading the message
    u32 msg_type                   relay message type (3 = MSG_DEPTH_FRAME_V2, 2 = camera, ...)
    u32 len
    u8[len]                        relay payload, untouched

Records of type MSG_MARKER are written by the listener itself (never sent by ALVR); their
payload is UTF-8 JSON {"label", "unix_ns", "utc"}.

Standard library only.
"""

import json
import queue
import struct
import threading
from collections import namedtuple
from datetime import datetime, timezone

MAGIC = b"RKTAP"
VERSION = 1
MSG_MARKER = 0xFFFF0001
_RECORD = struct.Struct("<QII")

Record = namedtuple("Record", "recv_ns msg_type payload")


def utc_iso(unix_ns):
    dt = datetime.fromtimestamp(unix_ns // 1_000_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ".%09dZ" % (unix_ns % 1_000_000_000)


def marker_payload(label, unix_ns):
    return json.dumps({"label": label, "unix_ns": unix_ns, "utc": utc_iso(unix_ns)},
                      ensure_ascii=False).encode("utf-8")


def parse_marker(payload):
    return json.loads(payload.decode("utf-8"))


class TapWriter:
    """Synchronous writer. Use TapRecorder when the caller must never block on disk."""

    def __init__(self, path, metadata):
        self._f = open(path, "wb")
        meta = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        self._f.write(MAGIC + struct.pack("<II", VERSION, len(meta)) + meta)

    def write(self, recv_ns, msg_type, payload):
        self._f.write(_RECORD.pack(recv_ns, msg_type, len(payload)))
        self._f.write(payload)

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TapRecorder:
    """TapWriter behind an unbounded queue and a writer thread; put() never touches the disk."""

    _STOP = object()

    def __init__(self, path, metadata):
        self._writer = TapWriter(path, metadata)
        self._queue = queue.Queue()
        self.written = 0
        self.error = None
        self._thread = threading.Thread(target=self._run, name="tap-writer", daemon=True)
        self._thread.start()

    def put(self, recv_ns, msg_type, payload):
        self._queue.put((recv_ns, msg_type, payload))

    def pending(self):
        return self._queue.qsize()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is self._STOP:
                break
            if self.error is not None:
                continue
            try:
                self._writer.write(*item)
                self.written += 1
                if self._queue.empty():
                    self._writer.flush()
            except OSError as e:
                self.error = e

    def close(self):
        self._queue.put(self._STOP)
        self._thread.join()
        self._writer.close()


class TapReader:
    """Iterate a .rktap file. After iteration, .truncated tells whether the tail was cut off."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(len(MAGIC) + 8)
            if len(head) < len(MAGIC) + 8 or head[:len(MAGIC)] != MAGIC:
                raise ValueError("%s is not a .rktap file" % path)
            self.version, meta_len = struct.unpack_from("<II", head, len(MAGIC))
            if self.version != VERSION:
                raise ValueError("unsupported .rktap version %d" % self.version)
            self.metadata = json.loads(f.read(meta_len).decode("utf-8"))
            self._data_offset = f.tell()
        self.truncated = False

    def __iter__(self):
        self.truncated = False
        with open(self.path, "rb") as f:
            f.seek(self._data_offset)
            while True:
                head = f.read(_RECORD.size)
                if not head:
                    return
                if len(head) < _RECORD.size:
                    self.truncated = True
                    return
                recv_ns, msg_type, length = _RECORD.unpack(head)
                payload = f.read(length)
                if len(payload) < length:
                    self.truncated = True
                    return
                yield Record(recv_ns, msg_type, payload)
