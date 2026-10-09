#!/usr/bin/env python3
"""Minimal viewer for the realkk ALVR depth relay (MSG_DEPTH_FRAME_V2, MSG_ROOM_SNAPSHOT).

The ALVR streamer connects *to* this listener (default 127.0.0.1:9944) when
"Video > XR Data Streaming" is enabled. This script enables the depth feed,
prints resolution / near / far / per-view FOV and the measured fps, and can
optionally dump decoded frames as .npy files.

--tap records every relay message verbatim (with host receive time) into a .rktap file
that tap_replay.py can play back and tap_inspect.py can summarise; see rktap.py for the
format. Socket reads run on their own thread and only hand messages to a queue, so a slow
disk or --dump never stalls the relay (it drops the viewer after a 500 ms write timeout).

Quest Space Setup snapshots (MSG_ROOM_SNAPSHOT) and recenter events (MSG_PLAYSPACE_CHANGED)
are summarised on arrival; --request-room asks the streamer for a snapshot right after
connecting (MSG_ROOM_REQUEST; "recapture" asks for Space Setup, which the headset only honours
in the lobby).

Standard library only; --dump additionally needs numpy and lz4 (pip install numpy lz4).

Usage:
    python depth_listener.py [--port 9944] [--seconds 30] [--dump OUT_DIR]
                             [--tap FILE.rktap [--mark-key]] [--camera]
                             [--request-room [resend|recapture]]
"""

import argparse
import json
import math
import queue
import socket
import struct
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import rktap

LISTENER_VERSION = "0.3.0"

MSG_CAMERA_FRAME = 2
MSG_DEPTH_FRAME_V2 = 3
MSG_ROOM_SNAPSHOT = 4
MSG_PLAYSPACE_CHANGED = 5
MSG_STREAM_CONTROL = 100
MSG_ROOM_REQUEST = 101
STREAM_DEPTH = 0
STREAM_CAMERA = 1
FORMAT_NAMES = {0: "RawD16", 1: "H264", 2: "Lz4D16"}


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("streamer closed the connection")
        buf += chunk
    return bytes(buf)


def send_control(sock, stream_id, enabled):
    payload = bytes([stream_id, 1 if enabled else 0])
    sock.sendall(struct.pack("<II", MSG_STREAM_CONTROL, len(payload)) + payload)


def room_request_message(recapture):
    """Framed MSG_ROOM_REQUEST: 0 = resend cached / requery, 1 = ask for Space Setup."""
    payload = bytes([1 if recapture else 0])
    return struct.pack("<II", MSG_ROOM_REQUEST, len(payload)) + payload


def parse_room_snapshot(payload, unpack_mesh=True):
    """MSG_ROOM_SNAPSHOT (CONTRACT-roomd.md). Poses are (px, py, pz, qx, qy, qz, qw), recentered.

    This is the only parser of the layout; roomd.protocol.decode_room_snapshot wraps it.
    unpack_mesh=False leaves each mesh's data as raw little-endian bytes ("vertex_data" f32x3,
    "index_data" u32) instead of Python tuples, for callers that load them into arrays.
    """
    header_size, version, snapshot_id = struct.unpack_from("<III", payload, 0)
    snap = {
        "version": version,
        "snapshot_id": snapshot_id,
        "recenter_pose": struct.unpack_from("<7f", payload, 12),
    }
    off = header_size
    json_len = struct.unpack_from("<I", payload, off)[0]
    off += 4
    snap["scene"] = json.loads(payload[off:off + json_len].decode("utf-8"))
    off += json_len
    mesh_count = struct.unpack_from("<I", payload, off)[0]
    off += 4
    meshes = []
    for _ in range(mesh_count):
        anchor_uuid = str(uuid.UUID(bytes=bytes(payload[off:off + 16])))
        pose = struct.unpack_from("<7f", payload, off + 16)
        vcount, icount = struct.unpack_from("<II", payload, off + 44)
        off += 52
        vertex_data = bytes(payload[off:off + 12 * vcount])
        index_data = bytes(payload[off + 12 * vcount:off + 12 * vcount + 4 * icount])
        if len(vertex_data) != 12 * vcount or len(index_data) != 4 * icount:
            raise struct.error("room snapshot mesh data truncated")
        off += 12 * vcount + 4 * icount
        mesh = {"anchor_uuid": anchor_uuid, "pose": pose, "vcount": vcount, "icount": icount}
        if unpack_mesh:
            flat = struct.unpack("<%df" % (3 * vcount), vertex_data)
            mesh["vertices"] = [flat[i:i + 3] for i in range(0, len(flat), 3)]
            mesh["indices"] = struct.unpack("<%dI" % icount, index_data)
        else:
            mesh["vertex_data"] = vertex_data
            mesh["index_data"] = index_data
        meshes.append(mesh)
    snap["meshes"] = meshes
    return snap


def summarize_room_snapshot(snap):
    anchors = snap["scene"].get("anchors", [])
    labels = Counter()
    for anchor in anchors:
        for label in (anchor.get("labels") or "").split(","):
            if label:
                labels[label] += 1
    return {
        "snapshot_id": snap["snapshot_id"],
        "rooms": len(snap["scene"].get("rooms", [])),
        "anchors": len(anchors),
        "located": sum(1 for a in anchors if a.get("pose") is not None),
        "labels": dict(labels),
        "meshes": len(snap["meshes"]),
        "triangles": sum(m["icount"] // 3 for m in snap["meshes"]),
    }


def parse_playspace_changed(payload):
    version = struct.unpack_from("<I", payload, 0)[0]
    return {"version": version, "recenter_pose": struct.unpack_from("<7f", payload, 4)}


def print_room_snapshot(payload):
    s = summarize_room_snapshot(parse_room_snapshot(payload))
    labels = ", ".join("%s=%d" % kv for kv in sorted(s["labels"].items())) or "-"
    print("room snapshot #%d: %d rooms, %d anchors (%d located), labels: %s; %d meshes, %d triangles"
          % (s["snapshot_id"], s["rooms"], s["anchors"], s["located"], labels, s["meshes"],
             s["triangles"]), flush=True)
    if s["anchors"] == 0:
        print("  (empty: no Space Setup room on the headset)", flush=True)


def parse_depth_v2(payload):
    header_size = struct.unpack_from("<I", payload, 0)[0]
    h = {"header_size": header_size}
    h["client_timestamp_ns"] = struct.unpack_from("<Q", payload, 4)[0]
    h["view_poses"] = [struct.unpack_from("<7f", payload, off) for off in (12, 40)]
    h["width"], h["height"] = struct.unpack_from("<II", payload, 68)
    h["near_z"], h["far_z"] = struct.unpack_from("<ff", payload, 76)
    h["format"] = struct.unpack_from("<I", payload, 84)[0]
    h["fov_angles"] = [struct.unpack_from("<4f", payload, off) for off in (88, 104)]
    h["intrinsics"] = [struct.unpack_from("<4f", payload, off) for off in (120, 136)]
    if header_size >= 176:
        h["server_timestamp_unix_ns"] = struct.unpack_from("<Q", payload, 152)[0]
        h["clock_offset_ns"] = struct.unpack_from("<q", payload, 160)[0]
        h["server_receive_unix_ns"] = struct.unpack_from("<Q", payload, 168)[0]
    return h, payload[header_size:]


def decode_d16(header, data):
    """Return the stacked depth image as a uint16 numpy array (needs numpy, and lz4 for Lz4D16)."""
    import numpy as np

    if header["format"] == 2:
        import lz4.block

        # lz4_flex::compress_prepend_size: u32 LE uncompressed size + raw LZ4 block
        size = struct.unpack_from("<I", data, 0)[0]
        raw = lz4.block.decompress(data[4:], uncompressed_size=size)
    elif header["format"] == 0:
        raw = data
    else:
        raise ValueError("unsupported depth format %d" % header["format"])
    return np.frombuffer(raw, dtype="<u2").reshape(header["height"], header["width"])


def read_messages(sock, out, recorder):
    """Socket reader thread: drain the relay as fast as possible, never touching the disk."""
    try:
        while True:
            msg_type, length = struct.unpack("<II", recv_exact(sock, 8))
            payload = recv_exact(sock, length)
            recv_ns = time.time_ns()
            if recorder is not None:
                recorder.put(recv_ns, msg_type, payload)
            out.put((recv_ns, msg_type, payload))
    except (ConnectionError, OSError) as e:
        out.put(e)


def read_marks(recorder):
    """Each Enter on the terminal writes a marker record; typed text becomes its label."""
    n = 0
    for line in sys.stdin:
        n += 1
        label = line.strip() or "mark %d" % n
        now = time.time_ns()
        recorder.put(now, rktap.MSG_MARKER, rktap.marker_payload(label, now))
        print("marker #%d '%s' at %s" % (n, label, rktap.utc_iso(now)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9944)
    ap.add_argument("--seconds", type=float, default=30.0, help="0 = until Ctrl+C or disconnect")
    ap.add_argument("--dump", type=Path, default=None, help="directory for .npy frames")
    ap.add_argument("--tap", type=Path, default=None, help="record every relay message to this .rktap file")
    ap.add_argument("--mark-key", action="store_true",
                    help="with --tap: press Enter (optionally after typing a label) to write a marker")
    ap.add_argument("--camera", action="store_true", help="also enable the camera feed")
    ap.add_argument("--request-room", nargs="?", const="resend", choices=["resend", "recapture"],
                    help="send MSG_ROOM_REQUEST after connecting (recapture = Space Setup, lobby only)")
    args = ap.parse_args()
    if args.mark_key and not args.tap:
        ap.error("--mark-key needs --tap")
    if args.dump:
        try:
            import lz4.block  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as e:
            sys.exit("--dump needs numpy and lz4 (pip install numpy lz4): %s" % e)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print("waiting for ALVR streamer on 127.0.0.1:%d (it retries every 5 s)..." % args.port)
    sock, addr = srv.accept()
    print("streamer connected from", addr)
    feeds = {"depth": True, "camera": args.camera}
    send_control(sock, STREAM_DEPTH, feeds["depth"])
    send_control(sock, STREAM_CAMERA, feeds["camera"])
    if args.request_room:
        sock.sendall(room_request_message(args.request_room == "recapture"))
        print("room request sent (%s)" % args.request_room)

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)

    recorder = None
    if args.tap:
        args.tap.parent.mkdir(parents=True, exist_ok=True)
        recorder = rktap.TapRecorder(args.tap, {
            "recording_start_utc": datetime.now(timezone.utc).isoformat(),
            "listener_version": LISTENER_VERSION,
            "feeds": feeds,
            "port": args.port,
            "marker_msg_type": rktap.MSG_MARKER,
        })
        print("recording relay messages to", args.tap)
        if args.mark_key:
            threading.Thread(target=read_marks, args=(recorder,), daemon=True).start()
            print("press Enter (optionally type a label first) to write a marker")

    messages = queue.Queue()
    reader = threading.Thread(target=read_messages, args=(sock, messages, recorder), daemon=True)
    reader.start()

    start = time.monotonic()
    first = None
    count = 0
    nbytes = 0
    try:
        while args.seconds <= 0 or time.monotonic() - start < args.seconds:
            try:
                item = messages.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(item, Exception):
                print("streamer disconnected:", item)
                break
            _, msg_type, payload = item
            if msg_type == MSG_ROOM_SNAPSHOT:
                try:
                    print_room_snapshot(payload)
                except (struct.error, ValueError) as e:
                    print("malformed room snapshot (%d bytes): %s" % (len(payload), e))
                continue
            if msg_type == MSG_PLAYSPACE_CHANGED:
                p = parse_playspace_changed(payload)
                print("playspace changed: recenter p(%.3f %.3f %.3f) q(%.3f %.3f %.3f %.3f)"
                      % p["recenter_pose"], flush=True)
                continue
            if msg_type != MSG_DEPTH_FRAME_V2:
                continue
            header, data = parse_depth_v2(payload)
            count += 1
            nbytes += len(data)
            if first is None:
                first = time.monotonic()
                print("first depth frame:")
                print("  size (stacked) : %d x %d  -> per view %d x %d"
                      % (header["width"], header["height"], header["width"], header["height"] // 2))
                print("  format         : %s" % FORMAT_NAMES.get(header["format"], header["format"]))
                print("  near / far (m) : %.4f / %s" % (header["near_z"], header["far_z"]))
                for i in range(2):
                    fov_deg = [math.degrees(a) for a in header["fov_angles"][i]]
                    print("  view %d fov deg : L%.1f R%.1f U%.1f D%.1f" % (i, *fov_deg))
                    print("  view %d fx fy cx cy: %.1f %.1f %.1f %.1f" % (i, *header["intrinsics"][i]))
                    print("  view %d pose   : q(%.3f %.3f %.3f %.3f) p(%.3f %.3f %.3f)"
                          % (i, *header["view_poses"][i]))
                if "clock_offset_ns" in header:
                    print("  clock offset   : %.3f ms (server_unix - client_xr)"
                          % (header["clock_offset_ns"] / 1e6))
            if args.dump:
                import numpy as np

                np.save(args.dump / ("depth_%06d.npy" % count), decode_d16(header, data))
            if count % 50 == 0:
                elapsed = time.monotonic() - first
                print("%d frames, %.2f fps, %.2f Mbps"
                      % (count, (count - 1) / elapsed if elapsed else 0, nbytes * 8 / elapsed / 1e6 if elapsed else 0))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        try:
            send_control(sock, STREAM_DEPTH, False)
            send_control(sock, STREAM_CAMERA, False)
        except OSError:
            pass
        sock.close()
        srv.close()
        reader.join(2)
        if recorder is not None:
            print("flushing %d queued messages to %s ..." % (recorder.pending(), args.tap))
            recorder.close()
            if recorder.error is not None:
                print("tap write error:", recorder.error)
            print("tap: %d messages written" % recorder.written)

    if first is None:
        print("no depth frames received")
        return
    elapsed = time.monotonic() - first
    print("total %d frames in %.1f s -> %.2f fps" % (count, elapsed, (count - 1) / elapsed if elapsed else 0))


if __name__ == "__main__":
    main()
