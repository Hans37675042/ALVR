#!/usr/bin/env python3
"""Summarise a .rktap recording made by depth_listener.py --tap.

Prints message counts per type, duration, depth fps, the first depth frame's header
(size, near/far, FOV, intrinsics, pose), markers, and how many depth frames are fake:
a failed GPU readback on the headset yields a frame whose decoded D16 samples are all
0x8080 (every byte 0x80).

Standard library only; fake-frame detection of Lz4D16 frames needs lz4 (pip install lz4).

Usage:
    python tap_inspect.py FILE.rktap [--no-decode]
"""

import argparse
import math
import struct
import sys
from collections import Counter
from pathlib import Path

import rktap
from depth_listener import FORMAT_NAMES, MSG_CAMERA_FRAME, MSG_DEPTH_FRAME_V2, parse_depth_v2

TYPE_NAMES = {MSG_DEPTH_FRAME_V2: "depth_v2", MSG_CAMERA_FRAME: "camera", rktap.MSG_MARKER: "marker"}


def d16_bytes(header, data):
    """Decoded D16 bytes of a depth frame (Lz4D16 needs the lz4 package)."""
    if header["format"] == 2:
        import lz4.block

        size = struct.unpack_from("<I", data, 0)[0]
        return lz4.block.decompress(data[4:], uncompressed_size=size)
    if header["format"] == 0:
        return data
    raise ValueError("unsupported depth format %d" % header["format"])


def is_fake_frame(header, data):
    raw = d16_bytes(header, data)
    return len(raw) > 0 and raw.count(0x80) == len(raw)


def summarize(path, decode=True):
    reader = rktap.TapReader(path)
    by_type = Counter()
    first_ns = last_ns = None
    depth_times = []
    first_depth = None
    markers = []
    fake_indices = []
    decode_errors = 0
    for rec in reader:
        by_type[rec.msg_type] += 1
        # marker and socket records come from different threads, so the file is only
        # roughly time-ordered
        if first_ns is None:
            first_ns = last_ns = rec.recv_ns
        first_ns = min(first_ns, rec.recv_ns)
        last_ns = max(last_ns, rec.recv_ns)
        if rec.msg_type == rktap.MSG_MARKER:
            m = rktap.parse_marker(rec.payload)
            m["t_s"] = (rec.recv_ns - first_ns) / 1e9
            markers.append(m)
        elif rec.msg_type == MSG_DEPTH_FRAME_V2:
            index = len(depth_times)
            depth_times.append(rec.recv_ns)
            header, data = parse_depth_v2(rec.payload)
            if first_depth is None:
                first_depth = header
            if decode:
                try:
                    if is_fake_frame(header, data):
                        fake_indices.append(index)
                except Exception:  # corrupt or unsupported payload
                    decode_errors += 1
    depth_span = (depth_times[-1] - depth_times[0]) / 1e9 if len(depth_times) > 1 else 0.0
    return {
        "metadata": reader.metadata,
        "messages": sum(by_type.values()),
        "by_type": dict(by_type),
        "duration_s": (last_ns - first_ns) / 1e9 if first_ns is not None else 0.0,
        "depth_frames": len(depth_times),
        "depth_fps": (len(depth_times) - 1) / depth_span if depth_span else 0.0,
        "first_depth": first_depth,
        "markers": markers,
        "fake_frames": len(fake_indices) if decode else None,
        "fake_frame_indices": fake_indices,
        "decode_errors": decode_errors,
        "truncated": reader.truncated,
    }


def print_summary(path, s):
    print("file           : %s" % path)
    for key, value in s["metadata"].items():
        print("  meta %-9s: %s" % (key, value))
    print("messages       : %d%s" % (s["messages"], "  (TRUNCATED tail)" if s["truncated"] else ""))
    for msg_type, n in sorted(s["by_type"].items()):
        print("  type %-10s %s: %d" % ("0x%X" % msg_type, TYPE_NAMES.get(msg_type, "?"), n))
    print("duration       : %.2f s" % s["duration_s"])
    print("depth frames   : %d, %.2f fps" % (s["depth_frames"], s["depth_fps"]))
    h = s["first_depth"]
    if h is not None:
        print("first depth frame:")
        print("  size (stacked) : %d x %d  -> per view %d x %d"
              % (h["width"], h["height"], h["width"], h["height"] // 2))
        print("  format         : %s" % FORMAT_NAMES.get(h["format"], h["format"]))
        print("  near / far (m) : %.4f / %s" % (h["near_z"], h["far_z"]))
        for i in range(2):
            fov_deg = [math.degrees(a) for a in h["fov_angles"][i]]
            print("  view %d fov deg : L%.1f R%.1f U%.1f D%.1f" % (i, *fov_deg))
            print("  view %d fx fy cx cy: %.1f %.1f %.1f %.1f" % (i, *h["intrinsics"][i]))
            print("  view %d pose   : q(%.3f %.3f %.3f %.3f) p(%.3f %.3f %.3f)" % (i, *h["view_poses"][i]))
        if "clock_offset_ns" in h:
            print("  clock offset   : %.3f ms (server_unix - client_xr)" % (h["clock_offset_ns"] / 1e6))
    if s["fake_frames"] is None:
        print("fake frames    : not checked (--no-decode)")
    else:
        pct = 100.0 * s["fake_frames"] / s["depth_frames"] if s["depth_frames"] else 0.0
        print("fake frames    : %d / %d (%.1f%%) all-0x80 readback failures"
              % (s["fake_frames"], s["depth_frames"], pct))
        if s["fake_frame_indices"]:
            shown = s["fake_frame_indices"][:20]
            print("  depth indices  : %s%s" % (shown, " ..." if len(s["fake_frame_indices"]) > 20 else ""))
        if s["decode_errors"]:
            print("  decode errors  : %d" % s["decode_errors"])
    print("markers        : %d" % len(s["markers"]))
    for m in s["markers"]:
        print("  t=%8.2f s  %s  %s" % (m["t_s"], m.get("utc", ""), m.get("label", "")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tap", type=Path)
    ap.add_argument("--no-decode", action="store_true", help="skip fake-frame detection (no lz4 needed)")
    args = ap.parse_args(argv)
    decode = not args.no_decode
    if decode:
        try:
            import lz4.block  # noqa: F401
        except ImportError:
            sys.exit("fake-frame detection needs lz4 (pip install lz4), or pass --no-decode")
    print_summary(args.tap, summarize(args.tap, decode=decode))


if __name__ == "__main__":
    main()
