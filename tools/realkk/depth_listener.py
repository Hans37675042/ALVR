#!/usr/bin/env python3
"""Minimal viewer for the realkk ALVR depth relay (MSG_DEPTH_FRAME_V2).

The ALVR streamer connects *to* this listener (default 127.0.0.1:9944) when
"Video > XR Data Streaming" is enabled. This script enables the depth feed,
prints resolution / near / far / per-view FOV and the measured fps, and can
optionally dump decoded frames as .npy files.

Standard library only; --dump additionally needs numpy and lz4 (pip install numpy lz4).

Usage:
    python depth_listener.py [--port 9944] [--seconds 30] [--dump OUT_DIR]
"""

import argparse
import math
import socket
import struct
import time
from pathlib import Path

MSG_CAMERA_FRAME = 2
MSG_DEPTH_FRAME_V2 = 3
MSG_STREAM_CONTROL = 100
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9944)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--dump", type=Path, default=None, help="directory for .npy frames")
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print("waiting for ALVR streamer on 127.0.0.1:%d (it retries every 5 s)..." % args.port)
    sock, addr = srv.accept()
    print("streamer connected from", addr)
    send_control(sock, STREAM_DEPTH, True)
    send_control(sock, STREAM_CAMERA, False)

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    first = None
    count = 0
    nbytes = 0
    try:
        while time.monotonic() - start < args.seconds:
            msg_type, length = struct.unpack("<II", recv_exact(sock, 8))
            payload = recv_exact(sock, length)
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
    finally:
        send_control(sock, STREAM_DEPTH, False)
        sock.close()
        srv.close()

    if first is None:
        print("no depth frames received")
        return
    elapsed = time.monotonic() - first
    print("total %d frames in %.1f s -> %.2f fps" % (count, elapsed, (count - 1) / elapsed if elapsed else 0))


if __name__ == "__main__":
    main()
