"""Synthetic relay messages built from the MSG_DEPTH_FRAME_V2 layout in REALKK.md."""

import struct

HEADER_SIZE = 176
IDENTITY_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 1.6, 0.0)


def depth_payload(width=4, height=4, fmt=2, pixels=None, near=0.1, far=float("inf"),
                  client_ts=1_000, poses=(IDENTITY_POSE, IDENTITY_POSE),
                  fov=((-0.9, 0.8, 0.85, -0.95), (-0.8, 0.9, 0.85, -0.95)),
                  intrinsics=((100.0, 101.0, 50.0, 51.0), (102.0, 103.0, 52.0, 53.0)),
                  server_ts=2_000, clock_offset=-500, server_recv=2_100):
    """Build a depth payload. pixels: uint16 list (row-major, stacked views), None for a ramp."""
    if pixels is None:
        pixels = [(i * 997) & 0xFFFF for i in range(width * height)]
    raw = struct.pack("<%dH" % len(pixels), *pixels)
    if fmt == 2:
        import lz4.block

        # same framing as lz4_flex::compress_prepend_size
        data = lz4.block.compress(raw, store_size=True)
    else:
        data = raw
    buf = bytearray(HEADER_SIZE)
    struct.pack_into("<I", buf, 0, HEADER_SIZE)
    struct.pack_into("<Q", buf, 4, client_ts)
    struct.pack_into("<7f", buf, 12, *poses[0])
    struct.pack_into("<7f", buf, 40, *poses[1])
    struct.pack_into("<II", buf, 68, width, height)
    struct.pack_into("<ff", buf, 76, near, far)
    struct.pack_into("<I", buf, 84, fmt)
    struct.pack_into("<4f", buf, 88, *fov[0])
    struct.pack_into("<4f", buf, 104, *fov[1])
    struct.pack_into("<4f", buf, 120, *intrinsics[0])
    struct.pack_into("<4f", buf, 136, *intrinsics[1])
    struct.pack_into("<QqQ", buf, 152, server_ts, clock_offset, server_recv)
    return bytes(buf) + data


def fake_depth_payload(width=4, height=4, fmt=2):
    """Frame produced by a failed GPU readback: every D16 sample is 0x8080."""
    return depth_payload(width, height, fmt, pixels=[0x8080] * (width * height))
