"""Offline fusion of a .rktap recording: python -m roomd.fusion.replay TAP [--out DIR].

Feeds every MSG_DEPTH_FRAME_V2 of the tap through TsdfFusion as fast as possible and writes
  mesh.ply        fused mesh (Unity stage space, floor excluded), binary PLY
  heightmap.png   top-down NAV_HEIGHTMAP: black unknown, grey walkable, red-yellow obstacle
                  height; image up = +Z (Unity forward), right = +X
  stats.json      frame counts, integration timings, floor fit, mesh size, markers

--make-synthetic FILE writes a floor + furniture tap (with one fake readback frame) instead,
for trying the pipeline without a headset recording.
"""

import argparse
import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from .config import FusionConfig
from .decode import MSG_DEPTH_FRAME_V2
from .outputs import FLAG_KNOWN, FLAG_OBSTACLE, FLAG_WALKABLE
from .synthscene import SynthScene, encode_payload
from .tsdf import TsdfFusion

# .rktap layout: tools/realkk/rktap.py (kept dependency-free here)
TAP_MAGIC = b"RKTAP"
TAP_VERSION = 1
MSG_MARKER = 0xFFFF0001
_RECORD = struct.Struct("<QII")


def read_tap(path):
    """Yield (recv_ns, msg_type, payload); stops quietly at a truncated tail."""
    with open(path, "rb") as f:
        head = f.read(len(TAP_MAGIC) + 8)
        if head[:len(TAP_MAGIC)] != TAP_MAGIC:
            raise ValueError("%s is not a .rktap file" % path)
        version, meta_len = struct.unpack_from("<II", head, len(TAP_MAGIC))
        if version != TAP_VERSION:
            raise ValueError("unsupported .rktap version %d" % version)
        f.read(meta_len)
        while True:
            rec = f.read(_RECORD.size)
            if len(rec) < _RECORD.size:
                return
            recv_ns, msg_type, length = _RECORD.unpack(rec)
            payload = f.read(length)
            if len(payload) < length:
                return
            yield recv_ns, msg_type, payload


def write_synthetic_tap(path, frames=60, fps=10.0):
    """Room with a table, a chair-sized box and a wall, scanned on an arc; frame 3 is fake."""
    scene = SynthScene(floor_y=0.0, boxes=[
        ((0.25, 0.0, 1.25), (0.75, 0.45, 1.75)),     # stool
        ((-1.2, 0.0, 1.0), (-0.4, 0.72, 1.8)),       # table block
        ((-3.0, 0.0, 3.0), (3.0, 2.6, 3.2)),         # wall
    ])
    meta = json.dumps({"synthetic": True, "frames": frames}).encode()
    t0 = time.time_ns()
    with open(path, "wb") as f:
        f.write(TAP_MAGIC + struct.pack("<II", TAP_VERSION, len(meta)) + meta)

        def put(i, msg_type, payload):
            f.write(_RECORD.pack(t0 + int(i * 1e9 / fps), msg_type, len(payload)) + payload)

        for i in range(frames):
            a = math.radians(-60 + 120 * i / max(frames - 1, 1))
            eye = (1.4 * math.sin(a), 1.6, 1.4 - 1.6 * math.cos(a))
            target = (0.6 * math.sin(2.5 * a), 0.3, 1.6)
            if i == 3:
                pose = ((0.0, 1.6, 0.0), (0.0, 0.0, 0.0, 1.0))
                fake = np.full((512, 256), 0x8080, np.uint16)
                put(i, MSG_DEPTH_FRAME_V2, encode_payload(fake, [pose, pose], [(128.0, 128.0, 128.0, 128.0)] * 2,
                                                          0.1, math.inf))
            put(i, MSG_DEPTH_FRAME_V2, scene.frame_payload(eye, target, client_ts=i))
        label = b'{"label": "synthetic end"}'
        put(frames, MSG_MARKER, label)


def write_ply(path, verts, indices):
    tris = np.asarray(indices, np.uint32).reshape(-1, 3)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\ncomment Unity left-handed stage space\n"
                 "element vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                 "element face %d\nproperty list uchar uint vertex_indices\nend_header\n"
                 % (len(verts), len(tris))).encode())
        f.write(np.asarray(verts, "<f4").tobytes())
        rows = np.zeros(len(tris), dtype=[("n", "u1"), ("i", "<u4", 3)])
        rows["n"] = 3
        rows["i"] = tris
        f.write(rows.tobytes())


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def heightmap_rgb(hm):
    rgb = np.zeros((hm.height, hm.width, 3), np.uint8)
    known = (hm.flags & FLAG_KNOWN) > 0
    rgb[known] = (60, 60, 60)
    rgb[(hm.flags & FLAG_WALKABLE) > 0] = (150, 150, 150)
    obst = (hm.flags & FLAG_OBSTACLE) > 0
    h = np.clip((hm.top_y - hm.floor_y) / 1.2, 0, 1)
    rgb[obst, 0] = 255
    rgb[obst, 1] = (h[obst] * 230).astype(np.uint8)
    rgb[obst, 2] = 40
    return rgb[::-1].copy()  # row 0 = max z


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m roomd.fusion.replay", description=__doc__.split("\n")[0])
    ap.add_argument("tap", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <tap>.fusion)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--until", type=float, default=0.0, metavar="S",
                    help="stop at this many seconds after the first tap record (host receive time)")
    ap.add_argument("--voxel", type=float, default=None, help="voxel size in metres")
    ap.add_argument("--save-state", type=Path, default=None, metavar="FILE.npz",
                    help="also save the fused volume (TsdfFusion.load_state reads it back)")
    ap.add_argument("--flip-rows", action="store_true", help="depth rows are bottom-up")
    ap.add_argument("--make-synthetic", type=Path, default=None, metavar="FILE",
                    help="write a synthetic tap to FILE (and replay it when no TAP is given)")
    args = ap.parse_args(argv)
    if args.make_synthetic:
        write_synthetic_tap(args.make_synthetic)
        print("synthetic tap written to", args.make_synthetic)
        if args.tap is None:
            args.tap = args.make_synthetic
    if args.tap is None:
        ap.error("TAP is required")
    out = args.out or args.tap.with_suffix(".fusion")
    out.mkdir(parents=True, exist_ok=True)

    cfg = FusionConfig(flip_rows=args.flip_rows)
    if args.voxel:
        cfg.voxel_size = args.voxel
    fusion = TsdfFusion(cfg)
    counts = {"messages": 0, "depth": 0, "malformed": 0, "other": 0}
    markers = []
    first_ns = last_ns = None
    t_start = time.perf_counter()
    for recv_ns, msg_type, payload in read_tap(args.tap):
        first_ns = recv_ns if first_ns is None else first_ns
        if args.until and (recv_ns - first_ns) / 1e9 > args.until:
            break
        counts["messages"] += 1
        last_ns = recv_ns
        if msg_type == MSG_MARKER:
            try:
                markers.append(json.loads(payload.decode("utf-8")).get("label"))
            except ValueError:
                markers.append(None)
            continue
        if msg_type != MSG_DEPTH_FRAME_V2:
            counts["other"] += 1
            continue
        counts["depth"] += 1
        try:
            fusion.integrate(payload)
        except ValueError as e:
            counts["malformed"] += 1
            print("frame %d: %s" % (counts["depth"], e), file=sys.stderr)
        if args.max_frames and counts["depth"] >= args.max_frames:
            break
    integrate_s = time.perf_counter() - t_start

    t0 = time.perf_counter()
    snap = fusion.snapshot_outputs(force=True)
    verts, idx = fusion.extract_mesh()
    mesh_s = time.perf_counter() - t0
    write_ply(out / "mesh.ply", verts, idx)
    if snap.heightmap is not None:
        write_png(out / "heightmap.png", heightmap_rgb(snap.heightmap))
    stats = dict(fusion.stats())
    stats.update({
        "tap": str(args.tap), "counts": counts, "markers": markers,
        "tapSeconds": (last_ns - first_ns) / 1e9 if first_ns is not None else 0.0,
        "replaySeconds": integrate_s, "meshSeconds": mesh_s,
        "meshVertices": int(len(verts)), "meshTriangles": int(len(idx) // 3),
        "meshChunks": sum(1 for c in snap.mesh_chunks if len(c.indices)),
        "floor": None if snap.floor is None else {"y": snap.floor.y, "normal": list(snap.floor.normal),
                                                  "rms": snap.floor.rms, "count": snap.floor.count},
        "bounds": fusion.bounds(),
    })
    if snap.heightmap is not None:
        f = snap.heightmap.flags
        stats["heightmapCells"] = {"known": int(np.count_nonzero(f & FLAG_KNOWN)),
                                   "obstacle": int(np.count_nonzero(f & FLAG_OBSTACLE)),
                                   "walkable": int(np.count_nonzero(f & FLAG_WALKABLE))}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    if args.save_state:
        fusion.save_state(args.save_state)
    print(json.dumps({k: stats[k] for k in ("frames", "fakeFrames", "integrateMsP50", "integrateMsP95",
                                            "meshVertices", "meshTriangles", "floor")}, indent=2))
    print("written to", out)
    return stats


if __name__ == "__main__":
    main()
