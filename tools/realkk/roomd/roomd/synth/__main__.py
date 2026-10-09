"""python -m roomd.synth OUT.rktap [--gt OUT.json] [--snapshot OUT.bin] [--seconds 60] [--fps 10]
[--seed 1] [--size 320] [--like REAL.rktap] [--clean] [--no-fake] [--no-snapshot]"""

import argparse
import sys
from pathlib import Path

from .. import protocol as P
from .._legacy import rktap
from .render import DepthCamera, SynthUnavailable
from .scenario import Scenario, generate


def camera_like(path):
    for rec in rktap.TapReader(path):
        if rec.msg_type == P.MSG_DEPTH_FRAME_V2:
            return DepthCamera.from_header(P.decode_depth_v2(rec.payload).header)
    raise SystemExit("%s has no depth frame to copy the camera from" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m roomd.synth", description=__doc__)
    ap.add_argument("tap", type=Path, help="output .rktap")
    ap.add_argument("--gt", type=Path, help="ground-truth JSON (default: <tap>.gt.json)")
    ap.add_argument("--snapshot", type=Path, help="also write the raw MSG_ROOM_SNAPSHOT payload here")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=320, help="per-view width and height in pixels")
    ap.add_argument("--like", type=Path, help="copy size / near / far / FOV / baseline from a real tap")
    ap.add_argument("--clean", action="store_true", help="no sensor noise")
    ap.add_argument("--no-fake", action="store_true", help="no all-0x80 readback failure frames")
    ap.add_argument("--no-snapshot", action="store_true", help="leave the Scene API snapshot out of the tap")
    args = ap.parse_args(argv)

    camera = camera_like(args.like) if args.like else DepthCamera(width=args.size, height=args.size)
    sc = Scenario(seconds=args.seconds, fps=args.fps, seed=args.seed, camera=camera, noisy=not args.clean,
                  fake_every=0 if args.no_fake else 97, snapshot=not args.no_snapshot)
    gt_path = args.gt or args.tap.with_suffix(".gt.json")
    try:
        gt = generate(sc, args.tap, gt_path, args.snapshot, log=print)
    except SynthUnavailable as e:
        print("error:", e, file=sys.stderr)
        return 2
    print("wrote %s (%d depth frames, %d fake) and %s" % (args.tap, gt["frames"], len(gt["fake_frame_indices"]),
                                                          gt_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
