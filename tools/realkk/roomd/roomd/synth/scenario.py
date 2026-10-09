"""Scripted synthetic session -> .rktap (relay messages verbatim + markers) + ground truth.

Default 60 s timeline (times scale with --seconds):
  0-33 %   scan the room walking an ellipse, yaw sweeping, looking down ~30 deg
  33-37 %  chair (with its clothes pile) slides +1 m along X while the user watches
  50-60 %  a person walks across between the couch and the coffee table
  67-83 %  the user sits on the middle couch seat, then stands up
  rest     look around
Every `fake_every`-th depth frame is a failed-readback frame (all 0x80 bytes).
"""

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .. import coords
from .. import model as M
from .. import protocol as P
from .._legacy import rktap
from .noise import NoiseModel
from .render import DepthCamera, Renderer, view_poses_xr
from .room import block, default_room
from .snapshot import build_snapshot

T0_NS = 1_760_000_000_000_000_000  # fixed epoch so outputs are reproducible
SYNTH_VERSION = "roomd-synth 0.1.0"


@dataclass
class Scenario:
    seconds: float = 60.0
    fps: float = 10.0
    seed: int = 1
    camera: DepthCamera = field(default_factory=DepthCamera)
    noise: NoiseModel = field(default_factory=NoiseModel)
    noisy: bool = True
    fake_every: int = 97  # 0 = no fake frames
    snapshot: bool = True
    chair_move: tuple = (1.0, 0.0)
    fmt: int = P.FORMAT_LZ4_D16

    def phase(self, frac):
        return frac * self.seconds


def _head_pose(pos, yaw, pitch_down):
    q = coords.quat_mul(coords.yaw_quat(yaw), coords.quat_from_axis_angle((1, 0, 0), pitch_down))
    return tuple(float(v) for v in pos) + tuple(q)


def _look_at_yaw(pos, target):
    return math.atan2(target[0] - pos[0], target[1] - pos[2])


def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Session:
    """Evaluates the room state and the head pose at time t for a Scenario."""

    def __init__(self, sc, room=None):
        self.sc = sc
        self.room0 = room or default_room()
        couch = self.room0.find("couch")
        seats = M.generate_seats(couch.to_object(M.SOURCE_MANUAL), couch.seat_height)
        self.sit_seat = seats[len(seats) // 2]
        s = sc.seconds
        self.t_move = (s * 0.33, s * 0.37)
        self.t_walk = (s * 0.50, s * 0.60)
        self.t_sit = (s * 0.67, s * 0.70, s * 0.80, s * 0.83)  # sit down, seated, stand up, standing

    def room_at(self, t):
        room = self.room0.copy()
        a, b = self.t_move
        k = _smooth((t - a) / (b - a))
        if k > 0:
            room.move("chair", self.sc.chair_move[0] * k, self.sc.chair_move[1] * k)
        a, b = self.t_walk
        if a <= t <= b:
            x = -2.2 + 4.0 * (t - a) / (b - a)
            room.furniture.append(block("person", M.Kind.Other, (x, 0.95), math.pi / 2, (0.3, 1.7, 0.45),
                                        label=None))
        return room

    def head_at(self, t):
        s = self.sc.seconds
        a, b = self.t_move
        if t < a:  # scan
            ph = 2 * math.pi * t / a
            pos = (0.3 + 1.3 * math.cos(ph), 1.6, 0.0 + 0.9 * math.sin(ph))
            yaw = 2 * ph + 0.5 * math.sin(3 * ph)
            return _head_pose(pos, yaw, math.radians(30 + 10 * math.sin(5 * ph)))
        chair = self.room0.find("chair").center_xz
        watch = (-0.9, 1.6, -0.6)
        if t < self.t_walk[0]:
            target = (chair[0] + self.sc.chair_move[0] * _smooth((t - a) / (b - a)), chair[1])
            return _head_pose(watch, _look_at_yaw(watch, target), math.radians(35))
        stand = (-0.8, 1.6, 0.75)
        if t < self.t_sit[0]:
            target = (0.0, 0.95) if t <= self.t_walk[1] else (0.3, 0.2)
            return _head_pose(stand, _look_at_yaw(stand, target), math.radians(20))
        sp = self.sit_seat.Pose.Position
        fwd = self.sit_seat.Pose.Rotation.rotate((0.0, 0.0, 1.0))
        seated = (sp.X + 0.1 * fwd[0], sp.Y + 0.72, sp.Z + 0.1 * fwd[2])
        yaw = math.atan2(fwd[0], fwd[2])
        t0, t1, t2, t3 = self.t_sit
        if t < t3:
            k = _smooth((t - t0) / (t1 - t0)) if t < t2 else 1.0 - _smooth((t - t2) / (t3 - t2))
            return _head_pose(_lerp(stand, seated, k), yaw, math.radians(15))
        ph = 2 * math.pi * (t - t3) / max(s - t3, 1e-6)
        return _head_pose(stand, yaw + 2.5 * math.sin(ph), math.radians(25))

    def markers(self):
        a, b = self.t_move
        w0, w1 = self.t_walk
        s0, s1, s2, s3 = self.t_sit
        return [(0.0, "scan_start"), (a, "chair_move_start"), (b, "chair_move_end"),
                (w0, "person_enter"), (w1, "person_exit"), (s0, "user_sit_start"), (s1, "user_seated"),
                (s2, "user_stand_start"), (s3, "user_standing")]

    def ground_truth(self, frames, fake_indices):
        def objs(room):
            return [M.object_to_dict(o) for o in room.gt_objects()]

        def seats(room):
            return [M.seat_to_dict(s) for s in room.gt_seats()]

        end = self.room_at(self.sc.seconds)
        cam = self.sc.camera
        return {
            "generator": SYNTH_VERSION,
            "frame": "Unity left-handed stage space (RoomModel v2 coordinates)",
            "seconds": self.sc.seconds, "fps": self.sc.fps, "seed": self.sc.seed, "frames": frames,
            "fake_frame_indices": fake_indices,
            "camera": {"width": cam.width, "height_per_view": cam.height, "fov": cam.fov, "near": cam.near,
                       "far": "inf" if math.isinf(cam.far) else cam.far, "baseline": cam.baseline},
            "noise": vars(self.sc.noise) if self.sc.noisy else None,
            "room": {"x": [self.room0.x_min, self.room0.x_max], "z": [self.room0.z_min, self.room0.z_max],
                     "floor_y": self.room0.floor_y, "wall_height": self.room0.wall_height},
            "walls": [{"a": list(a), "b": list(b)} for _, a, b, _ in self.room0.walls()],
            "scene_api_labels": {f.name: f.label for f in self.room0.furniture},
            "scene_api_ids": {f.name: "scene:" + f.uuid for f in self.room0.furniture if f.label},
            "objects_initial": objs(self.room0),
            "seats_initial": seats(self.room0),
            "objects_final": objs(end),
            "seats_final": seats(end),
            "events": [
                {"t0": self.t_move[0], "t1": self.t_move[1], "type": "object_moved", "object": "chair",
                 "delta_xz": list(self.sc.chair_move), "carries": ["clothes"]},
                {"t0": self.t_walk[0], "t1": self.t_walk[1], "type": "person_walk", "path_z": 0.95,
                 "x": [-2.2, 1.8]},
                {"t0": self.t_sit[1], "t1": self.t_sit[2], "type": "user_seated", "seat": self.sit_seat.Id,
                 "seat_state": M.SeatState.OccupiedByUser.value},
            ],
            "markers": [{"t": t, "label": lb} for t, lb in self.markers()],
        }


def generate(sc, tap_path, gt_path=None, snapshot_path=None, log=None):
    """Render the scenario into tap_path (and the ground truth / raw snapshot payload)."""
    log = log or (lambda *a: None)
    rng = np.random.default_rng(sc.seed)
    session = Session(sc)
    renderer = Renderer(sc.camera)
    n = int(round(sc.seconds * sc.fps))
    dt_ns = int(1e9 / sc.fps)
    fake_indices = []
    marks = sorted(session.markers())
    meta = {"recording_start_utc": rktap.utc_iso(T0_NS), "listener_version": SYNTH_VERSION,
            "feeds": {"depth": True, "camera": False}, "synthetic": True, "seed": sc.seed,
            "fps": sc.fps, "marker_msg_type": rktap.MSG_MARKER}
    snap_payload = P.encode_room_snapshot(build_snapshot(session.room0, snapshot_id=1))
    if snapshot_path is not None:
        with open(snapshot_path, "wb") as f:
            f.write(snap_payload)
    with rktap.TapWriter(tap_path, meta) as w:
        if sc.snapshot:
            w.write(T0_NS, P.MSG_ROOM_SNAPSHOT, snap_payload)
        mi = 0
        for i in range(n):
            t = i / sc.fps
            recv_ns = T0_NS + i * dt_ns + 5_000_000
            while mi < len(marks) and marks[mi][0] <= t:
                ns = T0_NS + int(marks[mi][0] * 1e9)
                w.write(ns, rktap.MSG_MARKER, rktap.marker_payload(marks[mi][1], ns))
                mi += 1
            room = session.room_at(t)
            boxes = room.shell_boxes() + [b for f in room.furniture for b in f.world_boxes()]
            key = tuple((f.name, round(f.center_xz[0], 4), round(f.center_xz[1], 4), round(f.yaw, 4))
                        for f in room.furniture)
            renderer.set_geometry(boxes, key)
            poses = view_poses_xr(session.head_at(t), sc.camera.baseline)
            if sc.fake_every and i % sc.fake_every == sc.fake_every - 1:
                d16 = np.full((2 * sc.camera.height, sc.camera.width), 0x8080, np.uint16)
                fake_indices.append(i)
            else:
                views = []
                hand = sc.noisy and rng.random() < sc.noise.hand_p  # a hand shows in both views
                for v in range(2):
                    z = renderer.render_view(v, poses[v])
                    if sc.noisy:
                        z = sc.noise.apply(z, rng, hand=hand)
                    views.append(P.encode_d16_from_z(z, sc.camera.near, sc.camera.far))
                d16 = np.vstack(views)
            wire = [tuple(p[3:]) + tuple(p[:3]) for p in poses]  # header order: q then p
            payload = P.encode_depth_v2(d16, sc.camera.near, sc.camera.far, wire, sc.camera.fov, fmt=sc.fmt,
                                        client_timestamp_ns=i * dt_ns, server_timestamp_unix_ns=T0_NS + i * dt_ns,
                                        clock_offset_ns=T0_NS, server_receive_unix_ns=recv_ns - 1_000_000)
            w.write(recv_ns, P.MSG_DEPTH_FRAME_V2, payload)
            if (i + 1) % 100 == 0:
                log("synth: %d / %d frames" % (i + 1, n))
    gt = session.ground_truth(n, fake_indices)
    if gt_path is not None:
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=1)
    return gt
