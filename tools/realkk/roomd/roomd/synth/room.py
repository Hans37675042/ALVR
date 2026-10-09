"""Synthetic test room, defined in Unity stage space (the frame of RoomModel / ground truth).

Every piece of furniture is a set of boxes in its own local frame (Unity: +Y up, local +Z =
front, Pose at the bottom-face centre), so the same definition drives the depth renderer
(after the flip to OpenXR), the synthetic Scene API snapshot and the ground-truth JSON.
Pure numpy; rendering lives in render.py (Open3D).
"""

import copy
import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from .. import model as M

UUID_NS = uuid.UUID("6f1d2a2e-6d0a-4c39-9a51-7e0f5b9c1a11")


def synth_uuid(name):
    return str(uuid.uuid5(UUID_NS, name))


@dataclass
class Part:
    center: tuple  # local box centre (x, y, z)
    size: tuple  # (sx, sy, sz)


@dataclass
class Furniture:
    name: str
    kind: M.Kind
    center_xz: tuple  # bottom-face centre, Unity stage
    yaw: float  # Unity yaw: local +Z -> (sin yaw, 0, cos yaw)
    size: tuple  # overall (X width, Y height, Z depth)
    parts: list
    label: str = None  # Meta Scene label; None = not reported by the Scene API (chairs)
    seat_height: float = None  # sittable when set
    seat_count: int = None  # override (MRUK single-seat couch)
    blocked: bool = False  # something lies on the seat
    movable: bool = True
    attached_to: str = None  # moves with that furniture (clothes pile on a chair)

    @property
    def uuid(self):
        return synth_uuid(self.name)

    def pose(self):
        return M.Pose(M.Vec3(self.center_xz[0], 0.0, self.center_xz[1]), M.Quat.from_yaw(self.yaw))

    def to_object(self, source=M.SOURCE_SCENE):
        return M.RoomObject(Id="scene:%s" % self.uuid if source == M.SOURCE_SCENE else self.name,
                            Kind=self.kind, Pose=self.pose(), Size=M.Vec3(*self.size),
                            Sittable=self.seat_height is not None, Movable=self.movable,
                            Source=source, Label=self.label or "", Locked=False)

    def world_boxes(self):
        """[(8x3 corner array in Unity world)] for every part."""
        pose = self.pose()
        out = []
        for p in self.parts:
            hx, hy, hz = (s / 2 for s in p.size)
            corners = [(p.center[0] + sx * hx, p.center[1] + sy * hy, p.center[2] + sz * hz)
                       for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            out.append(np.array([pose.transform_point(c).tuple() for c in corners]))
        return out


@dataclass
class SynthRoom:
    x_min: float = -2.5
    x_max: float = 2.5
    z_min: float = -2.0
    z_max: float = 2.0
    floor_y: float = 0.0
    wall_height: float = 2.5
    furniture: list = field(default_factory=list)

    def find(self, name):
        return next(f for f in self.furniture if f.name == name)

    def copy(self):
        return copy.deepcopy(self)

    def walls(self):
        """[(name, a, b, inward normal)] wall segments in xz."""
        x0, x1, z0, z1 = self.x_min, self.x_max, self.z_min, self.z_max
        return [("wall_south", (x0, z0), (x1, z0), (0.0, 1.0)),
                ("wall_east", (x1, z0), (x1, z1), (-1.0, 0.0)),
                ("wall_north", (x1, z1), (x0, z1), (0.0, -1.0)),
                ("wall_west", (x0, z1), (x0, z0), (1.0, 0.0))]

    def shell_boxes(self, thickness=0.1):
        """Floor, ceiling and the four walls as 8-corner boxes just outside the room volume."""
        x0, x1, z0, z1, y0 = self.x_min, self.x_max, self.z_min, self.z_max, self.floor_y
        y1 = y0 + self.wall_height
        t = thickness
        spans = [((x0 - t, x1 + t), (y0 - t, y0), (z0 - t, z1 + t)),
                 ((x0 - t, x1 + t), (y1, y1 + t), (z0 - t, z1 + t)),
                 ((x0 - t, x1 + t), (y0, y1), (z0 - t, z0)),
                 ((x0 - t, x1 + t), (y0, y1), (z1, z1 + t)),
                 ((x0 - t, x0), (y0, y1), (z0, z1)),
                 ((x1, x1 + t), (y0, y1), (z0, z1))]
        return [np.array([(xs[i], ys[j], zs[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
                for xs, ys, zs in spans]

    def move(self, name, dx=0.0, dz=0.0, dyaw=0.0):
        """Move a piece of furniture and everything attached to it."""
        for f in self.furniture:
            if f.name == name or f.attached_to == name:
                f.center_xz = (f.center_xz[0] + dx, f.center_xz[1] + dz)
                f.yaw += dyaw

    def gt_objects(self):
        return [f.to_object(M.SOURCE_MANUAL) for f in self.furniture if f.attached_to is None]

    def gt_seats(self):
        seats = []
        for f in self.furniture:
            if f.seat_height is None or f.attached_to is not None:
                continue
            for s in M.generate_seats(f.to_object(M.SOURCE_MANUAL), f.seat_height, f.seat_count):
                s.State = M.SeatState.Blocked.value if f.blocked else M.SeatState.Available.value
                seats.append(s)
        return seats


# corner index = ix*4 + iy*2 + iz (ix, iy, iz in {0, 1}, 0 = negative side)
BOX_TRIANGLES = np.array([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],  # -x, +x
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],  # -y, +y
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],  # -z, +z
], dtype=np.uint32)


def boxes_to_mesh(boxes):
    """Concatenate 8-corner boxes (world_boxes order) into (vertices, triangles)."""
    if not boxes:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint32)
    verts = np.concatenate(boxes)
    tris = np.concatenate([BOX_TRIANGLES + 8 * i for i in range(len(boxes))])
    return verts, tris


def _legs(w, d, h, t=0.05, inset=0.03):
    xs = w / 2 - inset - t / 2
    zs = d / 2 - inset - t / 2
    return [Part((sx * xs, h / 2, sz * zs), (t, h, t)) for sx in (-1, 1) for sz in (-1, 1)]


def table(name, center_xz, yaw, w=1.2, h=0.72, d=0.8, label="TABLE"):
    top = 0.04
    parts = [Part((0, h - top / 2, 0), (w, top, d))] + _legs(w, d, h - top)
    return Furniture(name, M.Kind.Table, center_xz, yaw, (w, h, d), parts, label=label)


def chair(name, center_xz, yaw, w=0.45, d=0.45, seat=0.45, back=0.85):
    slab = 0.04
    parts = [Part((0, seat - slab / 2, 0), (w, slab, d)),
             Part((0, (seat + back) / 2, -d / 2 + 0.02), (w, back - seat, 0.04))]
    parts += _legs(w, d, seat - slab)
    return Furniture(name, M.Kind.Chair, center_xz, yaw, (w, back, d), parts, seat_height=seat)


def couch(name, center_xz, yaw, w=1.8, h=0.8, d=0.9, seat=0.45, arm=0.15, label="COUCH", seat_count=None):
    back_t = 0.2
    parts = [Part((0, seat / 2, back_t / 2), (w - 2 * arm, seat, d - back_t)),
             Part((0, h / 2, -d / 2 + back_t / 2), (w, h, back_t)),
             Part((-w / 2 + arm / 2, 0.3, back_t / 2), (arm, 0.6, d - back_t)),
             Part((w / 2 - arm / 2, 0.3, back_t / 2), (arm, 0.6, d - back_t))]
    kind = M.Kind.Chair if seat_count == 1 else M.Kind.Couch
    return Furniture(name, kind, center_xz, yaw, (w, h, d), parts, label=label, seat_height=seat,
                     seat_count=seat_count, movable=seat_count == 1)


def bed(name, center_xz, yaw, w=2.0, h=0.5, d=1.4):
    return Furniture(name, M.Kind.Bed, center_xz, yaw, (w, h, d), [Part((0, h / 2, 0), (w, h, d))],
                     label="BED", seat_height=h, movable=False)


def block(name, kind, center_xz, yaw, size, label="OTHER", y0=0.0, attached_to=None):
    return Furniture(name, kind, center_xz, yaw, size, [Part((0, y0 + size[1] / 2, 0), size)],
                     label=label, attached_to=attached_to)


def default_room():
    """5 x 4 m room: table + chair (chair carries a clothes pile), couch, bed, coffee table,
    cabinet, armchair. Fronts follow the "away from the nearest wall" rule."""
    room = SynthRoom()
    room.furniture = [
        table("table", (0.3, 0.2), math.pi),
        chair("chair", (0.3, -0.55), 0.0),
        block("clothes", M.Kind.Other, (0.3, -0.52), 0.0, (0.35, 0.15, 0.3), label=None, y0=0.45,
              attached_to="chair"),
        couch("couch", (-0.8, 1.55), math.pi),
        bed("bed", (-1.5, -1.3), 0.0),
        table("coffee_table", (-0.8, 0.5), math.pi, w=1.0, h=0.4, d=0.5),
        block("cabinet", M.Kind.Other, (2.3, 1.0), -math.pi / 2, (0.8, 0.9, 0.4), label="STORAGE"),
        couch("armchair", (1.6, -1.45), 0.0, w=0.85, h=0.85, d=0.85, seat=0.42, arm=0.15, seat_count=1),
    ]
    room.find("chair").blocked = True
    return room
