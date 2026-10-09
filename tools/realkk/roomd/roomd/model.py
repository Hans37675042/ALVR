"""RoomModel v2 (CONTRACT-roomd.md) and its JSON form.

Field names are the C# field names of RealKK.Core.Room so that the JSON matches
RoomSerializer: written with exact names, read case-insensitively, unknown keys ignored.
Coordinates are Unity left-handed stage space (+Y up, metres). An object's Pose is the
centre of its box's bottom face; local +Z is its front.
"""

import json
import math
from dataclasses import dataclass, field
from enum import Enum

CURRENT_VERSION = 2
SEAT_SPACING = 0.6


class Kind(str, Enum):
    Floor = "Floor"
    Wall = "Wall"
    Couch = "Couch"
    Chair = "Chair"
    Table = "Table"
    Bed = "Bed"
    Other = "Other"


class ObjectState(str, Enum):
    Present = "Present"
    Moving = "Moving"
    Missing = "Missing"
    Removed = "Removed"
    Rejected = "Rejected"


class SeatState(str, Enum):
    Available = "Available"
    Moving = "Moving"
    Blocked = "Blocked"
    OccupiedByUser = "OccupiedByUser"
    Missing = "Missing"
    Removed = "Removed"


SOURCE_MANUAL = "Manual"
SOURCE_SCENE = "SceneApi"
SOURCE_FUSED = "Fused"
FRAME_MANUAL = "Manual3Point"
FRAME_STAGE = "Stage"


@dataclass
class Vec3:
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0

    def tuple(self):
        return (self.X, self.Y, self.Z)


@dataclass
class Quat:
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    W: float = 1.0

    def tuple(self):
        return (self.X, self.Y, self.Z, self.W)

    @classmethod
    def from_yaw(cls, yaw):
        """Rotation about +Y; local +Z ends up along (sin yaw, 0, cos yaw)."""
        return cls(0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2))

    def rotate(self, v):
        """Rotate vector v (3-tuple)."""
        x, y, z, w = self.X, self.Y, self.Z, self.W
        vx, vy, vz = v
        # t = 2 * cross(q.xyz, v); v' = v + w*t + cross(q.xyz, t)
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        return (vx + w * tx + (y * tz - z * ty),
                vy + w * ty + (z * tx - x * tz),
                vz + w * tz + (x * ty - y * tx))


@dataclass
class Pose:
    Position: Vec3 = field(default_factory=Vec3)
    Rotation: Quat = field(default_factory=Quat)

    def transform_point(self, local):
        r = self.Rotation.rotate(local)
        p = self.Position
        return Vec3(p.X + r[0], p.Y + r[1], p.Z + r[2])


@dataclass
class RoomObject:
    Id: str = ""
    Kind: Kind = Kind.Other
    Pose: Pose = field(default_factory=Pose)
    Size: Vec3 = field(default_factory=Vec3)
    Sittable: bool = False
    Movable: bool = False
    # v2
    Source: str = SOURCE_MANUAL
    Label: str = ""
    Confidence: float = 1.0
    State: str = ObjectState.Present.value
    Locked: bool = True
    Tracked: bool = False
    Revision: int = 0
    CreatedUtc: str = None
    UpdatedUtc: str = None
    LastSeenUtc: str = None


@dataclass
class SeatPoint:
    Id: str = ""
    ObjectId: str = ""
    Pose: Pose = field(default_factory=Pose)
    Height: float = 0.0
    # v2
    State: str = SeatState.Available.value
    Source: str = SOURCE_MANUAL
    Enabled: bool = True


@dataclass
class Wall:
    a: tuple  # (x, z)
    b: tuple  # (x, z)


@dataclass
class RoomModel:
    Id: str = ""
    Version: int = CURRENT_VERSION
    FloorY: float = 0.0
    Objects: list = field(default_factory=list)
    Seats: list = field(default_factory=list)
    # v2
    Revision: int = 0
    FrameSource: str = FRAME_MANUAL
    FloorRms: float = 0.0
    Walls: list = field(default_factory=list)

    def find(self, object_id):
        return next((o for o in self.Objects if o.Id == object_id), None)

    def seats_of(self, object_id):
        return [s for s in self.Seats if s.ObjectId == object_id]


# --- seats (RealKK.Core.Room.SeatGenerator) ---

def generate_seats(obj, seat_height, count=None, source=None):
    """Seats along the longer horizontal side: one per full 0.6 m (at least one), each centred
    in its equal segment at the depth centre; pose = object rotation (faces local +Z).
    `count` overrides the per-0.6 m count (MRUK treats near-square couches as one seat)."""
    if not obj.Sittable:
        return []
    along_z = obj.Size.Z > obj.Size.X
    length = abs(obj.Size.Z if along_z else obj.Size.X)
    if count is None:
        count = max(1, int(math.floor(length / SEAT_SPACING + 1e-6)))
    seats = []
    for i in range(count):
        offset = -length / 2.0 + (i + 0.5) * length / count
        local = (0.0, seat_height, offset) if along_z else (offset, seat_height, 0.0)
        seats.append(SeatPoint(Id="%s#%d" % (obj.Id or "obj", i), ObjectId=obj.Id,
                               Pose=Pose(obj.Pose.transform_point(local), Quat(*obj.Pose.Rotation.tuple())),
                               Height=seat_height, Source=source or obj.Source))
    return seats


# --- JSON ---

def _vec_dict(v):
    return {"X": float(v.X), "Y": float(v.Y), "Z": float(v.Z)}


def _pose_dict(p):
    r = p.Rotation
    return {"Position": _vec_dict(p.Position),
            "Rotation": {"X": float(r.X), "Y": float(r.Y), "Z": float(r.Z), "W": float(r.W)}}


def object_to_dict(o):
    return {
        "Id": o.Id, "Kind": Kind(o.Kind).value, "Pose": _pose_dict(o.Pose), "Size": _vec_dict(o.Size),
        "Sittable": bool(o.Sittable), "Movable": bool(o.Movable),
        "Source": o.Source, "Label": o.Label or "", "Confidence": float(o.Confidence),
        "State": _enum_value(o.State), "Locked": bool(o.Locked), "Tracked": bool(o.Tracked),
        "Revision": int(o.Revision), "CreatedUtc": o.CreatedUtc, "UpdatedUtc": o.UpdatedUtc,
        "LastSeenUtc": o.LastSeenUtc,
    }


def seat_to_dict(s):
    return {"Id": s.Id, "ObjectId": s.ObjectId, "Pose": _pose_dict(s.Pose), "Height": float(s.Height),
            "State": _enum_value(s.State), "Source": s.Source, "Enabled": bool(s.Enabled)}


def room_to_dict(room):
    return {
        "Version": CURRENT_VERSION, "Id": room.Id, "FloorY": float(room.FloorY),
        "Revision": int(room.Revision), "FrameSource": room.FrameSource, "FloorRms": float(room.FloorRms),
        "Walls": [{"a": [float(w.a[0]), float(w.a[1])], "b": [float(w.b[0]), float(w.b[1])]}
                  for w in room.Walls],
        "Objects": [object_to_dict(o) for o in room.Objects if o is not None],
        "Seats": [seat_to_dict(s) for s in room.Seats if s is not None],
    }


def room_to_json(room, indent=None):
    return json.dumps(room_to_dict(room), ensure_ascii=False, indent=indent)


def _enum_value(v):
    return v.value if isinstance(v, Enum) else v


class _CI:
    """Case-insensitive view of a JSON object."""

    def __init__(self, d, what):
        if not isinstance(d, dict):
            raise ValueError("%s: expected a JSON object" % what)
        self._d = {k.lower(): v for k, v in d.items()}

    def get(self, key, default=None):
        v = self._d.get(key.lower())
        return default if v is None else v


def _vec(d, what):
    c = _CI(d, what)
    return Vec3(float(c.get("X", 0.0)), float(c.get("Y", 0.0)), float(c.get("Z", 0.0)))


def _pose(d, what):
    c = _CI(d, what)
    pose = Pose()
    if c.get("Position") is not None:
        pose.Position = _vec(c.get("Position"), what + ".Position")
    if c.get("Rotation") is not None:
        r = _CI(c.get("Rotation"), what + ".Rotation")
        pose.Rotation = Quat(float(r.get("X", 0.0)), float(r.get("Y", 0.0)), float(r.get("Z", 0.0)),
                             float(r.get("W", 1.0)))
    return pose


def _kind(value):
    if value is None:
        return Kind.Other
    if isinstance(value, str):
        for k in Kind:
            if k.value.lower() == value.lower():
                return k
        return Kind.Other
    kinds = list(Kind)
    n = int(value)
    return kinds[n] if 0 <= n < len(kinds) else Kind.Other


def _enum_name(value, enum, default):
    if isinstance(value, str):
        for e in enum:
            if e.value.lower() == value.lower():
                return e.value
    return default


def room_from_dict(root):
    """Parse a v1 or v2 room; v1 is migrated to v2 defaults (Manual, Locked, Present, Available)."""
    c = _CI(root, "room")
    version = c.get("Version")
    if version is None:
        raise ValueError('room: missing "Version".')
    version = float(version)
    if version != math.floor(version) or version < 1:
        raise ValueError("room: invalid Version %s." % version)
    if version > CURRENT_VERSION:
        raise ValueError("room: Version %d is newer than supported (%d)." % (version, CURRENT_VERSION))
    v2 = version >= 2
    room = RoomModel(Id=c.get("Id"), FloorY=float(c.get("FloorY", 0.0)))
    if v2:
        room.Revision = int(c.get("Revision", 0))
        room.FrameSource = c.get("FrameSource", FRAME_MANUAL)
        room.FloorRms = float(c.get("FloorRms", 0.0))
        for i, w in enumerate(c.get("Walls", [])):
            wc = _CI(w, "Walls[%d]" % i)
            room.Walls.append(Wall(tuple(float(x) for x in wc.get("a")), tuple(float(x) for x in wc.get("b"))))
    for i, item in enumerate(c.get("Objects", [])):
        what = "Objects[%d]" % i
        d = _CI(item, what)
        o = RoomObject(Id=d.get("Id"), Kind=_kind(d.get("Kind")), Sittable=bool(d.get("Sittable", False)),
                       Movable=bool(d.get("Movable", False)))
        if d.get("Pose") is not None:
            o.Pose = _pose(d.get("Pose"), what + ".Pose")
        if d.get("Size") is not None:
            o.Size = _vec(d.get("Size"), what + ".Size")
        if v2:
            o.Source = d.get("Source", SOURCE_MANUAL)
            o.Label = d.get("Label", "")
            o.Confidence = float(d.get("Confidence", 1.0))
            o.State = _enum_name(d.get("State"), ObjectState, ObjectState.Present.value)
            o.Locked = bool(d.get("Locked", o.Source == SOURCE_MANUAL))
            o.Tracked = bool(d.get("Tracked", False))
            o.Revision = int(d.get("Revision", 0))
            o.CreatedUtc = d.get("CreatedUtc")
            o.UpdatedUtc = d.get("UpdatedUtc")
            o.LastSeenUtc = d.get("LastSeenUtc")
        room.Objects.append(o)
    for i, item in enumerate(c.get("Seats", [])):
        what = "Seats[%d]" % i
        d = _CI(item, what)
        s = SeatPoint(Id=d.get("Id"), ObjectId=d.get("ObjectId"), Height=float(d.get("Height", 0.0)))
        if d.get("Pose") is not None:
            s.Pose = _pose(d.get("Pose"), what + ".Pose")
        if v2:
            s.State = _enum_name(d.get("State"), SeatState, SeatState.Available.value)
            s.Source = d.get("Source", SOURCE_MANUAL)
            s.Enabled = bool(d.get("Enabled", True))
        room.Seats.append(s)
    return room


def room_from_json(text):
    return room_from_dict(json.loads(text))


_VOLATILE = {"Revision", "UpdatedUtc", "LastSeenUtc"}


def content_key(room):
    """Canonical JSON of the model without revision counters and seen/updated timestamps:
    equal keys mean nothing a consumer acts on has changed."""
    d = room_to_dict(room)
    d.pop("Revision")
    for o in d["Objects"]:
        for k in _VOLATILE:
            o.pop(k, None)
    return json.dumps(d, sort_keys=True)
