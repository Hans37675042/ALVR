import json
import math

import pytest

from roomd import model as M

V1_JSON = """
{ "Version":1, "Id":"living", "FloorY":0.01,
  "Objects":[{"Id":"couch","Kind":"Couch",
              "Pose":{"Position":{"X":0,"Y":0,"Z":1},"Rotation":{"X":0,"Y":0,"Z":0,"W":1}},
              "Size":{"X":1.8,"Y":0.8,"Z":0.9},"Sittable":true,"Movable":false},
             {"id":"lamp","kind":"Lamp","pose":null,"size":{"x":0.3,"y":1.5,"z":0.3}}],
  "Seats":[{"Id":"couch#0","ObjectId":"couch",
            "Pose":{"Position":{"X":-0.6,"Y":0.45,"Z":1},"Rotation":{"X":0,"Y":0,"Z":0,"W":1}},
            "Height":0.45}] }
"""


def test_v1_migrates_to_v2_defaults():
    room = M.room_from_json(V1_JSON)
    assert room.Version == 2 and room.Id == "living" and room.FloorY == pytest.approx(0.01)
    assert room.FrameSource == "Manual3Point" and room.Revision == 0 and room.Walls == []
    couch = room.find("couch")
    assert couch.Kind == M.Kind.Couch and couch.Size.X == pytest.approx(1.8)
    assert couch.Source == "Manual" and couch.Confidence == 1.0 and couch.Locked is True
    assert couch.State == "Present" and couch.Tracked is False and couch.Label == ""
    lamp = room.find("lamp")  # keys are case-insensitive, unknown kind -> Other
    assert lamp.Kind == M.Kind.Other and lamp.Size.Y == pytest.approx(1.5)
    seat = room.Seats[0]
    assert seat.State == "Available" and seat.Source == "Manual" and seat.Enabled is True
    assert seat.Pose.Position.X == pytest.approx(-0.6)


def test_v2_json_field_names_follow_contract():
    obj = M.RoomObject(Id="scene:abc", Kind=M.Kind.Table, Size=M.Vec3(1.2, 0.72, 0.8),
                       Source="SceneApi", Label="TABLE", Confidence=0.9, State="Present",
                       Locked=False, Tracked=False, Revision=3, CreatedUtc="2026-10-09T00:00:00Z",
                       UpdatedUtc="2026-10-09T00:00:01Z", LastSeenUtc="2026-10-09T00:00:02Z")
    room = M.RoomModel(Id="stage", FloorY=0.0, Revision=7, FrameSource="Stage", FloorRms=0.004,
                       Walls=[M.Wall((0.0, 0.0), (3.0, 0.0))], Objects=[obj])
    d = json.loads(M.room_to_json(room))
    assert set(d) == {"Version", "Id", "FloorY", "Objects", "Seats", "Revision", "FrameSource",
                      "FloorRms", "Walls"}
    assert d["Version"] == 2 and d["Walls"] == [{"a": [0.0, 0.0], "b": [3.0, 0.0]}]
    o = d["Objects"][0]
    assert set(o) == {"Id", "Kind", "Pose", "Size", "Sittable", "Movable", "Source", "Label",
                      "Confidence", "State", "Locked", "Tracked", "Revision", "CreatedUtc",
                      "UpdatedUtc", "LastSeenUtc"}
    assert o["Kind"] == "Table" and o["Pose"]["Rotation"] == {"X": 0.0, "Y": 0.0, "Z": 0.0, "W": 1.0}
    assert o["Size"] == {"X": 1.2, "Y": 0.72, "Z": 0.8}
    seat = M.SeatPoint(Id="a#0", ObjectId="a", Height=0.45, State="Blocked", Source="Fused", Enabled=False)
    s = M.seat_to_dict(seat)
    assert set(s) == {"Id", "ObjectId", "Pose", "Height", "State", "Source", "Enabled"}
    back = M.room_from_json(M.room_to_json(room))
    assert back == room


def test_newer_version_is_rejected():
    with pytest.raises(ValueError):
        M.room_from_json('{"Version":3,"Id":"x"}')
    with pytest.raises(ValueError):
        M.room_from_json('{"Id":"x"}')


def test_enums_cover_contract():
    assert [s.value for s in M.ObjectState] == ["Present", "Moving", "Missing", "Removed", "Rejected"]
    assert [s.value for s in M.SeatState] == ["Available", "Moving", "Blocked", "OccupiedByUser",
                                              "Missing", "Removed"]
    assert [k.value for k in M.Kind] == ["Floor", "Wall", "Couch", "Chair", "Table", "Bed", "Other"]


def test_seat_generator_matches_core_rule():
    # 1.8 m along X -> 3 seats centred in 0.6 m segments, facing the object front
    couch = M.RoomObject(Id="couch", Kind=M.Kind.Couch, Sittable=True, Size=M.Vec3(1.8, 0.8, 0.9),
                         Pose=M.Pose(M.Vec3(1.0, 0.0, 2.0), M.Quat.from_yaw(math.pi / 2)))
    seats = M.generate_seats(couch, 0.45)
    assert [s.Id for s in seats] == ["couch#0", "couch#1", "couch#2"]
    # yaw 90 deg: local +X -> world -Z in Unity (left-handed)
    assert (seats[0].Pose.Position.X, seats[0].Pose.Position.Y, seats[0].Pose.Position.Z) == \
        pytest.approx((1.0, 0.45, 2.6))
    assert seats[1].Pose.Position.Z == pytest.approx(2.0)
    assert seats[0].Pose.Rotation == couch.Pose.Rotation and seats[0].Height == 0.45
    # longer along Z -> spread along Z; under 0.6 m -> still one seat
    bench = M.RoomObject(Id="b", Kind=M.Kind.Couch, Sittable=True, Size=M.Vec3(0.5, 0.4, 1.25))
    assert [round(s.Pose.Position.Z, 4) for s in M.generate_seats(bench, 0.4)] == [-0.3125, 0.3125]
    stool = M.RoomObject(Id="s", Kind=M.Kind.Chair, Sittable=True, Size=M.Vec3(0.4, 0.45, 0.4))
    assert len(M.generate_seats(stool, 0.45)) == 1
    # count override (MRUK single-seat couch) and non-sittable objects
    assert len(M.generate_seats(couch, 0.45, count=1)) == 1
    assert M.generate_seats(M.RoomObject(Id="t", Kind=M.Kind.Table, Size=M.Vec3(1, 1, 1)), 0.7) == []
    # exactly 1.2 m -> 2 seats (floor with 1e-6 tolerance)
    two = M.RoomObject(Id="c", Kind=M.Kind.Couch, Sittable=True, Size=M.Vec3(1.2, 0.8, 0.8))
    assert len(M.generate_seats(two, 0.45)) == 2


def test_content_key_ignores_revision_and_timestamps():
    a = M.RoomModel(Id="x", Objects=[M.RoomObject(Id="o", UpdatedUtc="t1", LastSeenUtc="t1", Revision=1)])
    b = M.RoomModel(Id="x", Revision=5, Objects=[M.RoomObject(Id="o", UpdatedUtc="t2", LastSeenUtc="t2",
                                                              Revision=2)])
    assert M.content_key(a) == M.content_key(b)
    b.Objects[0].Size = M.Vec3(1, 0, 0)
    assert M.content_key(a) != M.content_key(b)
