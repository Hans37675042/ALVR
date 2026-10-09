"""RoomModel v2 fragment: field names, revisions, timestamps."""
import json

from roomd.semantics.synthetic import chair, couch
from sem_helpers import live, make_room, new_sem, run

OBJECT_KEYS = {"Id", "Kind", "Pose", "Size", "Sittable", "Movable", "Source", "Label",
               "Confidence", "State", "Locked", "Tracked", "Revision",
               "CreatedUtc", "UpdatedUtc", "LastSeenUtc"}
SEAT_KEYS = {"Id", "ObjectId", "Pose", "Height", "State", "Source", "Enabled"}
TOP_KEYS = {"Revision", "FrameSource", "FloorY", "FloorRms", "Walls", "Objects", "Seats"}


def test_fragment_fields_follow_contract():
    room = make_room()
    room.place("k", couch(0.0, 0.5, 0.0))
    sem = new_sem()
    out, _ = run(sem, room, 3, 1_791_000_000.0)
    assert set(out) == TOP_KEYS
    assert out["FrameSource"] == "Stage"
    json.dumps(out)  # plain JSON types only
    o = live(out)[0]
    assert set(o) == OBJECT_KEYS
    assert set(o["Pose"]) == {"Position", "Rotation"}
    assert set(o["Pose"]["Position"]) == {"X", "Y", "Z"}
    assert set(o["Pose"]["Rotation"]) == {"X", "Y", "Z", "W"}
    assert set(o["Size"]) == {"X", "Y", "Z"}
    assert o["Locked"] is False and o["Tracked"] is False
    assert o["CreatedUtc"].endswith("Z") and "T" in o["CreatedUtc"]
    assert o["Pose"]["Position"]["Y"] == out["FloorY"]  # bottom-face centre on the floor
    assert len(out["Seats"]) == 3
    for s in out["Seats"]:
        assert set(s) == SEAT_KEYS
    for w in out["Walls"]:
        assert set(w) == {"a", "b"} and len(w["a"]) == 2 and len(w["b"]) == 2


def test_revision_only_moves_on_change():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 100.0)
    rev = out["Revision"]
    assert rev >= 1
    out, t = run(sem, room, 4, t)
    assert out["Revision"] == rev
    room.place("c", chair(-0.5, 0.5, 0.0))
    out, t = run(sem, room, 4, t)
    assert out["Revision"] > rev
    o = live(out)[0]
    assert o["Revision"] >= 2
    assert o["UpdatedUtc"] > o["CreatedUtc"]
    assert o["LastSeenUtc"] >= o["UpdatedUtc"]


def test_revision_is_monotonic_and_starts_positive():
    room = make_room()
    sem = new_sem()
    revs = []
    for i in range(5):
        if i == 2:
            room.place("c", chair(0.0, 0.0, 0.0))
        revs.append(sem.update(room, float(i))["Revision"])
    assert revs == sorted(revs) and revs[0] >= 1
