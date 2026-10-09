"""Seat states: clothes pile -> Blocked, user sitting -> OccupiedByUser."""
from roomd.semantics.synthetic import chair, clothes_pile, couch, user_torso
from sem_helpers import by_id, live, make_room, new_sem, run, seats_of


def _setup(boxes):
    room = make_room()
    room.place("c", boxes)
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    objs = live(out)
    assert len(objs) == 1
    return room, sem, objs[0]["Id"], t


def test_clothes_on_chair_blocks_seat_without_losing_chair():
    room, sem, oid, t = _setup(chair(0.5, 0.5, 0.0))
    room.place("pile", clothes_pile(0.5, 0.52, base_y=0.45, size=0.35, height=0.2))
    out, t = run(sem, room, 4, t)
    o = by_id(out)[oid]
    assert o["State"] == "Present"
    assert [s["State"] for s in seats_of(out, oid)] == ["Blocked"]
    assert len(live(out)) == 1
    room.remove("pile")
    out, t = run(sem, room, 1, t)
    assert [s["State"] for s in seats_of(out, oid)] == ["Available"]


def test_pile_on_one_couch_seat_blocks_only_that_seat():
    room, sem, oid, t = _setup(couch(0.0, 0.5, 0.0, length=1.8, depth=0.9))
    room.place("pile", clothes_pile(-0.6, 0.55, base_y=0.45, size=0.4, height=0.2))
    out, t = run(sem, room, 2, t)
    states = {s["Id"]: s["State"] for s in seats_of(out, oid)}
    assert states == {f"{oid}#0": "Blocked", f"{oid}#1": "Available", f"{oid}#2": "Available"}


def test_user_sitting_marks_seat_occupied():
    room, sem, oid, t = _setup(chair(0.5, 0.5, 0.0))
    room.place("user", user_torso(0.5, 0.5, 0.0, seat_h=0.45))
    head = (0.5, 1.2, 0.42)
    out, t = run(sem, room, 4, t, user_head=head)
    o = by_id(out)[oid]
    assert o["State"] == "Present"
    assert [s["State"] for s in seats_of(out, oid)] == ["OccupiedByUser"]
    assert len(live(out)) == 1
    room.remove("user")
    out, t = run(sem, room, 1, t, user_head=(0.5, 1.65, 1.0))  # standing up and away
    assert [s["State"] for s in seats_of(out, oid)] == ["Available"]


def test_standing_user_next_to_chair_does_not_occupy():
    room, sem, oid, t = _setup(chair(0.5, 0.5, 0.0))
    out, t = run(sem, room, 1, t, user_head=(0.5, 1.65, 0.5))
    assert [s["State"] for s in seats_of(out, oid)] == ["Available"]
