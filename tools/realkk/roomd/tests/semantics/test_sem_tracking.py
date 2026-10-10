"""Cross-frame tracking: id stability, hysteresis, Missing/Removed, occlusion."""
import numpy as np

from roomd.semantics.synthetic import chair, table
from sem_helpers import by_id, live, make_room, near, new_sem, pos, run, seats_of, yaw, yaw_close


def _single(out):
    objs = live(out)
    assert len(objs) == 1, [(o["Id"], o["Kind"]) for o in objs]
    return objs[0]


def test_new_object_needs_three_consecutive_observations():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    assert live(sem.update(room, 0.0)) == []
    assert live(sem.update(room, 0.5)) == []
    o = _single(sem.update(room, 1.0))
    assert o["Id"].startswith("fused:") and o["Kind"] == "Chair"
    assert o["Source"] == "Fused" and o["State"] == "Present"


def test_flicker_does_not_create_object():
    room = make_room()
    sem = new_sem()
    t = 0.0
    for i in range(6):
        if i % 2 == 0:
            room.place("c", chair(0.5, 0.5, 0.0))
        else:
            room.remove("c")
        out = sem.update(room, t)
        t += 0.5
    assert live(out) == []


def test_noise_keeps_id_and_published_pose():
    rng = np.random.default_rng(5)
    room = make_room()
    room.place("c", chair(0.5, 0.5, 30.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    for _ in range(12):
        dx, dz = rng.uniform(-0.02, 0.02, 2)
        dyaw = rng.uniform(-3.0, 3.0)
        room.place("c", chair(0.5 + dx, 0.5 + dz, 30.0 + dyaw))
        out, t = run(sem, room, 1, t)
        o = _single(out)
        assert o["Id"] == oid and o["State"] == "Present"
    assert near(pos(o), (0.5, 0.5), 0.05)
    assert yaw_close(yaw(o), 30.0, 8.0)


def test_move_one_metre_keeps_id_and_goes_through_moving():
    room = make_room()
    room.place("c", chair(-1.0, 0.0, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.place("c", chair(0.0, 0.0, 0.0))
    states = []
    for _ in range(4):
        out, t = run(sem, room, 1, t)
        o = _single(out)
        assert o["Id"] == oid
        states.append(o["State"])
    assert "Moving" in states
    assert states[-1] == "Present"
    assert near(pos(o), (0.0, 0.0), 0.05)
    assert all(s["State"] == "Available" for s in seats_of(out, oid))


def test_small_shift_below_threshold_is_not_published():
    room = make_room(noise=0.0)
    room.place("t", table(0.0, 0.0, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    o0 = _single(out)
    room.place("t", table(0.03, 0.0, 0.0))
    out, t = run(sem, room, 4, t)
    o1 = _single(out)
    assert o1["Id"] == o0["Id"] and o1["Revision"] == o0["Revision"]
    assert o1["Pose"] == o0["Pose"]


def test_rotate_ninety_degrees_keeps_id():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.place("c", chair(0.5, 0.5, 90.0))
    out, t = run(sem, room, 4, t)
    o = _single(out)
    assert o["Id"] == oid
    assert yaw_close(yaw(o), 90.0, 8.0)


def test_swapping_identical_chairs_keeps_ids_in_place():
    room = make_room()
    room.place("a", chair(-0.8, 0.0, 0.0))
    room.place("b", chair(0.8, 0.0, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    before = {o["Id"]: (pos(o), o["Revision"]) for o in live(out)}
    assert len(before) == 2
    room.place("a", chair(0.8, 0.0, 0.0))
    room.place("b", chair(-0.8, 0.0, 0.0))
    for _ in range(4):
        out, t = run(sem, room, 1, t)
        assert all(o["State"] == "Present" for o in live(out))
    after = {o["Id"]: (pos(o), o["Revision"]) for o in live(out)}
    assert after == before


def test_swapping_distinguishable_chairs_ids_follow_chairs():
    # seat heights differ by more than gate_surface_dh (0.10)
    room = make_room()
    room.place("lo", chair(-0.8, 0.0, 0.0, seat_h=0.40))
    room.place("hi", chair(0.8, 0.0, 0.0, seat_h=0.55))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    ids = {round(pos(o)[0]): o["Id"] for o in live(out)}
    lo_id, hi_id = ids[-1], ids[1]
    room.place("lo", chair(0.8, 0.0, 0.0, seat_h=0.40))
    room.place("hi", chair(-0.8, 0.0, 0.0, seat_h=0.55))
    out, t = run(sem, room, 4, t)
    objs = by_id(out)
    assert near(pos(objs[lo_id]), (0.8, 0.0), 0.06)
    assert near(pos(objs[hi_id]), (-0.8, 0.0), 0.06)
    assert len(live(out)) == 2


def test_removed_object_goes_missing_then_removed_with_tombstone():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.remove("c")
    out, t = run(sem, room, 2, t)
    assert by_id(out)[oid]["State"] == "Missing"
    assert all(s["State"] == "Missing" for s in seats_of(out, oid))
    out, t = run(sem, room, 3, t)  # 1.5 s later: still Missing
    assert by_id(out)[oid]["State"] == "Missing"
    out, t = run(sem, room, 8, t)  # > 5 s of free-space evidence
    o = by_id(out)[oid]
    assert o["State"] == "Removed"
    assert all(s["State"] == "Removed" for s in seats_of(out, oid))
    assert live(out) == []


def test_reappear_while_missing_keeps_id():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.remove("c")
    out, t = run(sem, room, 3, t)
    assert by_id(out)[oid]["State"] == "Missing"
    room.place("c", chair(0.5, 0.5, 0.0))
    out, t = run(sem, room, 1, t)
    o = _single(out)
    assert o["Id"] == oid and o["State"] == "Present"


def test_reappear_after_removed_revives_same_id():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.remove("c")
    out, t = run(sem, room, 16, t)
    assert by_id(out)[oid]["State"] == "Removed"
    room.place("c", chair(0.6, 0.4, 0.0))
    out, t = run(sem, room, 3, t)
    o = _single(out)
    assert o["Id"] == oid and o["State"] == "Present"
    assert near(pos(o), (0.6, 0.4), 0.05)


def test_occluded_object_is_not_declared_missing():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    from roomd.semantics import Obb
    room.remove("c")
    room.set_hidden([Obb(0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 2.0)])
    out, t = run(sem, room, 20, t)
    assert by_id(out)[oid]["State"] == "Present"


def test_rejected_object_is_kept_and_not_redetected():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    sem.reject(oid)
    out, t = run(sem, room, 6, t)
    assert by_id(out)[oid]["State"] == "Rejected"
    assert live(out) == []


def test_removed_object_revives_on_first_observation():
    # live 2026-10-10: the office chair stayed Removed while back in the room
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.remove("c")
    out, t = run(sem, room, 16, t)
    assert by_id(out)[oid]["State"] == "Removed"
    room.place("c", chair(0.7, 0.4, 20.0))
    out, t = run(sem, room, 1, t)
    o = _single(out)
    assert o["Id"] == oid and o["State"] == "Present"
    assert [s["State"] for s in seats_of(out, oid)] == ["Available"]


def test_seat_height_mismatch_is_not_free_space_evidence():
    # real chairs read 0.375-0.46 m across views: a probe slab at the published seat height can
    # be free while the seat (seen as something else, e.g. no backrest) still stands there
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0, seat_h=0.52))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    from roomd.semantics.synthetic import stool
    room.place("c", stool(0.5, 0.5, 0.0, h=0.44, w=0.45))
    out, t = run(sem, room, 20, t)
    assert by_id(out)[oid]["State"] == "Present"


def test_chair_the_user_sits_on_never_goes_missing():
    # the fusion body mask hides a chair the user sits on; its slab can read free
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.remove("c")  # worst case: the map shows only free space there
    out, t = run(sem, room, 20, t, user_head=(0.5, 1.2, 0.45))
    o = by_id(out)[oid]
    assert o["State"] == "Present"
    assert [s["State"] for s in seats_of(out, oid)] == ["OccupiedByUser"]


def test_user_standing_on_the_spot_gives_no_free_space_evidence():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sem = new_sem()
    out, t = run(sem, room, 3, 0.0)
    oid = _single(out)["Id"]
    room.remove("c")
    out, t = run(sem, room, 20, t, user_head=(0.55, 1.65, 0.5))
    assert by_id(out)[oid]["State"] == "Present"
    out, t = run(sem, room, 2, t, user_head=(-1.5, 1.65, -1.5))  # walks away: now it is gone
    assert by_id(out)[oid]["State"] == "Missing"
