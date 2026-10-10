"""Kind lock (PM 2026-10-10): real furniture is moved, not added or removed. A track's kind
follows the classifier until it is stable, then stays fixed; pose and state keep updating.
A stable (confirmed) object is never removed automatically; re-classification is on request."""
from roomd.semantics import SemanticsParams
from roomd.semantics.synthetic import chair, coffee_table, stool
from roomd.semantics.tracker import iso_utc
from sem_helpers import by_id, live, make_room, near, new_sem, pos, run, seats_of

P = SemanticsParams()
CREATE = P.new_confirm                       # frames until the track exists
LOCK = P.new_confirm + P.kind_lock_obs - 1   # frames until its kind is locked


def _chair_track(frames, x=0.5, z=0.5, **params):
    room = make_room()
    room.place("c", chair(x, z, 0.0))
    sem = new_sem(**params)
    out, t = run(sem, room, frames, 0.0)
    chairs = live(out, "Chair")
    assert len(chairs) == 1, [(o["Id"], o["Kind"]) for o in live(out)]
    return room, sem, chairs[0]["Id"], out, t


def _locked_chair(x=0.5, z=0.5):
    room, sem, oid, out, t = _chair_track(LOCK, x, z)
    assert sem.tracks[oid].kind_locked
    return room, sem, oid, t


# ---------------------------------------------------------------- when a kind locks
def test_kind_locks_after_consecutive_same_kind_observations():
    room, sem, oid, out, t = _chair_track(CREATE)
    assert not sem.tracks[oid].kind_locked
    out, t = run(sem, room, LOCK - CREATE - 1, t)
    assert not sem.tracks[oid].kind_locked
    out, t = run(sem, room, 1, t)
    assert sem.tracks[oid].kind_locked


def test_kind_locks_after_seconds_of_the_same_kind():
    # created at t = 1.0 (3rd frame, dt 0.5); the time rule locks at t >= 1.0 + 4.0
    room, sem, oid, out, t = _chair_track(CREATE, kind_lock_obs=100, kind_lock_s=4.0)
    out, t = run(sem, room, 7, t)       # last update at t = 4.5
    assert not sem.tracks[oid].kind_locked
    out, t = run(sem, room, 1, t)       # t = 5.0
    assert sem.tracks[oid].kind_locked


def test_other_never_locks():
    # geometry-only clutter is not furniture: it keeps following the classifier
    room = make_room()
    room.place("s", stool(0.5, 0.5, 0.0, h=0.44))
    sem = new_sem()
    out, t = run(sem, room, 20, 0.0)
    others = live(out, "Other")
    assert len(others) == 1
    assert not sem.tracks[others[0]["Id"]].kind_locked


# ---------------------------------------------------------------- before / after the lock
def test_unstable_kind_follows_the_classifier():
    room, sem, oid, out, t = _chair_track(CREATE)
    room.place("c", stool(0.5, 0.5, 0.0, h=0.44))   # now reads as Other
    out, t = run(sem, room, 1, t)
    assert by_id(out)[oid]["Kind"] == "Chair"               # one reading is not enough
    out, t = run(sem, room, 1, t)
    o = by_id(out)[oid]
    assert o["Kind"] == "Other" and o["State"] == "Present"
    assert seats_of(out, oid) == []
    assert [x["Id"] for x in live(out)] == [oid]


def test_locked_chair_keeps_its_kind_when_read_differently():
    # e.g. someone leaning on it or a bag on the seat: same place, another classification
    room, sem, oid, t = _locked_chair()
    room.place("c", stool(0.5, 0.5, 0.0, h=0.44))
    out, t = run(sem, room, 6, t)
    o = by_id(out)[oid]
    assert o["Kind"] == "Chair" and o["State"] == "Present"
    assert o["LastSeenUtc"] == iso_utc(t - 0.5)            # still matched, not just kept
    assert len(seats_of(out, oid)) == 1
    assert [x["Id"] for x in live(out)] == [oid]


def test_locked_chair_still_follows_a_move():
    room, sem, oid, t = _locked_chair(-1.0, 0.0)
    room.place("c", chair(0.0, 0.0, 0.0))
    out, t = run(sem, room, 4, t)
    o = by_id(out)[oid]
    assert o["State"] == "Present" and near(pos(o), (0.0, 0.0), 0.05)
    assert len(live(out)) == 1


def test_locked_size_changes_only_within_tolerance():
    room, sem, oid, t = _locked_chair(-1.0, 0.0)
    locked = max(sem.tracks[oid].lock_size[0], sem.tracks[oid].lock_size[2])
    room.place("c", chair(0.0, 0.0, 0.0, width=0.80))      # re-measured much wider after a move
    out, t = run(sem, room, 4, t)
    o = by_id(out)[oid]
    assert o["Kind"] == "Chair" and near(pos(o), (0.0, 0.0), 0.06)
    assert max(o["Size"]["X"], o["Size"]["Z"]) <= locked * (1 + P.kind_size_tol) + 0.01


# ---------------------------------------------------------------- no automatic deletion
def test_locked_chair_goes_missing_and_is_never_removed():
    room, sem, oid, t = _locked_chair()
    room.remove("c")
    out, t = run(sem, room, 40, t)                          # 20 s, remove_after_s = 5
    o = by_id(out)[oid]
    assert o["State"] == "Missing"
    assert [s["State"] for s in seats_of(out, oid)] == ["Missing"]


def test_locked_missing_chair_resumes_where_it_is_seen_again_far_away():
    room, sem, oid, t = _locked_chair(-1.8, -1.8)
    room.remove("c")
    out, t = run(sem, room, 4, t)
    assert by_id(out)[oid]["State"] == "Missing"
    room.place("c", chair(1.8, 1.8, 90.0))                  # carried 5 m while unseen
    out, t = run(sem, room, 1, t)
    assert by_id(out)[oid]["State"] == "Missing"            # one far reading is not enough
    out, t = run(sem, room, 3, t)
    o = by_id(out)[oid]
    assert o["State"] == "Present" and near(pos(o), (1.8, 1.8), 0.06)
    assert [x["Id"] for x in live(out)] == [oid]


def test_never_stable_track_still_expires():
    room, sem, oid, out, t = _chair_track(10, kind_lock_obs=50, kind_lock_s=1e9)
    assert not sem.tracks[oid].kind_locked
    room.remove("c")
    out, t = run(sem, room, 16, t)
    assert by_id(out)[oid]["State"] == "Removed"


# ---------------------------------------------------------------- re-classification on request
def test_reclassify_unlocks_and_lets_the_classifier_decide_again():
    room, sem, oid, t = _locked_chair()
    room.place("c", coffee_table(0.5, 0.5, 0.0, w=1.0, d=0.5, h=0.42))
    out, t = run(sem, room, 3, t)
    assert by_id(out)[oid]["Kind"] == "Chair"               # locked: stays a chair
    assert sem.reclassify([oid]) == [oid]
    assert not sem.tracks[oid].kind_locked
    out, t = run(sem, room, P.kind_switch_confirm, t)
    o = by_id(out)[oid]
    assert o["Kind"] == "Table" and o["State"] == "Present"
    assert seats_of(out, oid) == []
    out, t = run(sem, room, P.kind_lock_obs, t)
    assert sem.tracks[oid].kind_locked and sem.tracks[oid].kind == "Table"
    assert [x["Id"] for x in live(out)] == [oid]


def test_reclassify_ids_unknown_ignored_empty_means_all():
    room = make_room()
    room.place("a", chair(-1.0, 0.0, 0.0))
    room.place("b", chair(1.0, 0.0, 0.0))
    sem = new_sem()
    out, t = run(sem, room, LOCK, 0.0)
    a, b = sorted(o["Id"] for o in live(out, "Chair"))
    sem.reject(b)
    assert sem.reclassify(["fused:999"]) == []
    assert sem.tracks[a].kind_locked
    assert sem.reclassify() == [a]                          # Rejected objects are left alone
    assert not sem.tracks[a].kind_locked
    out, t = run(sem, room, LOCK, t)
    assert sem.tracks[a].kind_locked
    assert sem.reclassify([]) == [a]
    assert by_id(out)[b]["State"] == "Rejected"


def test_reclassified_object_is_still_never_removed():
    room, sem, oid, t = _locked_chair()
    sem.reclassify([oid])
    room.remove("c")
    out, t = run(sem, room, 40, t)
    assert by_id(out)[oid]["State"] == "Missing"


def test_unstable_chair_keeps_its_kind_while_the_user_sits_on_it():
    # the seated body distorts the geometry; the seat must not vanish under the user
    room, sem, oid, out, t = _chair_track(CREATE)
    room.place("c", stool(0.5, 0.5, 0.0, h=0.44))           # reads as Other
    out, t = run(sem, room, 4, t, user_head=(0.5, 1.2, 0.45))
    o = by_id(out)[oid]
    assert o["Kind"] == "Chair"
    assert [s["State"] for s in seats_of(out, oid)] == ["OccupiedByUser"]
