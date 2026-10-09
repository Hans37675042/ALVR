"""Shared helpers for the semantics tests (synthetic room + frame loop)."""
from roomd.semantics import RoomSemantics, SyntheticRoom, quat_yaw, yaw_diff

LIVE = ("Present", "Moving", "Missing")


def make_room(**kw):
    kw.setdefault("noise", 0.003)
    kw.setdefault("seed", 1)
    return SyntheticRoom(**kw)


def run(sem, room, frames, t, dt=0.5, **kw):
    out = None
    for _ in range(frames):
        out = sem.update(room, t, **kw)
        t += dt
    return out, t


def by_id(out):
    return {o["Id"]: o for o in out["Objects"]}


def live(out, kind=None):
    return [o for o in out["Objects"]
            if o["State"] in LIVE and (kind is None or o["Kind"] == kind)]


def seats_of(out, object_id):
    return [s for s in out["Seats"] if s["ObjectId"] == object_id]


def pos(o):
    p = o["Pose"]["Position"]
    return p["X"], p["Z"]


def yaw(o):
    r = o["Pose"]["Rotation"]
    return quat_yaw((r["X"], r["Y"], r["Z"], r["W"]))


def near(a, b, tol):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def yaw_close(a, b, tol, symmetric=False):
    return yaw_diff(a, b, symmetric) <= tol


def new_sem(**params):
    from roomd.semantics import SemanticsParams
    return RoomSemantics(SemanticsParams(**params)) if params else RoomSemantics()
