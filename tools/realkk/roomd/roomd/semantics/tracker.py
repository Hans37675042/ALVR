"""Cross-frame tracking and RoomModel v2 objects/seats output."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

from .classify import Candidate, detect
from .geometry import footprint_iou, linear_assignment
from .mapview import MapView, as_heightmap
from .params import SemanticsParams
from .seats import generate_seats, pose_json, seat_areas, seat_occupancy
from .basics import (
    Kind, Obb, ObjectState, SceneLabel, SeatState, Source, wrap_deg, yaw_diff)

LIVE_STATES = (ObjectState.PRESENT, ObjectState.MOVING, ObjectState.MISSING)
_BIG = 1e6


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class _Obs:
    cx: float
    cz: float
    yaw: float
    sx: float
    sy: float
    sz: float
    surface_h: float

    @staticmethod
    def of(c: Candidate) -> "_Obs":
        return _Obs(c.cx, c.cz, c.yaw, c.sx, c.sy, c.sz, c.surface_h)


@dataclass
class Track:
    id: str
    kind: str
    source: str
    label: str
    sittable: bool
    movable: bool
    symmetric: bool
    pub: _Obs                       # published pose/size
    est: _Obs                       # EMA estimate (absorbs noise)
    hist: np.ndarray
    confidence: float
    state: str
    created: float
    updated: float
    last_seen: float
    revision: int = 1
    pub_conf: float = 0.0           # confidence as of the last revision
    pending: List[_Obs] = field(default_factory=list)
    free_count: int = 0
    missing_since: Optional[float] = None
    seat_states: List[str] = field(default_factory=list)

    def footprint(self, floor_y) -> Obb:
        o = self.pub
        return Obb(o.cx, o.cz, o.yaw, o.sx, o.sz, floor_y, floor_y + o.sy)

    def seat_frozen(self) -> bool:
        # The seated user's body distorts the object's geometry; a Blocked seat does
        # not freeze, since another object standing on the old spot also reads as Blocked.
        return SeatState.OCCUPIED_BY_USER in self.seat_states


@dataclass
class _Tentative:
    cand: Candidate
    count: int


def _align_yaw(yaw, ref, symmetric):
    if symmetric and yaw_diff(yaw + 180.0, ref) < yaw_diff(yaw, ref):
        return wrap_deg(yaw + 180.0)
    return yaw


def _deviates(a: _Obs, b: _Obs, symmetric, p: SemanticsParams) -> bool:
    return (math.hypot(a.cx - b.cx, a.cz - b.cz) > p.move_dist
            or yaw_diff(a.yaw, b.yaw, symmetric) > p.move_yaw)


def _mean_obs(obs: Sequence[_Obs]) -> _Obs:
    ref = obs[-1].yaw
    yaws = [ref + wrap_deg(o.yaw - ref) for o in obs]
    return _Obs(*(float(np.mean([getattr(o, f) for o in obs]))
                  for f in ("cx", "cz")),
                wrap_deg(float(np.mean(yaws))),
                *(float(np.mean([getattr(o, f) for o in obs])) for f in ("sx", "sy", "sz", "surface_h")))


class RoomSemantics:
    """Detection + tracking. ``update`` returns the RoomModel v2 fragment
    ``{Revision, FrameSource, FloorY, FloorRms, Walls, Objects, Seats}``."""

    def __init__(self, params: Optional[SemanticsParams] = None):
        self.p = params or SemanticsParams()
        self.tracks: Dict[str, Track] = {}
        self._tentative: List[_Tentative] = []
        self._next_fused = 1
        self.revision = 0
        self._floor = None
        self._walls: List = []
        self._last_seats_sig = None
        self._dirty = True

    # ------------------------------------------------------------ public API
    def reject(self, object_id: str):
        """User deleted an automatic object: keep a Rejected tombstone that
        suppresses re-detection at that place."""
        tr = self.tracks.get(object_id)
        if tr is not None and tr.state != ObjectState.REJECTED:
            tr.state = ObjectState.REJECTED
            tr.revision += 1
            self._dirty = True

    def update(self, view: MapView, t: float, scene_labels: Sequence[SceneLabel] = (),
               user_head: Optional[Sequence[float]] = None) -> dict:
        p = self.p
        det = detect(view, p, scene_labels)
        floor_y = det.floor_y
        self._publish_floor_walls(det)
        hm = as_heightmap(view.heightmap())

        # seat states first: a seat occupied by the user freezes its object's pose, and an
        # object someone sits on is not moving (the body only distorts its geometry)
        for tr in self.tracks.values():
            self._update_seats(tr, hm, user_head, floor_y)
            if tr.seat_frozen() and tr.state == ObjectState.MOVING:
                tr.pending.clear()
                self._bump(tr, ObjectState.PRESENT, t)

        cands = list(det.candidates)
        matched_tracks = set()
        unmatched = []

        # 1) scene candidates bind to their own track
        rest = []
        for c in cands:
            if c.scene_uuid is not None:
                tid = "scene:" + c.scene_uuid
                tr = self.tracks.get(tid)
                if tr is not None and tr.state == ObjectState.REJECTED:
                    continue
                if tr is not None:
                    self._observe(tr, c, t)
                    matched_tracks.add(tid)
                    continue
            rest.append(c)

        # 2) Hungarian between remaining candidates and live tracks
        live = [tr for tr in self.tracks.values()
                if tr.state in LIVE_STATES and tr.id not in matched_tracks]
        if rest and live:
            cost = np.full((len(rest), len(live)), _BIG)
            for i, c in enumerate(rest):
                for j, tr in enumerate(live):
                    cost[i, j] = self._cost(c, tr)
            rows, cols = linear_assignment(cost)
            taken = set()
            for i, j in zip(rows, cols):
                if cost[i, j] >= _BIG:
                    continue
                self._observe(live[j], rest[i], t)
                matched_tracks.add(live[j].id)
                taken.add(i)
            rest = [c for i, c in enumerate(rest) if i not in taken]

        # 3) leftovers: absorbed by an existing object, or new-object candidates
        for c in rest:
            fp = c.footprint(floor_y)
            if any(tr.state in LIVE_STATES + (ObjectState.REJECTED,)
                   and footprint_iou(fp, tr.footprint(floor_y)) >= p.absorb_iou
                   for tr in self.tracks.values()):
                continue
            unmatched.append(c)
        self._advance_tentatives(unmatched, t, floor_y)

        # 4) unmatched live tracks: only free-space evidence makes them disappear
        for tr in self.tracks.values():
            if tr.id in matched_tracks or tr.state not in LIVE_STATES:
                continue
            self._check_missing(tr, view, t, floor_y)

        for tr in self.tracks.values():
            self._update_seats(tr, hm, user_head, floor_y)
        return self._output(floor_y, t)

    # ------------------------------------------------------------ matching
    def _cost(self, c: Candidate, tr: Track) -> float:
        p = self.p
        if c.kind != tr.kind:
            return _BIG
        if c.scene_uuid is not None and tr.source == Source.SCENE_API:
            return _BIG  # another anchor's object
        dist = math.hypot(c.cx - tr.pub.cx, c.cz - tr.pub.cz)
        if dist >= p.gate_dist:
            return _BIG
        dsurf = abs(c.surface_h - tr.est.surface_h)
        if dsurf >= p.gate_surface_dh:
            return _BIG
        dsize = (abs(max(c.sx, c.sz) - max(tr.est.sx, tr.est.sz))
                 + abs(min(c.sx, c.sz) - min(tr.est.sx, tr.est.sz)))
        dhist = 0.5 * float(np.abs(c.hist - tr.hist).sum()) if len(c.hist) == len(tr.hist) else 1.0
        return dist + p.cost_w_size * dsize + p.cost_w_surface * dsurf + p.cost_w_hist * dhist

    def _observe(self, tr: Track, c: Candidate, t: float):
        p = self.p
        tr.last_seen = t
        tr.free_count = 0
        tr.missing_since = None
        tr.hist = (1 - p.ema_alpha) * tr.hist + p.ema_alpha * c.hist if len(tr.hist) == len(c.hist) else c.hist
        tr.confidence = (1 - p.ema_alpha) * tr.confidence + p.ema_alpha * c.confidence
        obs = _Obs.of(c)
        obs.yaw = _align_yaw(obs.yaw, tr.pub.yaw, tr.symmetric)
        if tr.state in (ObjectState.MISSING, ObjectState.REMOVED):
            tr.pub = obs
            tr.est = _Obs(**vars(obs))
            tr.pending.clear()
            self._bump(tr, ObjectState.PRESENT, t)
            return
        if tr.seat_frozen():
            tr.pending.clear()
            return
        if not _deviates(tr.pub, obs, tr.symmetric, p):
            tr.pending.clear()
            e = tr.est
            a = p.ema_alpha
            e.cx += a * (obs.cx - e.cx)
            e.cz += a * (obs.cz - e.cz)
            e.yaw = wrap_deg(e.yaw + a * wrap_deg(obs.yaw - e.yaw))
            e.sx += a * (obs.sx - e.sx)
            e.sy += a * (obs.sy - e.sy)
            e.sz += a * (obs.sz - e.sz)
            e.surface_h += a * (obs.surface_h - e.surface_h)
            if tr.state == ObjectState.MOVING:
                self._bump(tr, ObjectState.PRESENT, t)
            return
        tr.pending.append(obs)
        last = tr.pending[-p.move_confirm:]
        if len(last) >= p.move_confirm:
            m = _mean_obs(last)
            if not any(_deviates(m, o, tr.symmetric, p) for o in last):
                tr.pub = m
                tr.est = _Obs(**vars(m))
                tr.pending.clear()
                self._bump(tr, ObjectState.PRESENT, t)
                return
        if len(tr.pending) >= p.moving_after and tr.state != ObjectState.MOVING:
            self._bump(tr, ObjectState.MOVING, t)

    def _bump(self, tr: Track, state: str, t: float):
        tr.state = state
        tr.pub_conf = tr.confidence
        tr.revision += 1
        tr.updated = t
        self._dirty = True

    def _advance_tentatives(self, cands: List[Candidate], t: float, floor_y: float):
        p = self.p
        survivors = []
        used = set()
        for tent in self._tentative:
            best = None
            for i, c in enumerate(cands):
                if i in used or c.kind != tent.cand.kind:
                    continue
                d = math.hypot(c.cx - tent.cand.cx, c.cz - tent.cand.cz)
                if d < p.new_match_dist and (best is None or d < best[1]):
                    best = (i, d)
            if best is not None:
                used.add(best[0])
                survivors.append(_Tentative(cands[best[0]], tent.count + 1))
        for i, c in enumerate(cands):
            if i not in used:
                survivors.append(_Tentative(c, 1))
        self._tentative = []
        for tent in survivors:
            c = tent.cand
            if c.scene_uuid is not None or tent.count >= p.new_confirm:
                self._create(c, t)
            else:
                self._tentative.append(tent)

    def _create(self, c: Candidate, t: float):
        p = self.p
        if c.scene_uuid is None:
            for tr in self.tracks.values():
                if (tr.state == ObjectState.REMOVED and tr.kind == c.kind
                        and math.hypot(c.cx - tr.pub.cx, c.cz - tr.pub.cz) < p.revive_dist
                        and abs(c.surface_h - tr.est.surface_h) < p.gate_surface_dh):
                    self._observe(tr, c, t)
                    return
            tid = f"fused:{self._next_fused}"
            self._next_fused += 1
            source = Source.FUSED
        else:
            tid = "scene:" + c.scene_uuid
            source = Source.SCENE_API
        obs = _Obs.of(c)
        self.tracks[tid] = Track(
            id=tid, kind=c.kind, source=source, label=c.label, sittable=c.sittable,
            movable=c.kind == Kind.CHAIR, symmetric=c.symmetric, pub=obs, est=_Obs(**vars(obs)),
            hist=np.array(c.hist, dtype=float), confidence=c.confidence, pub_conf=c.confidence,
            state=ObjectState.PRESENT, created=t, updated=t, last_seen=t)
        self._dirty = True

    # ------------------------------------------------------------ disappearance
    def _check_missing(self, tr: Track, view: MapView, t: float, floor_y: float):
        p = self.p
        o = tr.pub
        probe = Obb(o.cx, o.cz, o.yaw, o.sx * p.probe_shrink, o.sz * p.probe_shrink,
                    floor_y + o.surface_h - p.probe_below, floor_y + o.surface_h + p.probe_above)
        vis = view.visible_fraction(probe)
        free = view.free_fraction(probe) if vis >= p.missing_visible_min else 0.0
        if vis < p.missing_visible_min or free < p.missing_free_min:
            tr.free_count = 0
            return  # occluded or still occupied: no evidence of disappearance
        tr.free_count += 1
        if tr.state != ObjectState.MISSING:
            if tr.free_count >= p.missing_confirm:
                tr.missing_since = t
                tr.pending.clear()
                self._bump(tr, ObjectState.MISSING, t)
        elif t - tr.missing_since >= p.remove_after_s:
            self._bump(tr, ObjectState.REMOVED, t)

    # ------------------------------------------------------------ seats
    def _update_seats(self, tr: Track, hm, user_head, floor_y):
        if not tr.sittable or tr.state in (ObjectState.REMOVED, ObjectState.REJECTED):
            tr.seat_states = []
            return
        o = tr.pub
        areas = seat_areas(o.cx, o.cz, o.yaw, (o.sx, o.sy, o.sz), self.p)
        tr.seat_states = seat_occupancy(areas, o.surface_h, hm, user_head, floor_y, self.p)

    def _seat_states_out(self, tr: Track, n: int) -> List[str]:
        if tr.state in (ObjectState.REMOVED, ObjectState.REJECTED):
            return [SeatState.REMOVED] * n
        if tr.state == ObjectState.MISSING:
            return [SeatState.MISSING] * n
        if tr.state == ObjectState.MOVING:
            return [SeatState.MOVING] * n
        states = tr.seat_states if len(tr.seat_states) == n else [SeatState.AVAILABLE] * n
        return list(states)

    # ------------------------------------------------------------ output
    def _publish_floor_walls(self, det):
        p = self.p
        if self._floor is None or abs(det.floor_y - self._floor[0]) > p.floor_change:
            self._floor = (det.floor_y, det.floor_rms)
            self._dirty = True
        if self._walls_changed(det.walls):
            self._walls = det.walls
            self._dirty = True

    def _walls_changed(self, walls) -> bool:
        if len(walls) != len(self._walls):
            return True
        tol = self.p.wall_change
        for w in walls:
            if not any(max(math.dist(w[0], o[0]), math.dist(w[1], o[1])) <= tol
                       or max(math.dist(w[0], o[1]), math.dist(w[1], o[0])) <= tol
                       for o in self._walls):
                return True
        return False

    def _output(self, floor_y: float, t: float) -> dict:
        objects, seats = [], []
        floor_pub = self._floor[0]
        for tr in sorted(self.tracks.values(), key=lambda x: x.created):
            o = tr.pub
            objects.append({
                "Id": tr.id,
                "Kind": tr.kind,
                "Pose": pose_json(o.cx, floor_pub, o.cz, o.yaw),
                "Size": {"X": round(o.sx, 4), "Y": round(o.sy, 4), "Z": round(o.sz, 4)},
                "Sittable": tr.sittable,
                "Movable": tr.movable,
                "Source": tr.source,
                "Label": tr.label,
                "Confidence": round(float(tr.pub_conf), 3),
                "State": tr.state,
                "Locked": False,
                "Tracked": False,
                "Revision": tr.revision,
                "CreatedUtc": iso_utc(tr.created),
                "UpdatedUtc": iso_utc(tr.updated),
                "LastSeenUtc": iso_utc(tr.last_seen),
            })
            if tr.sittable:
                n = len(generate_seats(tr.id, (0, 0, 0), 0.0, (o.sx, o.sy, o.sz), o.surface_h))
                seats.extend(generate_seats(tr.id, (o.cx, floor_pub, o.cz), o.yaw,
                                            (o.sx, o.sy, o.sz), o.surface_h, tr.source,
                                            self._seat_states_out(tr, n)))
        sig = tuple((s["Id"], s["State"]) for s in seats)
        if sig != self._last_seats_sig:
            self._last_seats_sig = sig
            self._dirty = True
        if self._dirty or self.revision == 0:
            self.revision += 1
            self._dirty = False
        return {
            "Revision": self.revision,
            "FrameSource": "Stage",
            "FloorY": round(float(floor_pub), 4),
            "FloorRms": round(float(self._floor[1]), 4),
            "Walls": [{"a": [round(a[0], 3), round(a[1], 3)], "b": [round(b[0], 3), round(b[1], 3)]}
                      for a, b in self._walls],
            "Objects": objects,
            "Seats": seats,
        }
