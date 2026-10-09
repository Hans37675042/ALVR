"""Per-frame detection: blobs -> furniture candidates (R15 table) + scene label fusion."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    adjacency_counts, detect_walls, footprint_iou, label_components, min_area_rect, rasterize_max)
from .mapview import MapView, as_heightmap, heightmap_cells
from .params import SemanticsParams
from .basics import Kind, Obb, SceneLabel, wrap_deg, yaw_diff, yaw_from_front, yaw_from_right

SITTABLE_KINDS = (Kind.CHAIR, Kind.COUCH)

# Meta scene semantic labels -> RoomObjectKind; labels not listed are not furniture.
LABEL_KIND = {
    "CHAIR": Kind.CHAIR,
    "COUCH": Kind.COUCH,
    "TABLE": Kind.TABLE,
    "BED": Kind.BED,
    "STORAGE": Kind.OTHER,
    "OTHER": Kind.OTHER,
    "SCREEN": Kind.OTHER,
    "LAMP": Kind.OTHER,
    "PLANT": Kind.OTHER,
}


@dataclass
class Candidate:
    """One furniture observation in a single frame (footprint box on the floor)."""
    kind: str
    cx: float
    cz: float
    yaw: float
    sx: float
    sy: float
    sz: float
    surface_h: float           # main horizontal surface above the floor (seat / table top)
    sittable: bool
    confidence: float
    symmetric: bool            # no front known: yaw and yaw+180 are the same object
    hist: np.ndarray
    has_back: bool = False
    perch: bool = False        # backless seat-height surface: not sittable by default
    label: str = ""
    scene_uuid: Optional[str] = None

    def footprint(self, floor_y: float) -> Obb:
        return Obb(self.cx, self.cz, self.yaw, self.sx, self.sz, floor_y, floor_y + self.sy)


@dataclass
class Detection:
    floor_y: float
    floor_rms: float
    walls: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    candidates: List[Candidate] = field(default_factory=list)


# ---------------------------------------------------------------- segmentation

@dataclass
class _Geom:
    base_xz: np.ndarray
    base_rel: np.ndarray
    upper_xz: np.ndarray
    upper_rel: np.ndarray
    surface_h: float
    flat_ratio: float


def _split_by_patches(pieces, n, patches, big, patch_mean, p: SemanticsParams):
    """Split a piece holding two or more large flat patches at different heights
    (bed + nightstand pushed together, both under ``split_dh`` apart): every cell
    goes to its nearest large patch."""
    out = pieces.copy()
    next_label = n
    for k in range(n):
        iz, ix = np.nonzero(pieces == k)
        pid = patches[iz, ix]
        ids = np.unique(pid[pid >= 0])
        ids = ids[big[ids]]
        if len(ids) < 2 or np.ptp(patch_mean[ids]) < p.split_patch_dh:
            continue
        seed = np.isin(pid, ids)
        tree = cKDTree(np.column_stack([iz[seed], ix[seed]]))
        _, nn = tree.query(np.column_stack([iz, ix]))
        owner = pid[seed][nn]
        for b in ids[1:]:
            sel = owner == b
            out[iz[sel], ix[sel]] = next_label
            next_label += 1
    return out, next_label


def _segment(raster, floor_y, p: SemanticsParams) -> List[_Geom]:
    top = raster.top
    if top.size == 0:
        return []
    rel = top - floor_y
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(top) & (rel >= p.blob_min_h) & (rel <= p.blob_max_h)
    pieces, n = label_components(mask, top, p.split_dh)
    if n == 0:
        return []
    patches, npatch = label_components(mask, top, p.patch_tol)
    cell_area = raster.cell ** 2

    # valid horizontal patches
    patch_ok = np.zeros(npatch, dtype=bool)
    patch_mean = np.zeros(npatch)
    if npatch:
        pl = patches[patches >= 0]
        pr = rel[patches >= 0]
        counts = np.bincount(pl, minlength=npatch)
        order = np.argsort(pl, kind="stable")
        bounds = np.searchsorted(pl[order], np.arange(npatch + 1))
        for k in np.nonzero(counts * cell_area >= p.patch_min_area)[0]:
            h = pr[order[bounds[k]:bounds[k + 1]]]
            lo, hi = np.percentile(h, (5, 95))
            patch_ok[k] = hi - lo <= p.patch_max_range
        patch_mean = np.bincount(pl, weights=pr, minlength=npatch) / np.maximum(counts, 1)
        big = patch_ok & (counts * cell_area >= p.min_object_area)
        pieces, n = _split_by_patches(pieces, n, patches, big, patch_mean, p)

    iz_all, ix_all = np.nonzero(pieces >= 0)
    lab = pieces[iz_all, ix_all]
    order = np.argsort(lab, kind="stable")
    iz_all, ix_all, lab = iz_all[order], ix_all[order], lab[order]
    bounds = np.searchsorted(lab, np.arange(n + 1))
    cells = [(iz_all[bounds[k]:bounds[k + 1]], ix_all[bounds[k]:bounds[k + 1]]) for k in range(n)]
    area = np.array([len(c[0]) * cell_area for c in cells])
    med = np.array([float(np.median(rel[c])) if len(c[0]) else 0.0 for c in cells])
    short = np.zeros(n)
    for k, (iz, ix) in enumerate(cells):
        if len(iz) == 0:
            continue
        _, _, _, eu, ev = min_area_rect(raster.centres(iz, ix))
        short[k] = min(eu, ev) + raster.cell

    # thin, higher pieces (backrests, arms, headboards) attach to their lower neighbour
    adj = adjacency_counts(pieces)
    nbrs = {}
    for (a, b), cnt in adj.items():
        nbrs.setdefault(a, []).append((b, cnt))
        nbrs.setdefault(b, []).append((a, cnt))
    parent = np.arange(n)
    for k in np.argsort(area):
        if short[k] > p.thin_max:
            continue
        best = None
        for q, cnt in nbrs.get(k, []):
            if area[q] > area[k] and med[k] > med[q] + p.attach_min_rise:
                if best is None or cnt > best[1]:
                    best = (q, cnt)
        if best is not None:
            parent[k] = best[0]

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k

    groups = {}
    for k in range(n):
        groups.setdefault(root(k), []).append(k)

    out = []
    for r, members in groups.items():
        if area[r] < p.min_piece_area:
            continue
        biz, bix = cells[r]
        uiz = np.concatenate([cells[k][0] for k in members if k != r] or [np.empty(0, int)])
        uix = np.concatenate([cells[k][1] for k in members if k != r] or [np.empty(0, int)])
        base_patch = patches[biz, bix]
        surface_h, flat = float(np.median(rel[biz, bix])), 0.0
        valid = base_patch[(base_patch >= 0)]
        valid = valid[patch_ok[valid]] if valid.size else valid
        if valid.size:
            ids, cnt = np.unique(valid, return_counts=True)
            best = ids[cnt.argmax()]
            sel = base_patch == best
            surface_h = float(np.median(rel[biz[sel], bix[sel]]))
            flat = float(cnt.max()) / len(biz)
        out.append(_Geom(raster.centres(biz, bix), rel[biz, bix],
                         raster.centres(uiz, uix), rel[uiz, uix] if len(uiz) else np.empty(0),
                         surface_h, flat))
    return out


# ---------------------------------------------------------------- classification

def _hist(rel, p: SemanticsParams):
    edges = np.arange(0.0, p.hist_max + p.hist_bin / 2, p.hist_bin)
    h, _ = np.histogram(np.clip(rel, 0.0, p.hist_max - 1e-6), bins=edges)
    s = h.sum()
    return h / s if s else h.astype(float)


def _in(v, rng):
    return rng[0] <= v <= rng[1]


def _classify(g: _Geom, cell: float, p: SemanticsParams) -> Optional[Candidate]:
    all_xz = np.vstack([g.base_xz, g.upper_xz]) if len(g.upper_xz) else g.base_xz
    all_rel = np.concatenate([g.base_rel, g.upper_rel])
    area = len(all_xz) * cell * cell
    if area < p.min_object_area:
        return None
    centre, u, v, eu, ev = min_area_rect(all_xz)
    eu += cell
    ev += cell
    s = g.surface_h
    sy = float(all_rel.max())

    # backrest: attached cells clearly above the main surface, spanning the seat width
    has_back = False
    front = None
    if len(g.upper_xz):
        back = g.upper_xz[g.upper_rel >= s + p.back_min_rise]
        if len(back):
            d = g.base_xz.mean(axis=0) - back.mean(axis=0)
            if abs(d @ u) >= abs(d @ v):
                f_axis, f_ext, p_axis, p_ext = u, eu, v, ev
            else:
                f_axis, f_ext, p_axis, p_ext = v, ev, u, eu
            sign = 1.0 if d @ f_axis >= 0 else -1.0
            back_len = np.ptp(back @ p_axis) + cell
            base_w = np.ptp(g.base_xz @ p_axis) + cell
            if back_len >= p.back_min_cover * base_w and abs(d @ f_axis) >= 0.05:
                has_back = True
                front = (sign * f_axis, f_ext, p_ext)

    if has_back:
        fvec, sz, sx = front
        yaw = yaw_from_front(fvec[0], fvec[1])
        symmetric = False
    else:
        long_axis = u if eu >= ev else v
        sx, sz = max(eu, ev), min(eu, ev)
        yaw = wrap_deg(yaw_from_right(long_axis[0], long_axis[1]))
        if yaw <= -90.0:
            yaw += 180.0
        elif yaw > 90.0:
            yaw -= 180.0
        symmetric = True

    flat = g.flat_ratio >= p.flat_min_ratio
    couch_like = has_back and sx > sz
    perch = False
    if area >= p.bed_min_area and _in(s, p.bed_h) and flat and not couch_like:
        kind, sittable, conf = Kind.BED, p.bed_sittable, p.conf_bed
    elif has_back and flat and _in(s, p.seat_h):
        if sx >= p.couch_min_len:
            kind, conf = Kind.COUCH, p.conf_couch
        else:
            kind, conf = Kind.CHAIR, p.conf_chair
        sittable = True
    elif flat and _in(s, p.table_h):
        kind, sittable, conf = Kind.TABLE, False, p.conf_table
    elif flat and not has_back and _in(s, p.perch_h):
        perch = True
        sittable, conf = False, p.conf_perch
        kind = Kind.TABLE if area >= p.perch_table_min_area else Kind.OTHER
    else:
        kind, sittable, conf = Kind.OTHER, False, p.conf_other
    conf = float(np.clip(conf * (0.5 + 0.5 * min(1.0, g.flat_ratio)), 0.0, 1.0))
    return Candidate(kind, float(centre[0]), float(centre[1]), float(yaw), float(sx), sy, float(sz),
                     s, sittable, conf, symmetric, _hist(all_rel, p), has_back=has_back, perch=perch)


# ---------------------------------------------------------------- scene labels

def _label_kind(labels: str) -> Optional[str]:
    for tok in labels.split(","):
        k = LABEL_KIND.get(tok.strip().upper())
        if k is not None:
            return k
    return None


def _away_from_wall_yaw(obb: Obb, walls):
    """MRUK rule for label-only seats: front faces away from the nearest wall."""
    if not walls:
        return obb.yaw, obb.sx, obb.sz
    c = np.array([obb.cx, obb.cz])
    best = None
    for a, b in walls:
        a, b = np.array(a), np.array(b)
        ab = b - a
        t = np.clip((c - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
        q = a + t * ab
        dist = np.linalg.norm(c - q)
        if best is None or dist < best[0]:
            best = (dist, c - q)
    away = best[1]
    options = []
    for k in range(4):
        y = obb.yaw + 90.0 * k
        f = np.array([math.sin(math.radians(y)), math.cos(math.radians(y))])
        options.append((f @ away, k, y))
    _, k, y = max(options)
    sx, sz = (obb.sz, obb.sx) if k % 2 else (obb.sx, obb.sz)
    return wrap_deg(y), sx, sz


def _is_stale(view: MapView, hm, obb: Obb, p: SemanticsParams) -> bool:
    rel, known = heightmap_cells(hm, obb)
    if rel.size == 0:
        return False
    if known.mean() < p.stale_known_min:
        return False
    with np.errstate(invalid="ignore"):
        occupied = (rel[known] > p.blob_min_h).mean()
    return occupied <= p.stale_occupied_max


def fuse_scene_labels(cands: List[Candidate], labels: Sequence[SceneLabel], view: MapView,
                      floor_y: float, walls, p: SemanticsParams) -> List[Candidate]:
    """Label decides kind; geometry decides pose; unmatched geometry stays Fused."""
    if not labels:
        return cands
    hm = None
    used = set()
    out = []
    for lab in labels:
        kind = _label_kind(lab.labels)
        if kind is None:
            continue
        best, best_score = None, 0.0
        for i, c in enumerate(cands):
            if i in used:
                continue
            fp = c.footprint(floor_y)
            iou = footprint_iou(fp, lab.obb)
            dist = math.hypot(c.cx - lab.obb.cx, c.cz - lab.obb.cz)
            if iou >= p.label_iou or dist < p.label_center_dist:
                score = iou + (1.0 - dist)
                if best is None or score > best_score:
                    best, best_score = i, score
        sittable = kind in SITTABLE_KINDS or (kind == Kind.BED and p.bed_sittable)
        if best is not None:
            used.add(best)
            g = cands[best]
            yaw = g.yaw
            if g.symmetric and yaw_diff(yaw + 180.0, lab.obb.yaw) < yaw_diff(yaw, lab.obb.yaw):
                yaw = wrap_deg(yaw + 180.0)
            if g.kind == kind:
                conf = max(p.label_conf, g.confidence)
            else:
                conf = min(p.label_conf, g.confidence) * p.label_conflict_scale
            surface = g.surface_h
            if sittable and not _in(surface, (p.seat_h[0] - 0.1, p.seat_h[1] + 0.1)):
                surface = p.default_seat_h
            out.append(replace(g, kind=kind, yaw=yaw, sittable=sittable, confidence=conf,
                               symmetric=g.symmetric and not sittable, surface_h=surface,
                               perch=False, label=lab.labels, scene_uuid=lab.uuid))
            continue
        if hm is None:
            hm = as_heightmap(view.heightmap())
        if _is_stale(view, hm, lab.obb, p):
            continue
        yaw, sx, sz = lab.obb.yaw, lab.obb.sx, lab.obb.sz
        if sittable:
            yaw, sx, sz = _away_from_wall_yaw(lab.obb, walls)
        sy = lab.obb.y1 - floor_y
        surface = p.default_seat_h if sittable else sy
        hist = _hist(np.array([surface]), p)
        out.append(Candidate(kind, lab.obb.cx, lab.obb.cz, yaw, sx, sy, sz, surface, sittable,
                             p.label_only_conf, not sittable, hist, label=lab.labels,
                             scene_uuid=lab.uuid))
    out.extend(c for i, c in enumerate(cands) if i not in used)
    return out


# ---------------------------------------------------------------- entry point

def detect(view: MapView, params: Optional[SemanticsParams] = None,
           scene_labels: Sequence[SceneLabel] = ()) -> Detection:
    p = params or SemanticsParams()
    floor_y, floor_rms = view.floor_plane()
    pts, nrm = view.surface_points(floor_y + 0.03, floor_y + p.blob_max_h + 1.0)
    walls = detect_walls(pts, nrm, floor_y, p)
    if len(pts):
        rel = pts[:, 1] - floor_y
        up = (nrm[:, 1] >= p.up_normal_min) & (rel >= p.blob_min_h) & (rel <= p.blob_max_h)
        raster = rasterize_max(pts[up], p.raster_cell)
    else:
        raster = rasterize_max(pts, p.raster_cell)
    cands = []
    for g in _segment(raster, floor_y, p):
        c = _classify(g, raster.cell, p)
        if c is not None:
            cands.append(c)
    cands = fuse_scene_labels(cands, scene_labels, view, floor_y, walls, p)
    return Detection(float(floor_y), float(floor_rms), walls, cands)
