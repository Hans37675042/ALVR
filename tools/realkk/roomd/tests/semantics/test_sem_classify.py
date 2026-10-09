"""R15 classification table and the R15 §1.3 style misclassification cases."""
import pytest

from roomd.semantics import SemanticsParams, detect, yaw_diff
from roomd.semantics.synthetic import (
    bed, chair, clothes_pile, coffee_table, couch, stool, storage, table)
from sem_helpers import make_room


def _detect(*items):
    room = make_room()
    for i, boxes in enumerate(items):
        room.place(f"o{i}", boxes)
    return detect(room).candidates


def _one(*items):
    cands = _detect(*items)
    assert len(cands) == 1, [(c.kind, c.surface_h) for c in cands]
    return cands[0]


def test_table():
    c = _one(table(0.3, 0.2, 20.0))
    assert c.kind == "Table" and not c.sittable
    assert abs(c.surface_h - 0.75) < 0.02
    assert 0.0 < c.confidence <= 1.0


def test_chair_seat_height_and_sittable():
    c = _one(chair(0.5, -0.5, 0.0, seat_h=0.47))
    assert c.kind == "Chair" and c.sittable and c.has_back
    assert abs(c.surface_h - 0.47) < 0.02
    assert abs(c.cx - 0.5) < 0.05 and abs(c.cz - (-0.5)) < 0.05


@pytest.mark.parametrize("yaw", [0.0, 90.0, -135.0, 170.0])
def test_chair_faces_away_from_backrest(yaw):
    c = _one(chair(0.0, 0.0, yaw))
    assert yaw_diff(c.yaw, yaw) < 8.0, c.yaw


def test_couch_long_side_is_x():
    c = _one(couch(0.0, 0.5, 30.0, length=2.0, depth=0.9))
    assert c.kind == "Couch" and c.sittable
    assert abs(c.sx - 2.0) < 0.06 and abs(c.sz - 0.9) < 0.06
    assert yaw_diff(c.yaw, 30.0) < 6.0


def test_bed_with_headboard_is_not_couch():
    c = _one(bed(0.0, 0.0, 0.0))
    assert c.kind == "Bed" and not c.sittable


def test_coffee_table_is_low_table_not_sittable():
    c = _one(coffee_table(0.0, 0.0, 0.0))
    assert c.kind == "Table" and not c.sittable and c.perch
    assert c.confidence < 0.6


def test_backless_stool_not_sittable():
    c = _one(stool(0.0, 0.0))
    assert c.kind == "Other" and not c.sittable and c.perch


@pytest.mark.parametrize("h", [0.9, 1.9])
def test_storage_is_other_obstacle(h):
    c = _one(storage(0.0, 0.0, 0.0, h=h))
    assert c.kind == "Other" and not c.sittable


def test_clothes_pile_on_floor_is_not_a_seat():
    c = _one(clothes_pile(0.0, 0.0, size=0.5, height=0.3))
    assert c.kind == "Other" and not c.sittable


def test_low_cabinet_against_wall_wall_is_not_a_backrest():
    room = make_room(half_x=2.5, half_z=2.5)
    room.place("cab", storage(0.0, 2.28, 0.0, w=0.8, d=0.4, h=0.5))
    cands = detect(room).candidates
    assert len(cands) == 1
    assert not cands[0].sittable and cands[0].kind != "Chair"


def test_bed_and_nightstand_are_separate():
    cands = _detect(bed(0.0, 0.0, 0.0, w=1.6, l=2.0), storage(1.05, -0.7, 0.0, w=0.45, d=0.4, h=0.55))
    kinds = sorted(c.kind for c in cands)
    assert len(cands) == 2 and "Bed" in kinds
    assert not any(c.sittable for c in cands)


def test_table_with_chair_beside_it():
    cands = _detect(table(0.0, 0.0, 0.0, w=1.2, d=0.8), chair(0.0, -0.75, 0.0))
    kinds = sorted(c.kind for c in cands)
    assert kinds == ["Chair", "Table"]


def test_chair_pushed_under_table_makes_no_phantom_furniture():
    cands = _detect(table(0.0, 0.0, 0.0, w=1.2, d=0.8), chair(0.0, -0.3, 0.0))
    kinds = [c.kind for c in cands]
    assert kinds.count("Table") == 1
    assert "Couch" not in kinds and "Bed" not in kinds
    assert all(c.kind == "Chair" for c in cands if c.sittable)


def test_floor_and_walls_are_not_objects():
    assert _detect() == []


def test_detection_reports_floor_and_walls():
    room = make_room()
    d = detect(room)
    assert abs(d.floor_y) < 0.005
    assert len(d.walls) == 4


def test_backrest_across_one_cell_gap_still_attaches():
    # TSDF drops tilted-normal points at the seat/backrest edge: a 2-3 cm gap remains
    from roomd.semantics import Obb
    seat = Obb(0.0, 0.0, 0.0, 0.45, 0.40, 0.41, 0.45)
    back = Obb(0.0, -0.20 - 0.03 - 0.025, 0.0, 0.45, 0.05, 0.0, 0.9)
    c = _one([seat, back])
    assert c.kind == "Chair" and c.has_back
    assert yaw_diff(c.yaw, 0.0) < 8.0


class _SparsePoints:
    """Real TSDF output: dense heightmap but only a fraction of upward surface points."""

    def __init__(self, room, keep=0.12, seed=0):
        self.room, self.keep = room, keep
        self.rng = __import__("numpy").random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self.room, name)

    def surface_points(self, min_y, max_y, region=None):
        p, n = self.room.surface_points(min_y, max_y, region)
        m = self.rng.uniform(size=len(p)) < self.keep
        return p[m], n[m]


@pytest.mark.parametrize("yaw", [0.0, 60.0])
def test_sparse_points_still_find_chair_and_table_from_heightmap(yaw):
    room = make_room()
    room.place("c", chair(0.6, -0.6, yaw))
    room.place("t", table(-0.8, 0.6, 0.0))
    cands = detect(_SparsePoints(room)).candidates
    kinds = sorted(c.kind for c in cands)
    assert kinds == ["Chair", "Table"], [(c.kind, c.surface_h) for c in cands]
    ch = next(c for c in cands if c.kind == "Chair")
    assert abs(ch.surface_h - 0.45) < 0.03
    assert yaw_diff(ch.yaw, yaw) < 12.0


def _fusion_like_room(**kw):
    # fusion heightmap: covers the walls too and stops at obstacle_max_height
    return make_room(hm_margin=0.3, hm_cap=1.9, **kw)


def test_capped_heightmap_walls_are_not_objects():
    room = _fusion_like_room()
    assert detect(room, SemanticsParams(structure_min_h=1.85)).candidates == []


def test_wall_next_to_chair_is_not_its_backrest():
    room = _fusion_like_room()
    room.place("c", chair(0.0, 2.2, 0.0))  # front (+Z) towards the wall at z = 2.5
    cands = detect(room, SemanticsParams(structure_min_h=1.85)).candidates
    assert [c.kind for c in cands] == ["Chair"]
    assert yaw_diff(cands[0].yaw, 0.0) < 12.0
    assert cands[0].sy < 1.0


def test_contoured_seat_is_still_a_seat():
    c = _one(chair(0.0, 0.0, 30.0, seat_h=0.44, seat_contour=0.07))
    assert c.kind == "Chair" and c.sittable
    assert abs(c.surface_h - 0.44) < 0.05


def test_narrow_ledge_with_tall_back_is_not_a_chair():
    # real tap: 0.2 m deep shelf edges at seat height in front of taller stuff read as chairs
    from roomd.semantics import Obb
    ledge = Obb(0.0, 0.0, 0.0, 0.5, 0.2, 0.40, 0.45)
    back = Obb(0.0, -0.125, 0.0, 0.5, 0.05, 0.0, 1.1)
    cands = _detect([ledge, back])
    assert cands and not any(c.kind == "Chair" or c.sittable for c in cands)


@pytest.mark.parametrize("cut", [None, 1.5])
def test_desk_under_loft_bed_is_found(cut):
    # real room: loft bed underside at ~1.55 m over the desk hides it in the heightmap
    from roomd.semantics import Obb
    room = make_room()
    room.place("desk", table(0.0, -1.6, 0.0, w=1.6, d=0.7, h=0.72))
    room.place("loft", [Obb(0.0, -1.6, 0.0, 2.0, 0.9, 1.55, 1.65)])
    room.place("cab", storage(1.6, 1.2, 0.0, w=0.8, d=0.5, h=1.3))
    cands = detect(room, SemanticsParams(structure_min_h=cut)).candidates
    tables = [c for c in cands if c.kind == "Table"]
    assert len(tables) == 1, [(c.kind, round(c.surface_h, 2)) for c in cands]
    assert abs(tables[0].surface_h - 0.72) < 0.03
    assert abs(tables[0].sx - 1.6) < 0.1 and abs(tables[0].sz - 0.7) < 0.1


def test_low_box_against_taller_furniture_is_not_a_chair():
    # real tap: suitcase / basket drawers next to taller furniture read as seat + backrest
    from roomd.semantics import Obb
    suitcase = Obb(0.0, 0.0, 0.0, 0.5, 0.35, 0.0, 0.5)
    shelf = Obb(0.0, -0.335, 0.0, 0.6, 0.28, 0.0, 1.2)  # 2 cm behind the suitcase
    cands = _detect([suitcase, shelf])
    assert cands and not any(c.sittable for c in cands), [(c.kind, c.sx, c.sz) for c in cands]


def test_things_under_a_loft_do_not_make_a_backrest():
    from roomd.semantics import Obb
    room = make_room()
    room.place("cab", storage(0.0, -1.6, 0.0, w=0.5, d=0.4, h=0.45))
    room.place("shelf", storage(0.0, -1.95, 0.0, w=0.5, d=0.2, h=1.1))
    room.place("loft", [Obb(0.0, -1.7, 0.0, 2.0, 0.9, 1.55, 1.65)])
    cands = detect(room).candidates
    assert not any(c.sittable for c in cands), [(c.kind, round(c.surface_h, 2)) for c in cands]


def test_seat_deeper_than_any_chair_is_not_a_chair():
    # real tap: a 0.37 x 1.35 m seat-height strip between two desks read as a chair
    from roomd.semantics import Obb
    strip = Obb(0.0, 0.0, 0.0, 0.40, 1.30, 0.38, 0.42)
    back = Obb(0.0, -0.70, 0.0, 0.40, 0.06, 0.0, 1.1)
    cands = _detect([strip, back])
    assert cands and not any(c.kind == "Chair" or c.sittable for c in cands), \
        [(c.kind, round(c.sx, 2), round(c.sz, 2)) for c in cands]


def test_seat_under_a_loft_bed_is_not_sittable():
    # PM: seat-height things under the loft (drawer unit, suitcase) must not become seats
    from roomd.semantics import Obb
    room = make_room()
    room.place("c", chair(0.0, -1.6, 0.0))
    room.place("loft", [Obb(0.0, -1.7, 0.0, 2.0, 0.9, 1.55, 1.65)])
    cands = detect(room).candidates
    assert not any(c.sittable for c in cands), [(c.kind, round(c.surface_h, 2)) for c in cands]
