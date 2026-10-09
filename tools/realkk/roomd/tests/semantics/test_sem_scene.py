"""SceneApi label fusion: label decides kind, geometry fixes pose and adds missed chairs."""
from roomd.semantics import Obb, SceneLabel, detect, yaw_diff
from roomd.semantics.synthetic import chair, couch, table
from sem_helpers import by_id, live, make_room, new_sem, pos, run


def test_label_decides_kind_and_geometry_fixes_pose():
    room = make_room()
    room.place("c", chair(0.5, 0.5, 90.0))
    # Scene box is 15 cm off and its yaw is wrong; geometry wins on pose.
    label = SceneLabel("u-chair", "CHAIR", Obb(0.62, 0.42, 0.0, 0.5, 0.5, 0.0, 0.9))
    cands = detect(room, scene_labels=[label]).candidates
    assert len(cands) == 1
    c = cands[0]
    assert c.scene_uuid == "u-chair" and c.label == "CHAIR" and c.kind == "Chair"
    assert abs(c.cx - 0.5) < 0.05 and abs(c.cz - 0.5) < 0.05
    assert yaw_diff(c.yaw, 90.0) < 8.0
    assert c.confidence >= 0.85


def test_conflicting_label_wins_kind_with_lower_confidence():
    room = make_room()
    room.place("c", chair(0.0, 0.0, 0.0))
    agree = detect(room, scene_labels=[SceneLabel("u1", "CHAIR", Obb(0, 0, 0, 0.45, 0.45, 0, 0.9))])
    clash = detect(room, scene_labels=[SceneLabel("u1", "TABLE", Obb(0, 0, 0, 0.45, 0.45, 0, 0.9))])
    a, b = agree.candidates[0], clash.candidates[0]
    assert b.kind == "Table" and not b.sittable
    assert b.confidence < a.confidence
    assert b.confidence < 0.6


def test_geometry_adds_chair_missing_from_scene():
    room = make_room()
    room.place("t", table(0.0, 0.0, 0.0))
    room.place("c", chair(1.2, -1.2, 0.0))
    labels = [SceneLabel("u-table", "TABLE", Obb(0.0, 0.0, 0.0, 1.2, 0.8, 0.0, 0.75))]
    cands = detect(room, scene_labels=labels).candidates
    kinds = {(c.kind, c.scene_uuid) for c in cands}
    assert ("Table", "u-table") in kinds
    assert ("Chair", None) in kinds


def test_label_only_object_when_region_unseen():
    room = make_room()
    box = Obb(1.0, 1.0, 0.0, 2.0, 0.9, 0.0, 0.85)
    room.set_hidden([box])
    cands = detect(room, scene_labels=[SceneLabel("u-couch", "COUCH", box)]).candidates
    assert len(cands) == 1
    c = cands[0]
    assert c.kind == "Couch" and c.sittable and c.scene_uuid == "u-couch"
    assert abs(c.surface_h - 0.45) < 1e-6  # default seat height
    assert c.confidence <= 0.6


def test_stale_label_in_visible_free_space_is_dropped():
    room = make_room()
    label = SceneLabel("u-old", "CHAIR", Obb(-1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.9))
    assert detect(room, scene_labels=[label]).candidates == []


def test_non_furniture_labels_ignored_and_storage_is_other():
    room = make_room()
    labels = [SceneLabel("w", "WALL_FACE", Obb(0, 2.5, 0, 5, 0.01, 0, 2.5)),
              SceneLabel("s", "STORAGE", Obb(0.0, 0.0, 0.0, 0.8, 0.4, 0.0, 0.9))]
    room.set_hidden([labels[1].obb])
    cands = detect(room, scene_labels=labels).candidates
    assert [(c.kind, c.scene_uuid) for c in cands] == [("Other", "s")]


def test_scene_object_id_and_immediate_publish():
    room = make_room()
    room.place("k", couch(0.0, 0.5, 0.0))
    label = SceneLabel("abcd-1234", "COUCH", Obb(0.0, 0.5, 0.0, 2.0, 0.9, 0.0, 0.85))
    sem = new_sem()
    out = sem.update(room, 0.0, scene_labels=[label])
    objs = by_id(out)
    assert "scene:abcd-1234" in objs
    o = objs["scene:abcd-1234"]
    assert o["Source"] == "SceneApi" and o["Label"] == "COUCH" and o["Kind"] == "Couch"
    assert len([s for s in out["Seats"] if s["ObjectId"] == o["Id"]]) == 3


def test_scene_object_follows_geometry_when_moved():
    room = make_room()
    room.place("k", couch(0.0, 0.5, 0.0))
    label = SceneLabel("k1", "COUCH", Obb(0.0, 0.5, 0.0, 2.0, 0.9, 0.0, 0.85))
    sem = new_sem()
    out, t = run(sem, room, 2, 0.0, scene_labels=[label])
    room.place("k", couch(0.0, -1.0, 0.0))  # snapshot is now stale
    out, t = run(sem, room, 5, t, scene_labels=[label])
    objs = live(out)
    assert [o["Id"] for o in objs] == ["scene:k1"]
    assert abs(pos(objs[0])[1] - (-1.0)) < 0.06
