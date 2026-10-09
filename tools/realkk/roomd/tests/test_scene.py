import math

import numpy as np
import pytest

from roomd import coords
from roomd import model as M
from roomd import protocol as P
from roomd import scene
from roomd.synth.room import default_room, synth_uuid
from roomd.synth.snapshot import build_snapshot


def _angle_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _yaw(obj):
    fwd = obj.Pose.Rotation.rotate((0.0, 0.0, 1.0))
    return math.atan2(fwd[0], fwd[2])


@pytest.fixture(scope="module")
def converted():
    room = default_room()
    snap = P.decode_room_snapshot(P.encode_room_snapshot(build_snapshot(room, snapshot_id=7)))
    return room, scene.room_from_snapshot(snap, now_utc="2026-10-09T00:00:00Z")


def test_floor_walls_and_frame(converted):
    room, model = converted
    assert model.Version == 2 and model.FrameSource == "Stage"
    assert model.FloorY == pytest.approx(room.floor_y, abs=1e-5)
    assert len(model.Walls) == 4
    ends = sorted(tuple(round(v, 3) for v in w.a) for w in model.Walls)
    assert ends == sorted([(-2.5, -2.0), (2.5, -2.0), (2.5, 2.0), (-2.5, 2.0)])
    walls = [o for o in model.Objects if o.Kind == M.Kind.Wall]
    assert len(walls) == 4 and all(not o.Sittable and o.Size.Y == pytest.approx(2.5) for o in walls)
    # wall boxes face into the room (front = inward normal) and sit behind the face
    south = model.find("scene:" + synth_uuid("wall_south"))
    assert _angle_diff(_yaw(south), 0.0) < 1e-4
    assert south.Pose.Position.Z < -2.0 and south.Size.X == pytest.approx(5.0, abs=1e-4)


def test_labels_ids_and_skips(converted):
    room, model = converted
    labels = {o.Label for o in model.Objects}
    assert not labels & {"CEILING", "DOOR_FRAME", "WINDOW_FRAME", "WALL_ART", "FLOOR", "GLOBAL_MESH"}
    for o in model.Objects:
        assert o.Id.startswith("scene:") and o.Source == "SceneApi" and o.Locked is False
        assert o.State == "Present" and o.CreatedUtc == "2026-10-09T00:00:00Z"
    by_name = {f.name: model.find("scene:" + f.uuid) for f in room.furniture}
    assert by_name["chair"] is None and by_name["clothes"] is None  # no Scene label for chairs
    assert by_name["table"].Kind == M.Kind.Table and by_name["coffee_table"].Kind == M.Kind.Table
    assert by_name["couch"].Kind == M.Kind.Couch and by_name["bed"].Kind == M.Kind.Bed
    assert by_name["cabinet"].Kind == M.Kind.Other and by_name["cabinet"].Label == "STORAGE"
    assert by_name["armchair"].Kind == M.Kind.Chair and by_name["armchair"].Label == "COUCH"


@pytest.mark.parametrize("name", ["table", "couch", "bed", "coffee_table", "cabinet", "armchair"])
def test_box_pose_size_and_front_match_ground_truth(converted, name):
    room, model = converted
    f = room.find(name)
    o = model.find("scene:" + f.uuid)
    p = o.Pose.Position
    assert (p.X, p.Y, p.Z) == pytest.approx((f.center_xz[0], room.floor_y, f.center_xz[1]), abs=1e-4)
    assert (o.Size.X, o.Size.Y, o.Size.Z) == pytest.approx(f.size, abs=1e-4)
    # front = away from the nearest wall (MRUK rule); GT yaws were chosen that way
    assert _angle_diff(_yaw(o), f.yaw) < 1e-3
    # pure yaw rotation
    assert o.Pose.Rotation.X == pytest.approx(0, abs=1e-6) and o.Pose.Rotation.Z == pytest.approx(0, abs=1e-6)


def test_seats_follow_scene_planes(converted):
    room, model = converted
    gt = {s.Id: s for s in room.gt_seats()}  # ground-truth ids are "<name>#<i>"
    for name, count in (("couch", 3), ("bed", 3), ("armchair", 1)):
        f = room.find(name)
        oid = "scene:" + f.uuid
        seats = model.seats_of(oid)
        assert [s.Id for s in seats] == ["%s#%d" % (oid, i) for i in range(count)]
        for i, s in enumerate(seats):
            g = gt["%s#%d" % (name, i)]
            assert s.Height == pytest.approx(f.seat_height, abs=1e-4)
            assert s.Pose.Position.tuple() == pytest.approx(g.Pose.Position.tuple(), abs=1e-4)
            assert s.State == "Available" and s.Source == "SceneApi" and s.Enabled
    assert model.seats_of("scene:" + room.find("table").uuid) == []


def test_openxr_to_unity_flip_single_anchor():
    q = (0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))  # XR yaw 45 deg, volume z-up below
    z_up = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    qq = coords.quat_mul(q, z_up)
    anchor = {"uuid": "11111111-2222-3333-4444-555555555555", "labels": "TABLE",
              "pose": [1.0, 0.72, 2.0, *qq], "bbox2d": None, "boundary2d": None,
              "bbox3d": [-0.5, -0.3, -0.72, 1.0, 0.6, 0.72]}
    snap = P.RoomSnapshot(1, (0, 0, 0, 0, 0, 0, 1), {"rooms": [], "anchors": [anchor]})
    model = scene.room_from_snapshot(snap)
    o = model.find("scene:11111111-2222-3333-4444-555555555555")
    assert o.Pose.Position.tuple() == pytest.approx((1.0, 0.0, -2.0), abs=1e-5)
    assert o.Size.Y == pytest.approx(0.72, abs=1e-5)
    assert sorted([o.Size.X, o.Size.Z]) == pytest.approx([0.6, 1.0], abs=1e-5)


def test_global_mesh_prior_is_unity_space():
    room = default_room()
    snap = build_snapshot(room)
    prior = scene.prior_from_snapshot(snap)
    assert prior.snapshot_id == snap.snapshot_id
    v, t = prior.mesh_vertices, prior.mesh_triangles
    assert v.dtype == np.float32 and t.shape[1] == 3 and len(t) > 0
    # the shell spans the room in Unity coordinates (z not mirrored)
    assert v[:, 2].min() == pytest.approx(room.z_min - 0.1, abs=1e-3)
    assert v[:, 2].max() == pytest.approx(room.z_max + 0.1, abs=1e-3)
    assert v[:, 1].min() == pytest.approx(room.floor_y - 0.1, abs=1e-3)
    # handedness flip reverses triangle winding against the raw mesh
    raw = snap.meshes[0].indices.reshape(-1, 3)
    assert np.array_equal(t[0], raw[0][[0, 2, 1]])
    assert len(prior.objects) == len(scene.room_from_snapshot(snap).Objects)
