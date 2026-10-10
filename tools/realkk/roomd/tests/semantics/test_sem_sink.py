"""SemanticsSink: wraps the fusion FusionSink and fills objects/seats/walls (roomd.model types)."""
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from roomd import model as M
from roomd.sink import FusionOutputs, ScenePrior
from roomd.semantics import SemanticsSink
from roomd.semantics.fusion_view import FusionMapView
from roomd.semantics.synthetic import chair, couch, user_torso
from roomd.semantics import Obb, SyntheticRoom


@dataclass
class FakeFusionObb:
    """Same fields as roomd.fusion.tsdf.Obb: centre, half extents, yaw in radians."""
    center: tuple
    half_extents: tuple
    yaw: float = 0.0


class FakeFusion:
    """TsdfFusion query API on top of SyntheticRoom (upward surface points only)."""

    def __init__(self, room):
        self.room = room
        self.obbs = []

    def floor_plane(self):
        y, rms = self.room.floor_plane()
        return SimpleNamespace(y=y, normal=(0.0, 1.0, 0.0), rms=rms, count=1000)

    def heightmap(self):
        hm = self.room.heightmap()
        h, w = hm.top_y.shape
        return SimpleNamespace(cell=hm.cell, origin_x=hm.origin_x, origin_z=hm.origin_z, width=w,
                               height=h, floor_y=hm.floor_y, top_y=hm.top_y, flags=hm.flags)

    def _mine(self, obb):
        self.obbs.append(obb)
        cx, cy, cz = obb.center
        hx, hy, hz = obb.half_extents
        return Obb(cx, cz, math.degrees(obb.yaw), 2 * hx, 2 * hz, cy - hy, cy + hy)

    def surface_points(self, min_y, max_y, region=None):
        p, n = self.room.surface_points(min_y, max_y, None if region is None else self._mine(region))
        up = n[:, 1] >= 0.85
        return p[up], n[up]

    def free_fraction(self, obb):
        return self.room.free_fraction(self._mine(obb))

    def occupied_fraction(self, obb):
        return self.room.occupied_fraction(self._mine(obb))

    def visible_fraction(self, obb, max_age_frames=10):
        return self.room.visible_fraction(self._mine(obb))


class FakeInner:
    name = "fake"

    def __init__(self, room, floor=True):
        self.fusion = FakeFusion(room)
        self.floor = floor
        self.calls = []

    def integrate(self, frame):
        self.calls.append(("integrate", frame))

    def snapshot_outputs(self):
        return FusionOutputs(map_revision=7, floor_y=0.0 if self.floor else None,
                             floor_rms=0.003 if self.floor else None)

    def set_scene_prior(self, prior):
        self.calls.append(("prior", prior))

    def on_playspace_changed(self, pose):
        self.calls.append(("recenter", pose))

    def close(self):
        self.calls.append(("close",))


def _room():
    return SyntheticRoom(noise=0.003, seed=4)


def _sink(room, **kw):
    clock = iter(np.arange(1_000.0, 2_000.0, 0.5))
    return SemanticsSink(FakeInner(room), clock=lambda: next(clock), obb_cls=FakeFusionObb, **kw)


def _frame(head_unity):
    x, y, z = head_unity
    # OpenXR stage: z flipped; two depth views 3 cm either side of the head
    views = [SimpleNamespace(pose_xr=(x + dx, y, -z, 0.0, 0.0, 0.0, 1.0)) for dx in (-0.03, 0.03)]
    return SimpleNamespace(views=views)


def test_map_view_converts_boxes_to_fusion_convention():
    room = _room()
    fusion = FakeFusion(room)
    view = FusionMapView(fusion, obb_cls=FakeFusionObb)
    view.free_fraction(Obb(1.0, 2.0, 90.0, 0.4, 0.6, 0.2, 0.5))
    got = fusion.obbs[-1]
    assert got.center == pytest.approx((1.0, 0.35, 2.0))
    assert got.half_extents == pytest.approx((0.2, 0.15, 0.3))
    assert got.yaw == pytest.approx(math.pi / 2)
    y, rms = view.floor_plane()
    assert y == 0.0
    hm = view.heightmap()
    assert hm.top_y.shape == (hm.floor_y.shape)


def test_sink_fills_fused_objects_and_seats_as_model_types():
    room = _room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sink = _sink(room)
    for _ in range(3):
        out = sink.snapshot_outputs()
    assert out.map_revision == 7 and out.floor_y == 0.0
    assert len(out.objects) == 1
    o = out.objects[0]
    assert isinstance(o, M.RoomObject)
    assert o.Id == "fused:1" and o.Kind == M.Kind.Chair and o.Source == M.SOURCE_FUSED
    assert o.Locked is False and o.Sittable
    assert [type(s) for s in out.seats] == [M.SeatPoint]
    assert out.seats[0].Id == "fused:1#0"
    # same geometry as model.generate_seats (SeatGenerator rule)
    ref = M.generate_seats(o, out.seats[0].Height)
    assert out.seats[0].Pose.Position.X == pytest.approx(ref[0].Pose.Position.X, abs=1e-5)
    assert out.seats[0].Pose.Position.Z == pytest.approx(ref[0].Pose.Position.Z, abs=1e-5)
    M.room_to_json(M.RoomModel(Objects=out.objects, Seats=out.seats))


def test_no_floor_means_no_objects():
    sink = SemanticsSink(FakeInner(_room(), floor=False), obb_cls=FakeFusionObb)
    out = sink.snapshot_outputs()
    assert out.objects is None and out.seats is None and out.walls is None


def test_scene_objects_can_be_left_to_roomd_and_prior_supplies_walls():
    room = _room()
    room.place("k", couch(0.0, 0.5, 0.0))
    sink = _sink(room, publish_scene_objects=False)
    prior_obj = M.RoomObject(Id="scene:u1", Kind=M.Kind.Couch, Source=M.SOURCE_SCENE, Label="COUCH",
                             Pose=M.Pose(M.Vec3(0.0, 0.0, 0.5), M.Quat.from_yaw(0.0)),
                             Size=M.Vec3(2.0, 0.85, 0.9), Locked=False)
    walls = [M.Wall((-2.5, 2.5), (2.5, 2.5))]
    prior = ScenePrior(snapshot_id=1, floor_y=0.0, objects=[prior_obj], walls=walls)
    sink.set_scene_prior(prior)
    assert ("prior", prior) in sink.inner.calls
    for _ in range(4):
        out = sink.snapshot_outputs()
    assert out.objects == [] and out.seats == []
    assert out.walls == walls  # fusion surface points are upward only: no fused walls


def test_scene_objects_published_by_default_and_follow_geometry():
    room = _room()
    room.place("k", couch(0.0, 0.5, 0.0))
    sink = _sink(room)
    prior_obj = M.RoomObject(Id="scene:u1", Kind=M.Kind.Couch, Source=M.SOURCE_SCENE, Label="COUCH",
                             Pose=M.Pose(M.Vec3(0.1, 0.0, 0.6), M.Quat.from_yaw(0.0)),
                             Size=M.Vec3(2.0, 0.85, 0.9), Locked=False)
    sink.set_scene_prior(ScenePrior(snapshot_id=1, floor_y=0.0, objects=[prior_obj], walls=[]))
    out = sink.snapshot_outputs()
    assert [o.Id for o in out.objects] == ["scene:u1"]
    assert out.objects[0].Source == M.SOURCE_SCENE
    assert abs(out.objects[0].Pose.Position.Z - 0.5) < 0.05  # geometry corrected the pose
    assert len(out.seats) == 3


def test_head_from_depth_views_marks_seat_occupied():
    room = _room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sink = _sink(room)
    for _ in range(3):
        sink.snapshot_outputs()
    room.place("u", user_torso(0.5, 0.5, 0.0))
    sink.integrate(_frame((0.5, 1.2, 0.42)))
    out = sink.snapshot_outputs()
    assert [s.State for s in out.seats] == [M.SeatState.OccupiedByUser.value]
    assert sink.inner.calls[0][0] == "integrate"


def test_recenter_resets_tracking_and_forwards():
    room = _room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sink = _sink(room)
    for _ in range(3):
        out = sink.snapshot_outputs()
    assert len(out.objects) == 1
    sink.on_playspace_changed((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert sink.inner.calls[-1][0] == "recenter"
    out = sink.snapshot_outputs()
    assert out.objects == []
    sink.close()
    assert sink.inner.calls[-1] == ("close",)


def test_structure_cut_follows_fusion_heightmap_cap():
    inner = FakeInner(_room())
    # structure_min_h = min(default 1.5, cap - 0.05)
    inner.fusion.config = SimpleNamespace(obstacle_max_height=1.9)
    sink = SemanticsSink(inner, obb_cls=FakeFusionObb)
    assert sink.semantics.p.structure_min_h == pytest.approx(1.5)
    inner.fusion.config = SimpleNamespace(obstacle_max_height=1.5)
    low = SemanticsSink(inner, obb_cls=FakeFusionObb)
    assert low.semantics.p.structure_min_h == pytest.approx(1.45)
    explicit = SemanticsSink(inner, params=__import__("roomd.semantics", fromlist=["x"]).SemanticsParams(
        structure_min_h=1.2), obb_cls=FakeFusionObb)
    assert explicit.semantics.p.structure_min_h == 1.2
    assert SemanticsSink(FakeInner(_room()), obb_cls=FakeFusionObb).semantics.p.structure_min_h == 1.5


def test_fused_other_objects_are_not_published_but_scene_other_is():
    from roomd.semantics.synthetic import storage
    room = _room()
    room.place("c", chair(0.5, 0.5, 0.0))
    room.place("box", storage(-1.2, -1.0, 0.0, w=0.6, d=0.4, h=0.8))
    room.place("cab", storage(1.5, -1.5, 0.0, w=0.8, d=0.4, h=0.9))
    sink = _sink(room)
    cab = M.RoomObject(Id="scene:cab", Kind=M.Kind.Other, Source=M.SOURCE_SCENE, Label="STORAGE",
                       Pose=M.Pose(M.Vec3(1.5, 0.0, -1.5), M.Quat.from_yaw(0.0)),
                       Size=M.Vec3(0.8, 0.9, 0.4), Locked=False)
    sink.set_scene_prior(ScenePrior(snapshot_id=1, floor_y=0.0, objects=[cab], walls=[]))
    for _ in range(4):
        out = sink.snapshot_outputs()
    published = {(o.Id, o.Kind, o.Source) for o in out.objects}
    assert ("scene:cab", M.Kind.Other, M.SOURCE_SCENE) in published
    assert not any(k == M.Kind.Other and s == M.SOURCE_FUSED for _, k, s in published)
    assert any(k == M.Kind.Chair for _, k, _ in published)
    assert any(o["Kind"] == "Other" and o["Source"] == "Fused"
               for o in sink.semantics._output(0.0, 0.0)["Objects"])  # still tracked internally


def test_map_view_lowers_fusion_normal_filter_only_during_its_query():
    room = _room()
    fusion = FakeFusion(room)
    fusion.config = SimpleNamespace(upward_min_normal_y=0.85)
    seen = []
    orig = fusion.surface_points

    def spy(min_y, max_y, region=None):
        seen.append(fusion.config.upward_min_normal_y)
        return orig(min_y, max_y, region)

    fusion.surface_points = spy
    view = FusionMapView(fusion, obb_cls=FakeFusionObb, up_normal_min=0.65)
    view.surface_points(0.03, 2.0)
    assert seen == [0.65]
    assert fusion.config.upward_min_normal_y == 0.85
    sink = SemanticsSink(FakeInner(room), obb_cls=FakeFusionObb)
    assert sink.view.up_normal_min == sink.semantics.p.up_normal_min


def test_sink_forwards_reclassify_to_the_tracker():
    from roomd.sink import FusionSink
    room = _room()
    room.place("c", chair(0.5, 0.5, 0.0))
    sink = _sink(room)
    for _ in range(10):
        out = sink.snapshot_outputs()
    assert [o.Id for o in out.objects] == ["fused:1"]
    assert sink.semantics.tracks["fused:1"].kind_locked
    assert sink.reclassify(["fused:1", "fused:42"]) == ["fused:1"]
    assert not sink.semantics.tracks["fused:1"].kind_locked
    assert FusionSink().reclassify(None) == []
