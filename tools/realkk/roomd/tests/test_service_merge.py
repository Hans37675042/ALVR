"""RoomService.compose: sink objects override Scene API objects with the same id."""
from roomd import model as M
from roomd import protocol as P
from roomd.io.plugin_server import PluginServer
from roomd.service import RoomService
from roomd.sink import FusionOutputs, FusionSink
from roomd.synth.room import default_room, synth_uuid
from roomd.synth.snapshot import build_snapshot


class OverrideSink(FusionSink):
    """Publishes the scene couch moved by 1 m (as semantics does when geometry moved it)."""

    def __init__(self):
        self.objects = []

    def snapshot_outputs(self):
        seats = [s for o in self.objects for s in M.generate_seats(o, 0.45)]
        return FusionOutputs(floor_y=0.0, floor_rms=0.002, objects=list(self.objects), seats=seats)


def test_sink_object_replaces_scene_object_with_same_id():
    server = PluginServer(0, log=lambda *a: None)
    sink = OverrideSink()
    svc = RoomService(sink, server, log=lambda *a: None)
    svc.handle_message(0, P.MSG_ROOM_SNAPSHOT, P.encode_room_snapshot(build_snapshot(default_room(), 1)))
    couch_id = "scene:" + synth_uuid("couch")
    scene_couch = svc.model.find(couch_id)
    assert scene_couch is not None
    scene_seats = len(svc.model.seats_of(couch_id))
    n_scene_objects = len(svc.model.Objects)

    moved = M.RoomObject(Id=couch_id, Kind=M.Kind.Couch, Source=M.SOURCE_SCENE, Label=scene_couch.Label,
                         Pose=M.Pose(M.Vec3(scene_couch.Pose.Position.X + 1.0, 0.0, scene_couch.Pose.Position.Z),
                                     M.Quat(*scene_couch.Pose.Rotation.tuple())),
                         Size=M.Vec3(1.8, 0.85, 0.9), Sittable=True, Locked=False)
    fused = M.RoomObject(Id="fused:1", Kind=M.Kind.Chair, Source=M.SOURCE_FUSED, Sittable=True,
                         Size=M.Vec3(0.45, 0.9, 0.45), Locked=False)
    sink.objects = [moved, fused]
    svc.poll_outputs()
    room = svc.model
    ids = [o.Id for o in room.Objects]
    assert ids.count(couch_id) == 1
    assert len(room.Objects) == n_scene_objects + 1
    couch = room.find(couch_id)
    assert abs(couch.Pose.Position.X - (scene_couch.Pose.Position.X + 1.0)) < 1e-9
    seats = room.seats_of(couch_id)
    assert len(seats) == 3 and scene_seats >= 1
    assert all(abs(s.Pose.Position.X - couch.Pose.Position.X) < 1.0 for s in seats)
    assert len(room.seats_of("fused:1")) == 1
    assert len({s.Id for s in room.Seats}) == len(room.Seats)
    server.close()
