import json
import math
import sys

import numpy as np
import pytest

import tap_inspect
from roomd import coords
from roomd import protocol as P
from roomd.synth import render
from roomd.synth.render import DepthCamera, SynthUnavailable
from roomd.synth.scenario import Scenario, Session, generate

open3d = pytest.importorskip("open3d")


def test_missing_open3d_is_reported_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(SynthUnavailable, match="uv sync --extra synth"):
        render.require_open3d()


def test_depth_is_view_z_of_a_wall():
    """Camera at the origin looking down OpenXR -Z at a wall 2 m away: every pixel z = 2."""
    cam = DepthCamera(width=16, height=12)
    r = render.Renderer(cam)
    wall = np.array([(x, y, z) for x in (-5, 5) for y in (-5, 5) for z in (1.9, 2.0)])  # Unity z = +2
    r.set_geometry([wall])
    z = r.render_view(0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert z.shape == (12, 16)
    assert z == pytest.approx(np.full((12, 16), 1.9), abs=1e-4)


def _unproject(frame, view):
    v = frame.views[view]
    z = frame.view_z(view)
    fx, fy, cx, cy = v.intrinsics
    h, w = z.shape
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ok = np.isfinite(z)
    cam = np.stack([(uu - cx) / fx * z, -(vv - cy) / fy * z, -z], axis=-1)[ok]
    return coords.flip_position(coords.transform_points(v.pose_xr, cam))


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    sc = Scenario(seconds=6.0, fps=5.0, seed=3, camera=DepthCamera(width=48, height=40), fake_every=7)
    gt = generate(sc, d / "s.rktap", d / "s.gt.json", d / "snap.bin")
    return sc, gt, d


def test_tap_is_readable_by_tap_inspect(small_run):
    sc, gt, d = small_run
    s = tap_inspect.summarize(d / "s.rktap")
    assert s["depth_frames"] == 30 == gt["frames"]
    assert s["fake_frames"] == len(gt["fake_frame_indices"]) == 4
    assert s["fake_frame_indices"] == gt["fake_frame_indices"]
    assert s["by_type"][P.MSG_ROOM_SNAPSHOT] == 1
    assert [m["label"] for m in s["markers"]][:3] == ["scan_start", "chair_move_start", "chair_move_end"]
    assert s["first_depth"]["width"] == 48 and s["first_depth"]["height"] == 80
    assert s["depth_fps"] == pytest.approx(5.0, rel=1e-3)
    assert not s["truncated"]


def test_ground_truth_and_snapshot_file(small_run):
    sc, gt, d = small_run
    disk = json.loads((d / "s.gt.json").read_text(encoding="utf-8"))
    assert disk["frames"] == gt["frames"]
    chair0 = next(o for o in gt["objects_initial"] if o["Id"] == "chair")
    chair1 = next(o for o in gt["objects_final"] if o["Id"] == "chair")
    assert chair1["Pose"]["Position"]["X"] - chair0["Pose"]["Position"]["X"] == pytest.approx(1.0)
    states = {s["Id"]: s["State"] for s in gt["seats_initial"]}
    assert states["chair#0"] == "Blocked" and states["couch#1"] == "Available"
    assert any(e["type"] == "user_seated" and e["seat"] == "couch#1" for e in gt["events"])
    snap = P.decode_room_snapshot((d / "snap.bin").read_bytes())
    assert any(a["labels"] == "COUCH" for a in snap.scene["anchors"])


def test_clean_depth_back_projects_onto_room_geometry(tmp_path):
    """Unprojecting with the header's pose and intrinsics must land on the synthetic room:
    inside the walls, never below the floor, and the floor itself at y = 0."""
    sc = Scenario(seconds=6.0, fps=2.0, camera=DepthCamera(width=64, height=64), noisy=False, fake_every=0,
                  snapshot=False)
    path = tmp_path / "c.rktap"
    generate(sc, path)
    import rktap

    frames = [P.decode_depth_v2(r.payload) for r in rktap.TapReader(path) if r.msg_type == P.MSG_DEPTH_FRAME_V2]
    pts = np.concatenate([_unproject(f, v) for f in frames for v in (0, 1)])
    assert len(pts) > 1000
    assert pts[:, 1].min() > -0.11  # floor slab top at 0 (D16 quantisation ~ mm)
    assert np.abs(pts[:, 0]).max() < 2.52 and np.abs(pts[:, 2]).max() < 2.02 and pts[:, 1].max() < 2.52
    # above the furniture and away from the other walls, points near x = +-2.5 are on the wall
    wall = pts[(np.abs(pts[:, 0]) > 2.4) & (pts[:, 1] > 1.0) & (pts[:, 1] < 2.4) & (np.abs(pts[:, 2]) < 1.9)]
    assert len(wall) > 50 and np.abs(np.abs(wall[:, 0]) - 2.5).max() < 0.02
    floor = pts[(pts[:, 1] < 0.05) & (np.abs(pts[:, 0]) < 2.4) & (np.abs(pts[:, 2]) < 1.9)]
    assert len(floor) > 100 and np.percentile(np.abs(floor[:, 1]), 90) < 0.005  # rest: furniture legs


def test_session_head_sits_on_the_couch():
    sess = Session(Scenario(seconds=60.0))
    seated = sess.head_at(sum(sess.t_sit[1:3]) / 2)
    seat = sess.sit_seat.Pose.Position
    assert seated[1] == pytest.approx(seat.Y + 0.72)
    assert math.hypot(seated[0] - seat.X, seated[2] - seat.Z) < 0.15
    assert sess.room_at(sess.t_walk[0] + 1).find("person") is not None
