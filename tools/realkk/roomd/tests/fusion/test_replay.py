import json

from roomd.fusion import replay


def test_replay_synthetic_tap(tmp_path):
    tap = tmp_path / "synth.rktap"
    replay.write_synthetic_tap(tap, frames=20)
    out = tmp_path / "out"
    stats = replay.main([str(tap), "--out", str(out)])
    assert (out / "mesh.ply").stat().st_size > 1000
    png = (out / "heightmap.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    saved = json.loads((out / "stats.json").read_text())
    assert saved["frames"] == stats["frames"] == 20
    assert saved["fakeFrames"] == 1  # write_synthetic_tap inserts one fake readback frame
    assert abs(saved["floor"]["y"]) < 0.01


def test_replay_until_stops_at_tap_time(tmp_path):
    tap = tmp_path / "synth.rktap"
    replay.write_synthetic_tap(tap, frames=20)  # 10 fps
    stats = replay.main([str(tap), "--out", str(tmp_path / "out"), "--until", "0.95"])
    assert stats["frames"] == 10
