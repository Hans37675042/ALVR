import math

import pytest

import rktap
import tap_inspect
from depth_listener import parse_depth_v2
from synth import depth_payload, fake_depth_payload

pytest.importorskip("lz4")


def test_fake_frame_detected_lz4():
    h, data = parse_depth_v2(fake_depth_payload(fmt=2))
    assert tap_inspect.is_fake_frame(h, data) is True


def test_fake_frame_detected_raw():
    h, data = parse_depth_v2(fake_depth_payload(fmt=0))
    assert tap_inspect.is_fake_frame(h, data) is True


def test_real_frame_not_fake():
    h, data = parse_depth_v2(depth_payload(fmt=2))
    assert tap_inspect.is_fake_frame(h, data) is False


def test_partial_0x80_not_fake():
    pixels = [0x8080] * 15 + [0x1234]
    h, data = parse_depth_v2(depth_payload(fmt=2, pixels=pixels))
    assert tap_inspect.is_fake_frame(h, data) is False


def _make_tap(path):
    base = 1_700_000_000_000_000_000
    step = 100_000_000  # 10 fps
    with rktap.TapWriter(path, {"feeds": {"depth": True, "camera": False}}) as w:
        for i in range(10):
            if i in (3, 7):
                payload = fake_depth_payload()
            else:
                payload = depth_payload(near=0.25, width=4, height=8, pixels=[i + 1] * 32)
            w.write(base + i * step, 3, payload)
            if i == 5:
                w.write(base + i * step + 1, rktap.MSG_MARKER,
                        rktap.marker_payload("chair moved", base + i * step))
        w.write(base + 450_000_000, 2, b"cam")
    return base


def test_summary(tmp_path):
    path = tmp_path / "s.rktap"
    _make_tap(path)
    s = tap_inspect.summarize(path)
    assert s["messages"] == 12
    assert s["by_type"] == {3: 10, 2: 1, rktap.MSG_MARKER: 1}
    assert s["duration_s"] == pytest.approx(0.9)
    assert s["depth_fps"] == pytest.approx(10.0)
    first = s["first_depth"]
    assert (first["width"], first["height"]) == (4, 8)
    assert first["near_z"] == pytest.approx(0.25)
    assert math.isinf(first["far_z"])
    assert first["view_poses"][0][5] == pytest.approx(1.6)
    assert s["fake_frames"] == 2
    assert s["fake_frame_indices"] == [3, 7]
    assert len(s["markers"]) == 1
    assert s["markers"][0]["label"] == "chair moved"
    assert s["markers"][0]["t_s"] == pytest.approx(0.5)
    assert s["truncated"] is False


def test_main_prints(tmp_path, capsys):
    path = tmp_path / "p.rktap"
    _make_tap(path)
    tap_inspect.main([str(path)])
    out = capsys.readouterr().out
    assert "fake" in out.lower()
    assert "chair moved" in out
