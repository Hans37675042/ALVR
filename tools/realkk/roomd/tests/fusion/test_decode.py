import math

import numpy as np
import pytest

from roomd.fusion import decode as dec
from roomd.fusion.synthscene import SynthScene, encode_payload, look_at_xr, metric_payload

from conftest import BOX_MAX, BOX_MIN

Q = (0.1, 0.2, 0.3, math.sqrt(1 - 0.14))


def _quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)


def test_header_fields_and_unity_pose():
    p0, p1 = (0.5, 1.6, -2.0), (0.6, 1.5, -2.1)
    depth = [np.full((8, 6), 2.0, np.float32), np.full((8, 6), 3.0, np.float32)]
    payload = metric_payload(depth, poses_xr=[(p0, Q), (p1, Q)],
                             intrinsics=[(10.0, 11.0, 3.0, 4.5), (12.0, 13.0, 2.5, 4.0)],
                             near=0.1, far=math.inf, client_ts=123, server_ts=456)
    f = dec.decode_depth_frame(payload)
    assert f.client_ts_ns == 123 and f.server_ts_ns == 456
    assert len(f.views) == 2
    v0, v1 = f.views
    np.testing.assert_allclose(v0.position_xr, p0, atol=1e-6)
    np.testing.assert_allclose(v0.position_unity, (0.5, 1.6, 2.0), atol=1e-6)
    np.testing.assert_allclose(v1.position_unity, (0.6, 1.5, 2.1), atol=1e-6)
    np.testing.assert_allclose(v0.rotation_unity, (-0.1, -0.2, 0.3, Q[3]), atol=1e-6)
    assert (v1.fx, v1.fy, v1.cx, v1.cy) == pytest.approx((12.0, 13.0, 2.5, 4.0))
    assert v0.depth.shape == (8, 6) and v1.depth.shape == (8, 6)
    np.testing.assert_allclose(v0.depth, 2.0, rtol=2e-3)
    np.testing.assert_allclose(v1.depth, 3.0, rtol=2e-3)
    np.testing.assert_allclose(f.head_position_unity, (0.55, 1.55, 2.05), atol=1e-6)


def test_unity_rotation_is_mirrored_xr_rotation():
    v = np.array([0.3, -0.7, 0.2])
    mirror = np.array([1.0, 1.0, -1.0])
    unity_q = dec.xr_to_unity_quat(Q)
    np.testing.assert_allclose(_quat_rotate(unity_q, v * mirror), _quat_rotate(Q, v) * mirror, atol=1e-6)


@pytest.mark.parametrize("far", [math.inf, 6.0])
def test_d16_roundtrip(far):
    z = np.linspace(0.3, 5.0, 200).astype(np.float32)
    back = dec.d16_to_metric(dec.metric_to_d16(z, 0.1, far), 0.1, far)
    assert np.all(np.abs(back - z) <= 2e-4 * z * z + 1e-4)


def test_d16_extremes_are_invalid():
    raw = np.array([0, 65535, 30000], dtype=np.uint16)
    out = dec.d16_to_metric(raw, 0.1, math.inf)
    assert out[0] == 0 and out[1] == 0 and out[2] > 0


@pytest.mark.parametrize("fmt", [0, 2])
def test_fake_frame_is_dropped(fmt):
    raw = np.full((16, 8), 0x8080, np.uint16)
    pose = ((0, 1.6, 0), (0, 0, 0, 1))
    payload = encode_payload(raw, [pose, pose], [(8.0, 8.0, 4.0, 4.0)] * 2, 0.1, math.inf, fmt=fmt)
    assert dec.decode_depth_frame(payload) is None


def test_truncated_payload_raises():
    payload = metric_payload([np.ones((4, 4), np.float32)] * 2, [((0, 0, 0), (0, 0, 0, 1))] * 2,
                             [(4.0, 4.0, 2.0, 2.0)] * 2, 0.1, math.inf, fmt=0)
    with pytest.raises(ValueError):
        dec.decode_depth_frame(payload[:-10])


def test_backprojection_hits_floor_and_box_top():
    scene = SynthScene(floor_y=0.0, boxes=[(BOX_MIN, BOX_MAX)])
    f = dec.decode_depth_frame(scene.frame_payload((0.5, 1.6, 0.2), (0.5, 0.0, 1.5)))
    for view in f.views:
        pts = dec.backproject_unity(view)
        assert len(pts) > 1000
        top = (np.abs(pts[:, 0] - 0.5) < 0.2) & (np.abs(pts[:, 2] - 1.5) < 0.2)
        assert top.sum() > 50
        assert np.abs(pts[top, 1] - 0.45).max() < 0.005
        low = pts[:, 1] < 0.2
        outside = low & ((np.abs(pts[:, 0] - 0.5) > 0.3) | (np.abs(pts[:, 2] - 1.5) > 0.3))
        assert outside.sum() > 1000
        assert np.abs(pts[outside, 1]).max() < 0.005


def test_flip_rows_flag_restores_orientation():
    scene = SynthScene(floor_y=0.0, boxes=[(BOX_MIN, BOX_MAX)])
    upright = dec.decode_depth_frame(scene.frame_payload((0.5, 1.6, 0.2), (0.5, 0.0, 1.5)))
    flipped_payload = scene.frame_payload((0.5, 1.6, 0.2), (0.5, 0.0, 1.5), flip_rows=True)
    wrong = dec.decode_depth_frame(flipped_payload)
    right = dec.decode_depth_frame(flipped_payload, flip_rows=True)
    for a, b, c in zip(upright.views, right.views, wrong.views):
        np.testing.assert_array_equal(a.depth, b.depth)
        assert not np.array_equal(a.depth, c.depth)


def test_look_at_points_minus_z_at_target():
    q = look_at_xr((0, 1.6, 0), (0, 0, -2))
    fwd = _quat_rotate(q, (0, 0, -1))
    expect = np.array([0, -1.6, -2]) / math.hypot(1.6, 2)
    np.testing.assert_allclose(fwd, expect, atol=1e-6)
