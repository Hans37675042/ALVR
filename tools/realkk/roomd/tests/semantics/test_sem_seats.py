"""Seat generation must match RealKK.Core SeatGenerator (ids, count, spacing, pose)."""
import math

import pytest

from roomd.semantics import generate_seats, yaw_quat


def _xyz(seat):
    p = seat["Pose"]["Position"]
    return p["X"], p["Y"], p["Z"]


def test_couch_three_seats_along_local_x():
    seats = generate_seats("fused:1", (1.0, 0.0, 2.0), 90.0, (1.8, 0.8, 0.9), 0.45, source="Fused")
    assert [s["Id"] for s in seats] == ["fused:1#0", "fused:1#1", "fused:1#2"]
    # local +X at yaw 90 is world (0, 0, -1); offsets -0.6, 0, +0.6
    expected = [(1.0, 0.45, 2.6), (1.0, 0.45, 2.0), (1.0, 0.45, 1.4)]
    for s, e in zip(seats, expected):
        assert _xyz(s) == pytest.approx(e, abs=1e-6)
        assert s["ObjectId"] == "fused:1"
        assert s["Height"] == pytest.approx(0.45)
        r = s["Pose"]["Rotation"]
        q = yaw_quat(90.0)
        assert (r["X"], r["Y"], r["Z"], r["W"]) == pytest.approx(q, abs=1e-6)
        assert s["State"] == "Available" and s["Source"] == "Fused" and s["Enabled"] is True
    assert yaw_quat(90.0) == pytest.approx((0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)))


def test_spread_along_local_z_when_depth_is_longer():
    seats = generate_seats("o", (0.0, 0.1, 0.0), 0.0, (0.5, 0.8, 1.3), 0.45)
    assert len(seats) == 2
    assert _xyz(seats[0]) == pytest.approx((0.0, 0.55, -0.325), abs=1e-6)
    assert _xyz(seats[1]) == pytest.approx((0.0, 0.55, 0.325), abs=1e-6)


def test_exact_multiple_of_spacing_counts_full_seats():
    assert len(generate_seats("o", (0, 0, 0), 0.0, (1.2, 0.8, 0.9), 0.45)) == 2
    assert len(generate_seats("o", (0, 0, 0), 0.0, (1.8, 0.8, 0.9), 0.45)) == 3


def test_short_object_gets_one_centred_seat():
    seats = generate_seats("scene:u", (2.0, 0.0, 1.0), 0.0, (0.45, 0.9, 0.45), 0.47)
    assert [s["Id"] for s in seats] == ["scene:u#0"]
    assert _xyz(seats[0]) == pytest.approx((2.0, 0.47, 1.0), abs=1e-6)


def test_equal_sides_spread_along_x():
    seats = generate_seats("o", (0, 0, 0), 0.0, (1.2, 0.8, 1.2), 0.4)
    assert [round(_xyz(s)[0], 6) for s in seats] == [-0.3, 0.3]
