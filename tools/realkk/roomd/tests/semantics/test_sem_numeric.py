"""Pure-numpy replacements for scipy (roomd's project env has no scipy)."""
import itertools

import numpy as np

from roomd.semantics.geometry import connected_labels, linear_assignment


def _brute(cost):
    n, m = cost.shape
    best = None
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            s = sum(cost[i, c] for i, c in enumerate(cols))
            best = s if best is None or s < best else best
    else:
        for rows in itertools.permutations(range(n), m):
            s = sum(cost[r, j] for j, r in enumerate(rows))
            best = s if best is None or s < best else best
    return best


def test_linear_assignment_matches_brute_force():
    rng = np.random.default_rng(3)
    for shape in [(1, 1), (3, 3), (4, 2), (2, 5), (5, 5), (6, 4)]:
        for _ in range(20):
            cost = rng.uniform(0, 10, shape)
            rows, cols = linear_assignment(cost)
            assert len(rows) == min(shape) == len(set(rows)) == len(set(cols))
            assert abs(cost[rows, cols].sum() - _brute(cost)) < 1e-9


def test_linear_assignment_with_gated_pairs():
    big = 1e6
    cost = np.array([[1.0, big], [big, big], [big, 2.0]])
    rows, cols = linear_assignment(cost)
    pairs = {(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < big}
    assert pairs == {(0, 0), (2, 1)}


def test_linear_assignment_empty():
    rows, cols = linear_assignment(np.zeros((0, 3)))
    assert len(rows) == 0 and len(cols) == 0


def test_connected_labels():
    # 0-1-2 chain, 3 alone, 4-5
    r = np.array([0, 1, 4])
    c = np.array([1, 2, 5])
    labels, n = connected_labels(6, r, c)
    assert n == 3
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5]
    assert len({labels[0], labels[3], labels[4]}) == 3


def test_connected_labels_long_snake():
    n = 5000
    r = np.arange(n - 1)[::-1].copy()
    labels, count = connected_labels(n, r, r + 1)
    assert count == 1
