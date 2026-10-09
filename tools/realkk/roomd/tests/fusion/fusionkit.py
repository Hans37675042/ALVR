"""Shared scene and scanning helpers for the fusion tests."""

import math

from roomd.fusion.synthscene import SynthScene

# Box used across the fusion tests (Unity stage coordinates, metres). It sits entirely in
# one 1 m chunk column (x 0..1, z 1..2) so its removal empties that chunk.
BOX_MIN = (0.25, 0.0, 1.25)
BOX_MAX = (0.75, 0.45, 1.75)
BOX_CENTER = (0.5, 0.225, 1.5)
HEAD_START = (0.0, 1.6, 0.0)


def scan_poses(n=24, center=BOX_CENTER, radius=1.6, height=1.6):
    """Eye/target pairs on an arc in front of the box, looking at and around it."""
    poses = []
    for i in range(n):
        a = math.radians(-70 + 140 * i / max(n - 1, 1))
        eye = (center[0] + radius * math.sin(a), height, center[2] - radius * math.cos(a))
        jitter = 0.4 * math.sin(3 * a)
        target = (center[0] + jitter, 0.15, center[2] + 0.3 * math.cos(2 * a))
        poses.append((eye, target))
    return poses


def scan(fusion, scene, poses, **render):
    results = []
    for eye, target in poses:
        results.append(fusion.integrate(scene.frame_payload(eye, target, **render)))
    return results


def box_scene(with_box=True):
    return SynthScene(floor_y=0.0, boxes=[(BOX_MIN, BOX_MAX)] if with_box else [])


def first_frame(fusion, scene):
    """Anchor the volume at HEAD_START, like a headset starting at the stage origin."""
    assert fusion.integrate(scene.frame_payload(HEAD_START, (0.0, 0.0, 1.2)))
