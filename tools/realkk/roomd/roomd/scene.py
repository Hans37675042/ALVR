"""Quest Scene API snapshot (MSG_ROOM_SNAPSHOT) -> RoomModel v2 and the fusion prior.

Anchor poses arrive in OpenXR stage space (right-handed); everything produced here is Unity
left-handed stage space. Label mapping (R13): FLOOR -> FloorY; WALL_FACE and
INVISIBLE_WALL_FACE -> thin Wall box + Walls segment; COUCH / BED -> sittable, seat height
from the 2D plane; TABLE; STORAGE / SCREEN / LAMP / PLANT / OTHER (and unknown labels with a
volume) -> Other; CEILING, DOOR_FRAME, WINDOW_FRAME, WALL_ART are ignored; GLOBAL_MESH only
feeds the prior mesh. Fronts follow MRUK: a box faces away from its nearest wall, and a
COUCH whose seat aspect ratio lies in (0.5, 2) is a single-seat chair (CalculateSeatPoses).
"""

import math
from datetime import datetime, timezone

import numpy as np

from . import coords
from . import model as M
from .sink import ScenePrior

WALL_THICKNESS = 0.05
DEFAULT_SEAT_HEIGHT = {M.Kind.Couch: 0.45, M.Kind.Chair: 0.45}
IGNORED = {"CEILING", "DOOR_FRAME", "WINDOW_FRAME", "WALL_ART", "GLOBAL_MESH"}
WALL_LABELS = {"WALL_FACE", "INVISIBLE_WALL_FACE"}
FURNITURE = {"COUCH": M.Kind.Couch, "BED": M.Kind.Bed, "TABLE": M.Kind.Table}


def _now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _labels(anchor):
    return [s.strip().upper() for s in (anchor.get("labels") or "").split(",") if s.strip()]


def _pose(anchor):
    p = anchor.get("pose")
    return tuple(float(v) for v in p) if p else None


def _plane_corners_unity(pose, bbox2d):
    """4 corners (Unity) of a plane rectangle [x, y, w, h] in its anchor's local XY."""
    x, y, w, h = bbox2d
    local = np.array([[x, y, 0], [x + w, y, 0], [x + w, y + h, 0], [x, y + h, 0]], float)
    return coords.flip_position(coords.transform_points(pose, local))


def _box_corners_unity(pose, bbox3d):
    ox, oy, oz, w, h, d = bbox3d
    local = np.array([(ox + i * w, oy + j * h, oz + k * d) for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                     float)
    return coords.flip_position(coords.transform_points(pose, local))


def _box_axes(corners):
    """World edge vectors of a box given its 8 corners in (i, j, k) order."""
    c = corners
    return [c[4] - c[0], c[2] - c[0], c[1] - c[0]]


def _wall_from_anchor(anchor):
    pose = _pose(anchor)
    if pose is None or not anchor.get("bbox2d"):
        return None
    x, y, w, h = anchor["bbox2d"]
    mid_y = y + h / 2
    ends = coords.flip_position(coords.transform_points(pose, np.array([[x, mid_y, 0], [x + w, mid_y, 0]])))
    corners = _plane_corners_unity(pose, anchor["bbox2d"])
    normal = coords.flip_position(coords.transform_points(pose, np.array([[0, 0, 1.0]]))
                                  - coords.transform_points(pose, np.zeros((1, 3))))[0]
    normal[1] = 0.0
    n = np.linalg.norm(normal)
    if n < 1e-6:
        return None
    normal /= n
    seg = M.Wall((float(ends[0][0]), float(ends[0][2])), (float(ends[1][0]), float(ends[1][2])))
    return seg, corners, normal


def _wall_object(anchor, corners, normal, now):
    y0 = float(corners[:, 1].min())
    height = float(corners[:, 1].max() - y0)
    a = corners[corners[:, 1].argsort()][:2]  # bottom edge
    length = float(np.linalg.norm((a[1] - a[0])[[0, 2]]))
    centre = (a[0] + a[1]) / 2 - normal * WALL_THICKNESS / 2
    yaw = coords.yaw_of_forward(normal)
    return M.RoomObject(Id="scene:%s" % anchor["uuid"], Kind=M.Kind.Wall,
                        Pose=M.Pose(M.Vec3(float(centre[0]), y0, float(centre[2])), M.Quat.from_yaw(yaw)),
                        Size=M.Vec3(length, height, WALL_THICKNESS), Sittable=False, Movable=False,
                        Source=M.SOURCE_SCENE, Label=anchor.get("labels") or "", Locked=False,
                        CreatedUtc=now, UpdatedUtc=now, LastSeenUtc=now)


def _nearest_wall_normal(centre_xz, walls):
    """Unit xz vector from the nearest wall segment towards centre_xz, or None."""
    best = None
    p = np.asarray(centre_xz, float)
    for w in walls:
        a = np.asarray(w.a, float)
        b = np.asarray(w.b, float)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        v = p - (a + t * ab)
        dist = np.linalg.norm(v)
        if dist > 1e-6 and (best is None or dist < best[0]):
            best = (dist, v / dist)
    return None if best is None else best[1]


def _furniture_object(anchor, kind, walls, now):
    pose = _pose(anchor)
    corners = _box_corners_unity(pose, anchor["bbox3d"])
    axes = _box_axes(corners)
    up_i = int(np.argmax([abs(a[1]) / max(np.linalg.norm(a), 1e-12) for a in axes]))
    horiz = [axes[i] for i in range(3) if i != up_i]
    height = float(corners[:, 1].max() - corners[:, 1].min())
    bottom_y = float(corners[:, 1].min())
    centre = corners.mean(axis=0)
    dirs = []
    for a in horiz:
        flat = np.array([a[0], 0.0, a[2]])
        length = float(np.linalg.norm(flat))
        dirs.append((flat / length, length))
    toward_room = _nearest_wall_normal((centre[0], centre[2]), walls)
    if toward_room is None:
        front, depth = dirs[1][0], dirs[1][1]
        width = dirs[0][1]
    else:
        n3 = np.array([toward_room[0], 0.0, toward_room[1]])
        best = max(((s * d, ln, i) for i, (d, ln) in enumerate(dirs) for s in (1, -1)),
                   key=lambda c: float(np.dot(c[0], n3)))
        front, depth, fi = best
        width = dirs[1 - fi][1]
    yaw = coords.yaw_of_forward(front)
    label = anchor.get("labels") or ""
    obj = M.RoomObject(Id="scene:%s" % anchor["uuid"], Kind=kind,
                       Pose=M.Pose(M.Vec3(float(centre[0]), bottom_y, float(centre[2])), M.Quat.from_yaw(yaw)),
                       Size=M.Vec3(width, height, depth), Source=M.SOURCE_SCENE, Label=label, Locked=False,
                       CreatedUtc=now, UpdatedUtc=now, LastSeenUtc=now)
    seat_count = None
    seat_height = None
    if kind in (M.Kind.Couch, M.Kind.Bed):
        plane = anchor.get("bbox2d")
        if plane:
            seat_height = pose[1] - bottom_y  # the 2D plane is the seat surface
            pw, ph = abs(plane[2]), abs(plane[3])
        else:
            seat_height = height if kind == M.Kind.Bed else min(DEFAULT_SEAT_HEIGHT[kind], height)
            pw, ph = width, depth
        if kind == M.Kind.Couch and ph > 0 and 0.5 < pw / ph < 2.0:
            obj.Kind = M.Kind.Chair
            seat_count = 1
        obj.Sittable = True
        obj.Movable = obj.Kind == M.Kind.Chair
    else:
        obj.Movable = True
    return obj, seat_height, seat_count


def room_from_snapshot(snap, room_id="stage", now_utc=None):
    """RoomModel v2 (FrameSource=Stage, Source=SceneApi) from a protocol.RoomSnapshot."""
    now = now_utc or _now_utc()
    anchors = snap.scene.get("anchors", [])
    room = M.RoomModel(Id=room_id, FrameSource=M.FRAME_STAGE)
    floor_ys = []
    wall_objects = []
    for a in anchors:
        labels = _labels(a)
        pose = _pose(a)
        if pose is None:
            continue
        if "FLOOR" in labels:
            floor_ys.append(pose[1])
        elif WALL_LABELS & set(labels):
            w = _wall_from_anchor(a)
            if w is not None:
                seg, corners, normal = w
                room.Walls.append(seg)
                wall_objects.append(_wall_object(a, corners, normal, now))
    if floor_ys:
        room.FloorY = float(min(floor_ys))
    room.Objects.extend(wall_objects)
    for a in anchors:
        labels = _labels(a)
        if _pose(a) is None or not a.get("bbox3d") or "FLOOR" in labels or WALL_LABELS & set(labels):
            continue
        kind = next((FURNITURE[lb] for lb in labels if lb in FURNITURE), None)
        if kind is None:
            if all(lb in IGNORED for lb in labels):
                continue
            kind = M.Kind.Other
        obj, seat_height, seat_count = _furniture_object(a, kind, room.Walls, now)
        room.Objects.append(obj)
        if obj.Sittable:
            room.Seats.extend(M.generate_seats(obj, seat_height, seat_count, source=M.SOURCE_SCENE))
    return room


def scene_mesh_unity(snap, global_only=True):
    """(vertices (N,3) float32, triangles (M,3) uint32) of the snapshot meshes in Unity stage
    space with the winding reversed for the handedness flip; global_only keeps only meshes
    whose anchor is labelled GLOBAL_MESH (all meshes when no anchor JSON matches)."""
    labels = {a.get("uuid"): _labels(a) for a in snap.scene.get("anchors", [])}
    meshes = [m for m in snap.meshes if not global_only or "GLOBAL_MESH" in labels.get(m.uuid, ["GLOBAL_MESH"])]
    verts, tris, base = [], [], 0
    for m in meshes:
        v = coords.flip_position(coords.transform_points(m.pose, np.asarray(m.vertices, float)))
        t = np.asarray(m.indices, np.uint32).reshape(-1, 3)[:, [0, 2, 1]]
        verts.append(v.astype(np.float32))
        tris.append(t + base)
        base += len(v)
    if not verts:
        return None, None
    return np.concatenate(verts), np.concatenate(tris).astype(np.uint32)


def prior_from_snapshot(snap, room=None):
    room = room or room_from_snapshot(snap)
    v, t = scene_mesh_unity(snap)
    return ScenePrior(snapshot_id=snap.snapshot_id, floor_y=room.FloorY, objects=list(room.Objects),
                      walls=list(room.Walls), mesh_vertices=v, mesh_triangles=t)
