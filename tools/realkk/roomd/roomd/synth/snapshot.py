"""Synthetic MSG_ROOM_SNAPSHOT for a SynthRoom (pure numpy, no Open3D).

Anchors follow the Meta Scene conventions in OpenXR (right-handed) stage space:
planes lie in their local XY with +Z as normal (floor normal up, wall normal into the
room); volumes have local +Z up, the pose sits on the top face (tables) or on the seat
plane (couches, beds), bbox3d spans local z downwards to the floor. Furniture without a
Scene label (chairs, the clothes pile) is left out, as on a real Quest.
"""

import numpy as np

from .. import coords
from .. import protocol as P
from .room import boxes_to_mesh, synth_uuid

UP = np.array([0.0, 1.0, 0.0])


def _rot_from_axes(x_axis, z_axis):
    """XR quaternion whose local X and Z map to the given world axes (Y = Z x X)."""
    x = np.asarray(x_axis, float)
    z = np.asarray(z_axis, float)
    y = np.cross(z, x)
    return coords.matrix_to_quat(np.column_stack([x, y, z]))


def _anchor(name, labels, pose=None, bbox2d=None, bbox3d=None):
    return {"uuid": synth_uuid(name), "labels": labels,
            "pose": [float(v) for v in pose] if pose is not None else None,
            "bbox2d": [float(v) for v in bbox2d] if bbox2d is not None else None,
            "boundary2d": None,
            "bbox3d": [float(v) for v in bbox3d] if bbox3d is not None else None}


def _xr(p):
    return coords.flip_position(np.asarray(p, float))


def furniture_anchor(f, floor_y):
    w, h, d = f.size
    pose_u = f.pose()
    ax_x = _xr(pose_u.Rotation.rotate((1.0, 0.0, 0.0)))
    q = _rot_from_axes(ax_x, UP)
    if f.label in ("COUCH", "BED"):
        anchor_h = f.seat_height  # anchor on the seat plane
        plane = (-w / 2, -d * 0.375, w, d * 0.75)
    else:
        anchor_h = h  # anchor on the top face
        plane = (-w / 2, -d / 2, w, d) if f.label == "TABLE" else None
    pos = _xr((f.center_xz[0], floor_y + anchor_h, f.center_xz[1]))
    bbox3d = (-w / 2, -d / 2, -anchor_h, w, d, h)
    return _anchor(f.name, f.label, tuple(pos) + q, plane, bbox3d)


def wall_anchor(name, a, b, normal, floor_y, height, label="WALL_FACE"):
    a_xr = _xr((a[0], 0, a[1]))
    b_xr = _xr((b[0], 0, b[1]))
    n_xr = _xr((normal[0], 0, normal[1]))
    x_axis = (b_xr - a_xr) / np.linalg.norm(b_xr - a_xr)
    if np.cross(n_xr, x_axis)[1] < 0:  # local +Y must point up
        x_axis = -x_axis
    length = float(np.linalg.norm(b_xr - a_xr))
    mid = (a_xr + b_xr) / 2 + np.array([0.0, floor_y + height / 2, 0.0])
    return _anchor(name, label, tuple(mid) + _rot_from_axes(x_axis, n_xr),
                   (-length / 2, -height / 2, length, height))


def build_snapshot(room, snapshot_id=1, mesh_anchor_pose=(0.2, 0.0, -0.1, 0.0, 0.0871557, 0.0, 0.9961947)):
    """RoomSnapshot with floor, ceiling, walls, labelled furniture, decorations that roomd
    must ignore, and a GLOBAL_MESH of the static geometry (labelled furniture + shell)."""
    anchors = []
    cx = (room.x_min + room.x_max) / 2
    cz = (room.z_min + room.z_max) / 2
    wx = room.x_max - room.x_min
    wz = room.z_max - room.z_min
    floor_q = _rot_from_axes((1, 0, 0), UP)
    anchors.append(_anchor("floor", "FLOOR", tuple(_xr((cx, room.floor_y, cz))) + floor_q,
                           (-wx / 2, -wz / 2, wx, wz)))
    ceil_q = _rot_from_axes((1, 0, 0), -UP)
    anchors.append(_anchor("ceiling", "CEILING",
                           tuple(_xr((cx, room.floor_y + room.wall_height, cz))) + ceil_q,
                           (-wx / 2, -wz / 2, wx, wz)))
    wall_ids = []
    for name, a, b, n in room.walls():
        anchors.append(wall_anchor(name, a, b, n, room.floor_y, room.wall_height))
        wall_ids.append(synth_uuid(name))
    # decorations on the south wall (z_min): door, window, picture -> ignored by roomd
    for name, label, x, y, w, h in (("door", "DOOR_FRAME", -1.8, 1.0, 0.9, 2.0),
                                    ("window", "WINDOW_FRAME", 0.5, 1.5, 1.2, 1.0),
                                    ("picture", "WALL_ART", 1.8, 1.6, 0.6, 0.4)):
        a = _anchor(name, label, None, (-w / 2, -h / 2, w, h))
        n_xr = _xr((0, 0, 1))
        a["pose"] = [float(v) for v in tuple(_xr((x, y, room.z_min + 0.01))) + _rot_from_axes((1, 0, 0), n_xr)]
        anchors.append(a)
    static = room.shell_boxes()
    for f in room.furniture:
        if f.label is None or f.attached_to is not None:
            continue
        anchors.append(furniture_anchor(f, room.floor_y))
        static += f.world_boxes()
    anchors.append(_anchor("global_mesh", "GLOBAL_MESH", mesh_anchor_pose))

    verts_u, tris = boxes_to_mesh(static)
    verts_xr = _xr(verts_u)
    # store the mesh in the GLOBAL_MESH anchor's local frame
    rot = coords.quat_to_matrix(mesh_anchor_pose[3:])
    local = (verts_xr - np.asarray(mesh_anchor_pose[:3])) @ rot
    mesh = P.SnapshotMesh(P.uuid_bytes(synth_uuid("global_mesh")), tuple(mesh_anchor_pose),
                          local.astype(np.float32), tris[:, ::-1].reshape(-1).astype(np.uint32))
    scene = {"rooms": [{"uuid": synth_uuid("room"), "floor": synth_uuid("floor"),
                        "ceiling": synth_uuid("ceiling"), "walls": wall_ids}],
             "anchors": anchors}
    return P.RoomSnapshot(snapshot_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), scene, [mesh])
