"""roomd main loop: relay input -> FusionSink -> RoomModel v2 / nav / mesh -> plugin server.

Single-threaded by design: socket readers only fill a RelayInbox; this loop decodes the
newest depth frame, drops fake (all-0x80) frames, integrates, converts Scene API snapshots
and publishes on the 9945 server:
- ROOM_MODEL when its content changes, plus a 1 Hz heartbeat (only once something produced
  a room: a snapshot or fused geometry);
- STATUS at 1 Hz;
- NAV_HEIGHTMAP / MESH_CHUNK whenever the sink's outputs (polled at 1 Hz) carry new ones;
- SCENE_MESH (the snapshot's GLOBAL_MESH in parts) when a snapshot brings a different mesh.
"""

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import model as M
from . import protocol as P
from . import scene
from .sink import FusionOutputs

OUTPUT_PERIOD_S = 1.0
HEARTBEAT_PERIOD_S = 1.0
FPS_WINDOW_S = 2.0


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Stats:
    depth_frames: int = 0
    fake_frames: int = 0
    decode_errors: int = 0
    last_snapshot_id: int = None
    snapshots: int = 0
    playspace_changes: int = 0
    map_revision: int = 0
    integrate_ms: deque = field(default_factory=lambda: deque(maxlen=200))
    depth_times: deque = field(default_factory=deque)

    def depth_fps(self, now):
        while self.depth_times and now - self.depth_times[0] > FPS_WINDOW_S:
            self.depth_times.popleft()
        n = len(self.depth_times)
        if n < 2:
            return 0.0
        span = self.depth_times[-1] - self.depth_times[0]
        return (n - 1) / span if span > 0 else 0.0

    def integrate_p95(self):
        return float(np.percentile(list(self.integrate_ms), 95)) if self.integrate_ms else 0.0


class RoomService:
    def __init__(self, sink, plugin, relay=None, room_id="stage", log=print, clock=time.monotonic):
        self.sink = sink
        self.plugin = plugin
        self.relay = relay
        self.room_id = room_id
        self.log = log
        self.clock = clock
        self.stats = Stats()
        self.scene_model = None
        self.outputs = FusionOutputs()
        self.model = None
        self.revision = 0
        self._content = None
        self._obj_meta = {}  # id -> [key, revision, created, updated]
        self._model_json = None
        self._last_outputs = None
        self._last_heartbeat = None
        self._nav_revision = 0
        self._unknown_types = set()
        self.scene_mesh_revision = 0
        self._scene_mesh_key = None
        self._scene_mesh_parts = 0

    # --- input ---

    def handle_message(self, recv_ns, msg_type, payload):
        if msg_type == P.MSG_ROOM_SNAPSHOT:
            try:
                snap = P.decode_room_snapshot(payload)
            except (ValueError, KeyError, IndexError) as e:
                self.log("roomd: bad room snapshot: %s" % e)
                return
            room = scene.room_from_snapshot(snap, self.room_id)
            self.scene_model = room
            self.stats.last_snapshot_id = snap.snapshot_id
            self.stats.snapshots += 1
            prior = scene.prior_from_snapshot(snap, room)
            self.sink.set_scene_prior(prior)
            self.publish_scene_mesh(snap.snapshot_id, prior.mesh_vertices, prior.mesh_triangles)
            self.log("roomd: scene snapshot #%d: %d objects, %d seats, %d walls"
                     % (snap.snapshot_id, len(room.Objects), len(room.Seats), len(room.Walls)))
            self.update_model()
        elif msg_type == P.MSG_PLAYSPACE_CHANGED:
            pose = P.decode_playspace_changed(payload)
            self.stats.playspace_changes += 1
            self.sink.on_playspace_changed(pose)
            self.log("roomd: playspace changed, recenter pose %s" % (tuple(round(v, 3) for v in pose),))
        elif msg_type not in self._unknown_types:
            self._unknown_types.add(msg_type)
            self.log("roomd: ignoring relay message type %d" % msg_type)

    def handle_depth(self, recv_ns, payload):
        try:
            frame = P.decode_depth_v2(payload, recv_ns)
            fake = frame.is_fake()
        except Exception as e:  # corrupt payload or unsupported format
            self.stats.decode_errors += 1
            if self.stats.decode_errors == 1:
                self.log("roomd: depth decode error: %s" % e)
            return
        if fake:
            self.stats.fake_frames += 1
            return
        t0 = time.perf_counter()
        self.sink.integrate(frame)
        self.stats.integrate_ms.append((time.perf_counter() - t0) * 1000.0)
        self.stats.depth_frames += 1
        self.stats.depth_times.append(self.clock())

    # --- output ---

    def poll_outputs(self):
        out = self.sink.snapshot_outputs() or FusionOutputs()
        self.outputs = out
        self.stats.map_revision = out.map_revision
        if out.nav_heightmap is not None and out.nav_revision != self._nav_revision:
            self._nav_revision = out.nav_revision
            self.plugin.publish(P.NAV_HEIGHTMAP, P.encode_nav_heightmap(out.nav_heightmap), remember=True)
        for chunk in out.mesh_chunks or []:
            key = ("mesh",) + chunk.key
            if chunk.is_removed:
                self.plugin.forget(key)
                self.plugin.publish(P.MESH_CHUNK, P.encode_mesh_chunk(chunk))
            else:
                self.plugin.publish(P.MESH_CHUNK, P.encode_mesh_chunk(chunk), remember=key)
        self.update_model()

    def publish_scene_mesh(self, snapshot_id, vertices, triangles):
        """SCENE_MESH parts of the GLOBAL_MESH (Unity stage); skipped when the mesh equals the
        one already published (ALVR re-sends the snapshot on recenter and viewer connect).
        The parts replace the previous revision's in the plugin server's replay state."""
        h = hashlib.sha1()
        if vertices is not None and triangles is not None:
            h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
            h.update(np.ascontiguousarray(triangles, np.uint32).tobytes())
        key = h.hexdigest()
        if key == self._scene_mesh_key:
            return
        self._scene_mesh_key = key
        self.scene_mesh_revision += 1
        parts = P.split_scene_mesh(vertices, triangles, self.scene_mesh_revision, snapshot_id)
        for n in range(self._scene_mesh_parts):
            self.plugin.forget(("scene_mesh", n))
        self._scene_mesh_parts = len(parts)
        for part in parts:
            self.plugin.publish(P.SCENE_MESH, P.encode_scene_mesh_part(part), remember=("scene_mesh", part.part))
        tris = sum(len(p.indices) // 3 for p in parts)
        self.log("roomd: scene mesh r%d: %d triangles in %d part(s)"
                 % (self.scene_mesh_revision, tris, parts[0].part_count))

    def compose(self):
        """Scene API model + fused objects; None while nothing has produced a room yet."""
        sc = self.scene_model
        out = self.outputs
        if sc is None and out.objects is None and out.floor_y is None:
            return None
        room = M.RoomModel(Id=self.room_id, FrameSource=M.FRAME_STAGE)
        room.FloorY = out.floor_y if out.floor_y is not None else (sc.FloorY if sc else 0.0)
        room.FloorRms = out.floor_rms if out.floor_rms is not None else (sc.FloorRms if sc else 0.0)
        room.Walls = list(out.walls) if out.walls is not None else (list(sc.Walls) if sc else [])
        # The sink's version of an object wins over the Scene API copy with the same id
        # (semantics tracks scene objects and moves them with the fused geometry).
        own = {o.Id for o in out.objects or []}
        room.Objects = [o for o in (sc.Objects if sc else []) if o.Id not in own] + list(out.objects or [])
        room.Seats = ([s for s in (sc.Seats if sc else []) if s.ObjectId not in own]
                      + list(out.seats or []))
        now = utc_now()
        for o in room.Objects:
            d = M.object_to_dict(o)
            for k in ("Revision", "CreatedUtc", "UpdatedUtc", "LastSeenUtc"):
                d.pop(k)
            key = json.dumps(d, sort_keys=True)
            meta = self._obj_meta.get(o.Id)
            if meta is None:
                meta = self._obj_meta[o.Id] = [key, 0, o.CreatedUtc or now, o.UpdatedUtc or now]
            elif meta[0] != key:
                meta[0], meta[1], meta[3] = key, meta[1] + 1, now
            o.Revision, o.CreatedUtc, o.UpdatedUtc = meta[1], meta[2], meta[3]
            if o.LastSeenUtc is None:
                o.LastSeenUtc = now
        return room

    def update_model(self):
        room = self.compose()
        if room is None:
            return
        key = M.content_key(room)
        if key == self._content:
            room.Revision = self.revision
            self.model = room
            return
        self._content = key
        self.revision += 1
        room.Revision = self.revision
        self.model = room
        self._model_json = M.room_to_json(room).encode("utf-8")
        self.plugin.publish(P.ROOM_MODEL, self._model_json, remember=True)

    def status(self):
        now = self.clock()
        return {
            "utc": utc_now(),
            "depthFps": round(self.stats.depth_fps(now), 2),
            "fakeFrames": self.stats.fake_frames,
            "lastSnapshotId": self.stats.last_snapshot_id,
            "integrateMsP95": round(self.stats.integrate_p95(), 3),
            "mapRevision": self.stats.map_revision,
            "clients": self.plugin.client_count,
        }

    def heartbeat(self):
        if self._model_json is not None:
            self.plugin.publish(P.ROOM_MODEL, self._model_json, remember=True)
        self.plugin.publish(P.STATUS, P.encode_json(self.status()))

    def tick(self, force=False):
        now = self.clock()
        if force or self._last_outputs is None or now - self._last_outputs >= OUTPUT_PERIOD_S:
            self._last_outputs = now
            self.poll_outputs()
        if force or self._last_heartbeat is None or now - self._last_heartbeat >= HEARTBEAT_PERIOD_S:
            self._last_heartbeat = now
            self.heartbeat()

    def process(self, inbox):
        """Handle everything waiting in the inbox; True if anything was handled."""
        did = False
        for recv_ns, msg_type, payload in inbox.take_messages():
            self.handle_message(recv_ns, msg_type, payload)
            did = True
        depth = inbox.take_depth()
        if depth is not None:
            self.handle_depth(*depth)
            did = True
        return did

    def run(self, inbox, stop, until=None):
        """Loop until `stop` is set, or `until` is set and the inbox has been drained (then one
        final output poll and heartbeat are published)."""
        while not stop.is_set():
            inbox.wait(0.05)
            did = self.process(inbox)
            self.tick()
            if until is not None and until.is_set() and not did:
                if not self.process(inbox):
                    self.tick(force=True)
                    return
