# roomd — realkk room service

roomd sits between the ALVR server and the KKS realkk plugin (plan v4, stages 1–3):

```
ALVR server ──TCP 127.0.0.1:9944──> roomd ──TCP 127.0.0.1:9945──> KKS realkk plugin
  (client)        relay viewer       │        plugin server          (client, reconnects every 2 s)
                                     ├─ scene.py   MSG_ROOM_SNAPSHOT → RoomModel v2 (SceneApi)
                                     ├─ FusionSink depth frames → fused geometry (fusion / semantics slices)
                                     └─ service.py ROOM_MODEL, STATUS, NAV_HEIGHTMAP, MESH_CHUNK, SCENE_MESH
```

The interface is defined by `realkk/docs/CONTRACT-roomd.md`; this package implements it.

## Setup and commands

Run everything from `tools/realkk/roomd/` with uv (Python 3.12, own `.venv`):

```powershell
uv sync --extra synth             # numpy, lz4, warp-lang 1.18.0; open3d 0.20.0 for synth/tests
uv run pytest                     # all roomd tests (Open3D tests skip without the extra)

# live: roomd is the only 9944 viewer (close depth_listener.py / other viewers first)
uv run python -m roomd                                   # 9944 in, 9945 out
uv run python -m roomd --record D:\AIP\Data\realkk\taps\live.rktap   # also record the relay
# offline: read a tap instead of listening on 9944
uv run python -m roomd --tap-in room.rktap --tap-speed 1 --dump-model model.json
# synthetic session (60 s, 2 x 320x320 views at 10 fps) + ground truth + raw snapshot
uv run python -m roomd.synth room.rktap --gt room.gt.json --snapshot room.snapshot.bin
uv run python ..\tap_inspect.py room.rktap
```

`python -m roomd` options: `--relay-port 9944`, `--plugin-port 9945`, `--host 127.0.0.1`,
`--tap-in FILE` (no 9944 socket), `--tap-speed S` (0 = as fast as possible), `--tap-loop N`
(0 = forever), `--record FILE`, `--fusion noop|module:factory`, `--room-id stage`,
`--seconds S`, `--dump-model FILE`.

`python -m roomd.synth` options: `--seconds 60 --fps 10 --seed 1 --size 320`, `--like REAL.rktap`
(copy size, near/far, FOV and baseline from a real recording), `--clean` (no noise),
`--no-fake`, `--no-snapshot`. Without Open3D it exits with code 2 and says to run
`uv sync --extra synth`.

## Modules

| module | role |
|---|---|
| `protocol.py` | encode/decode of every CONTRACT message; reuses `depth_listener.parse_depth_v2/decode_d16`, `rktap` and `tap_inspect.is_fake_frame` from `tools/realkk` via `_legacy.py` (no second copy) |
| `coords.py` | quaternion math, OpenXR → Unity flip: position `(x, y, -z)`, rotation `(-qx, -qy, qz, qw)` |
| `model.py` | RoomModel v2 dataclasses (C# field names), JSON read/write, v1 → v2 migration, `generate_seats` (= Core `SeatGenerator`) |
| `scene.py` | MSG_ROOM_SNAPSHOT → RoomModel v2 + `ScenePrior` (GLOBAL_MESH in Unity space) |
| `sink.py` | `FusionSink` interface and no-op implementation, `FusionOutputs`, `ScenePrior` |
| `io/relay.py` | 9944 viewer: accept ALVR, send STREAM_CONTROL (depth on, camera off) + ROOM_REQUEST(0), reader thread → `RelayInbox` (newest depth only, camera dropped, others queued), optional `.rktap` recording |
| `io/plugin_server.py` | 9945 server: many clients, per-client writer thread and queue (a stalled client is dropped at 64 MB), latest state replayed to new clients; per-client reader thread: PLUGIN_HELLO logged, ROOM_RECLASSIFY queued for the main loop (`take_reclassify`) |
| `io/tapsource.py` | `--tap-in`: plays a tap into the same `RelayInbox` |
| `service.py` | main loop: fake-frame drop, `integrate`, snapshot conversion, SCENE_MESH publishing, model revisioning, 1 Hz output poll / heartbeat, plugin ROOM_RECLASSIFY → `sink.reclassify` |
| `synth/` | synthetic room (Unity frame), Scene snapshot builder, Open3D renderer, noise, scripted session → tap + ground truth |
| `fusion/`, `semantics/` | owned by the fusion and semantics slices |

## Contract mapping

| CONTRACT item | where |
|---|---|
| frame `u32 type, u32 len, payload` | `protocol.pack_frame / unpack_frame / read_frame` |
| 3 MSG_DEPTH_FRAME_V2 | `decode_depth_v2` → `DepthFrame` (`views[i].pose_xr` p-first, `view_z(i)`, `is_fake()`), `encode_depth_v2` |
| 4 MSG_ROOM_SNAPSHOT | `RoomSnapshot`, `SnapshotMesh`, `encode/decode_room_snapshot`; `header_size` = offset of `json_len` (40), decoder skips to it |
| 5 MSG_PLAYSPACE_CHANGED | `encode/decode_playspace_changed`; service forwards to `FusionSink.on_playspace_changed` |
| 100 MSG_STREAM_CONTROL, 101 MSG_ROOM_REQUEST | `encode/decode_stream_control`, `encode/decode_room_request`; sent by `RelayViewer` on connect |
| 9945 1 ROOM_MODEL | `model.room_to_json`; on content change and every second (once a snapshot or fused geometry exists) |
| 9945 2 NAV_HEIGHTMAP | `NavHeightmap` (grids `[iz, ix]`), forwarded when `FusionOutputs.nav_revision` changes |
| 9945 3 MESH_CHUNK | `MeshChunk` (`vcount = 0` ⇒ removed), forwarded as the sink reports them |
| 9945 4 STATUS | `RoomService.status()`, exactly the CONTRACT keys, 1 Hz |
| 9945 5 SCENE_MESH | `SceneMeshPart`, `encode/decode_scene_mesh_part`, `split_scene_mesh`; `RoomService.publish_scene_mesh` on each snapshot (see below) |
| 9945 101 PLUGIN_HELLO | `PluginServer.hellos` |
| 9945 102 ROOM_RECLASSIFY | see below |
| RoomModel v2 | `model.py`; reading v1 gives `Source=Manual, Confidence=1, Locked=true, State=Present`, seats `Available` |

## Scene conversion rules (`scene.py`)

- FLOOR → `FloorY` (lowest floor anchor). WALL_FACE / INVISIBLE_WALL_FACE → `Walls` segment (plane mid-height,
  local X ends) and a `Wall` box 5 cm thick behind the face, front = into the room.
- Volumes (`bbox3d`): 8 corners → Unity; the box axis closest to vertical is the height; Pose = bottom-face
  centre; front = the horizontal box axis pointing most away from the nearest wall (MRUK).
- COUCH / BED: sittable; seat height = anchor (2D plane) height above the box bottom, defaults without a plane
  0.45 m (couch) / box height (bed). COUCH whose plane aspect ratio is in (0.5, 2) becomes `Kind=Chair` with one
  seat (MRUK `CalculateSeatPoses`); otherwise one seat per 0.6 m (`generate_seats`).
- TABLE → Table; STORAGE / SCREEN / LAMP / PLANT / OTHER / unknown labels with a volume → Other;
  CEILING, DOOR_FRAME, WINDOW_FRAME, WALL_ART, GLOBAL_MESH → no object.
- Ids `scene:<anchor uuid>`, `Source=SceneApi`, `Locked=false`, `Confidence=1`.

## Re-classification on request (ROOM_RECLASSIFY, plugin → roomd)

Automatic objects keep their kind once it is stable (semantics README, "Kind lock"); the plugin
asks for a new decision with type 102, JSON `{"ids": ["fused:3", "scene:<uuid>"]}`. A missing,
`null` or empty `ids` (or an empty payload) means every automatic object.

- The 9945 reader thread decodes it (`protocol.decode_room_reclassify`; a bad payload is logged
  and ignored) and queues it; it never touches the writer queues, so publishing is not blocked.
- The main loop drains the queue (`RoomService.process_plugin`), merges every queued request
  into one (the union of the ids; any "all" request makes it all), calls
  `FusionSink.reclassify(ids)` once (unknown ids ignored; the no-op sink unlocks nothing), logs
  `roomd: reclassify <ids|all> -> N object(s) unlocked`, then polls the outputs once and
  publishes STATUS and the ROOM_MODEL at once. The kinds then follow the classifier again and re-lock
  when stable; a changed kind reaches the plugin in the next ROOM_MODEL (content change).

Persistence: roomd keeps no state across restarts (tracks, ids, locked kinds); `--dump-model`
only writes the last model on exit. After a restart or a recenter the kinds re-lock after
~5 observations (~5 s).

## Quest scene mesh to the plugin (SCENE_MESH)

The snapshot's GLOBAL_MESH (`scene.scene_mesh_unity`: Unity stage, winding reversed) is the fusion prior and is
also sent to the plugin for the room debug view:

- `split_scene_mesh` sorts triangles by the 1 m (x, z, y) cell of their centroid and cuts runs of at most
  `SCENE_MESH_MAX_TRIANGLES` (10 000); each part carries only the vertices it uses. No mesh = one clear part
  (`part_count = 0`).
- `revision` is roomd's own counter; a snapshot whose mesh hashes the same as the last one (ALVR re-sends on
  recenter and viewer connect) publishes nothing.
- Parts are remembered per index (`("scene_mesh", i)`) and the previous revision's extra parts are forgotten, so a
  plugin that connects later receives exactly the current revision.
- Plugins without type 5 ignore it (unknown types are skipped).

## FusionSink (for the fusion and semantics slices)

```python
class FusionSink:                         # roomd.sink
    def integrate(self, frame): ...       # protocol.DepthFrame, newest real frame only (OpenXR stage)
    def snapshot_outputs(self): ...       # -> FusionOutputs, polled about once per second
    def set_scene_prior(self, prior): ... # ScenePrior(snapshot_id, floor_y, objects, walls, mesh_vertices, mesh_triangles)
    def on_playspace_changed(self, recenter_pose): ...
    def reclassify(self, ids): ...        # ROOM_RECLASSIFY, ids list or None = all -> ids unlocked
    def close(self): ...

@dataclass
class FusionOutputs:                      # all Unity left-handed stage space; None = not produced
    map_revision: int = 0
    floor_y: float = None; floor_rms: float = None
    objects: list = None                  # model.RoomObject, Source="Fused", ids "fused:<n>"
    seats: list = None                    # model.SeatPoint, ids "<objectId>#<i>"
    walls: list = None                    # model.Wall
    nav_heightmap: NavHeightmap = None; nav_revision: int = 0
    mesh_chunks: list = []                # MeshChunk changed since the previous call (drained)
```

All calls come from the single main-loop thread. Load an implementation with `--fusion package.module:factory`.
roomd owns `RoomModel.Revision` and each object's `Revision / CreatedUtc / UpdatedUtc` (it overwrites them on the
objects it receives); fused floor/walls replace the scene ones when present, objects and seats are concatenated.

## Synthetic data (`synth/`)

Room 5 × 4 m (Unity frame): table 0.72, chair (seat 0.45, back 0.85, four legs) carrying a clothes pile
(its seat is `Blocked` in the ground truth), 3-seat couch, bed, coffee table, cabinet, single-seat armchair.
Head trajectory: scan, watch the chair slide +1 m, a person walks across, the user sits on the middle couch seat
and stands up again. Noise: σ = 0.0025·z², edge holes and flying pixels, dropouts, hand blobs, range cut at 5 m,
D16 quantisation (`d = 1 − n/z` for far = ∞), every 97th frame an all-0x80 fake frame. The tap holds the
snapshot, all depth frames and markers; the ground-truth JSON holds furniture OBBs (RoomObject JSON), seats,
events and fake-frame indices.

## Known limits

- No automatic reconnect logic is needed on 9944 (ALVR retries every 5 s); a second connection replaces the first.
- Scene objects keep `State=Present` and seats `Available`; Moving / Missing / Blocked come from fusion.
- `--tap-in` with `--tap-speed 0` lets the inbox collapse depth frames (only the newest is integrated), like a slow
  consumer; use speed ≥ 1 to integrate every frame.
- The synthetic Scene snapshot uses the Meta plane / volume conventions as understood from R13; check them
  against the first real snapshot from the ALVR fork.
