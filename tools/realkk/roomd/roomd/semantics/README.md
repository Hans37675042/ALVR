# roomd.semantics

Furniture and seats from the fused map, tracked across frames (plan v4 slice S5,
R15 §1–2, R16 MRUK seat rules). Output is the `Objects` / `Seats` part of
RoomModel v2 (`docs/CONTRACT-roomd.md`); `roomd` I/O wraps it into ROOM_MODEL.

Coordinates: Unity left-handed stage space, +Y up, metres. Yaw in degrees,
clockwise from above; local +Z of an object is its front (seats face it).

## Plugging into roomd

```powershell
uv run python -m roomd --fusion roomd.semantics:create_sink
```

`SemanticsSink` wraps the fusion sink (`roomd.fusion:create_sink`): depth, mesh chunks and
the nav heightmap pass through unchanged; `snapshot_outputs()` adds `objects` / `seats` /
`walls` as `roomd.model` records. `FusionMapView` adapts TsdfFusion's queries
(`Obb(center, half_extents, yaw_rad)`, `FloorPlane`, `Heightmap`).

- Scene API objects from `set_scene_prior` become labels (kind priority, pose correction)
  and are published as tracked `scene:<uuid>` objects that follow the fused geometry;
  roomd's service replaces its own scene copy (and that object's seats) with the sink's
  version. `publish_scene_objects=False` publishes only `Source=Fused` objects.
- Walls: TsdfFusion's surface points are upward-only, so walls come from the scene prior.
- User head = midpoint of the two depth view poses of the latest frame.
- `on_playspace_changed` restarts tracking (the map is rebuilt in the new frame).
- Needs only numpy (assignment and connected components are implemented here).

## API

```python
from roomd.semantics import RoomSemantics, SemanticsParams, SceneLabel, Obb, detect

sem = RoomSemantics(SemanticsParams())
fragment = sem.update(map_view, t_unix, scene_labels=[...], user_head=(x, y, z))
sem.reject("fused:3")            # user deleted it -> Rejected tombstone
```

- `fragment` = `{Revision, FrameSource:"Stage", FloorY, FloorRms, Walls:[{a:[x,z], b:[x,z]}], Objects, Seats}`;
  every key and enum string as in the contract. `Revision` only increases when
  something published changes (object revision, seat state, floor > 1 cm, walls > 5 cm).
  `LastSeenUtc` refreshes every frame without bumping revisions; `Confidence` is
  republished only with a revision.
- `detect(map_view, params, scene_labels)` -> `Detection(floor_y, floor_rms, walls, candidates)`
  for a single frame (no tracking).
- `generate_seats(object_id, position, yaw, size, seat_h, source)` is a 1:1 port of
  `RealKK.Core.Room.SeatGenerator.ForObject` (ids `<objectId>#<i>`, 0.6 m spacing).

### Input: `MapView` (implemented by fusion)

| method | returns |
|---|---|
| `floor_plane()` | `(floor_y, rms)` |
| `heightmap()` | `HeightMap(cell, origin_x, origin_z, floor_y[h,w], top_y[h,w], flags[h,w])` or the same 6-tuple; row = z, column = x, origin = corner of cell (0, 0); flags = NAV_HEIGHTMAP bits |
| `surface_points(min_y, max_y, region=None)` | `(points Nx3, normals Nx3)`; `region` is an `Obb` footprint filter |
| `free_fraction(obb)` / `occupied_fraction(obb)` / `visible_fraction(obb)` | 0..1 of the box volume (visible = free + occupied) |

`Obb(cx, cz, yaw, sx, sz, y0, y1)` is a yaw-only box: footprint centre, full
extents along local X/Z, world y range. `SceneLabel(uuid, labels_csv, obb)` is a
Meta anchor already converted to Unity stage space.

`SyntheticRoom` implements `MapView` from boxes (builders `chair`, `couch`,
`table`, `coffee_table`, `stool`, `bed`, `storage`, `clothes_pile`, `user_torso`);
`set_hidden([...])` makes footprints unobserved.

## Pipeline

1. Raster: the map's heightmap (`raster_source="heightmap"`, default; TSDF upward points
   are too sparse for chair seats), or upward surface points on a 2 cm grid aligned to the
   points' lattice phase with single-cell dropouts filled (`"points"`). Cells 5 cm–2 m
   above the floor; with a capped heightmap (fusion stops at `obstacle_max_height`) cells
   at `structure_min_h` and above are walls/unknown and ignored (the sink sets it to the
   cap − 5 cm).
2. Blobs = 8-connected cells, cut where neighbours step > `split_dh`; a blob holding
   two large solid flat patches more than `split_patch_dh` apart is split between them.
3. Thin pieces (short side <= `thin_max`) higher than a neighbour within
   `attach_reach` cells attach to it (backrests, arms, headboards).
4. Main surface = the `surface_band` (10 cm) height window holding most base cells
   (cushioned seats, 5 cm heightmap); `flat_min_ratio` of the cells must be in it.
   Footprint extents come from the cell spread (sqrt(12 var + cell²)).
5. Classification (R15): Bed (>= 1.5 m², not couch-like) -> Chair/Couch (seat
   0.35–0.60, backrest >= 0.25 m above covering >= 60 % of the seat width; front =
   backrest -> seat; Couch when the seat side >= 1.2 m) -> Table (0.65–0.80) ->
   backless seat height = perch (Table if >= 0.25 m² else Other, not sittable) -> Other.
6. Scene labels: footprint IoU >= 0.3 or centres < 0.3 m = same object; label decides
   kind, geometry decides pose and seat height; kind conflict lowers confidence.
   Unmatched labels are dropped if their footprint is observed and empty (stale),
   otherwise published from the box (seats face away from the nearest wall).
7. Tracking: scene candidates bind to `scene:<uuid>`; the rest go through Hungarian
   assignment (gates: same kind, < 3 m, surface height < 5 cm) on
   distance + size + surface + height-histogram cost. Pose republishes only after 3
   consistent observations > 5 cm or > 10° away (`Moving` after 2); smaller changes go
   into an EMA. New objects need 3 consecutive observations (scene objects: 1);
   leftovers overlapping a live or Rejected object are absorbed. An unmatched object
   becomes `Missing` after 2 frames whose surface probe is visible and free, `Removed`
   5 s later (tombstone kept, revived if the same kind reappears within 1 m).
8. Seats: `OccupiedByUser` when the head is 1.1–1.3 m above the floor within 0.35 m of
   the seat (freezes the object pose); `Blocked` when > 40 % of the seat area is > 8 cm
   above the seat; object Moving/Missing/Removed propagate to its seats.

All thresholds live in `SemanticsParams` (`params.py`) for on-device tuning.

## Real tap (2026-10-09-room2.rktap, 129 s, office chair moved ~1 m at t≈74 s, PM sits)

Replayed through RoomService + SemanticsSink + TsdfFusionSink (feat/roomd-fusion defaults).

| setting | chair ids | id kept across move | OccupiedByUser | false chairs |
|---|---|---|---|---|
| shipped defaults (+ cap cut 1.85) | 2 | no (Removed, new id) | yes | 0 |
| suggested (below) | 1 | yes | yes (t≈95–102 s) | 0 |

Suggested on-device values: `structure_min_h` 1.5, `back_min_cover` 0.45 (the top of an
office chair backrest is narrower than seat + armrests), `flat_min_ratio` 0.35 (cushion +
armrests), `gate_surface_dh` 0.10 (seat height read 0.375–0.46 m from different views),
`move_yaw` 25 (yaw of a real chair jitters ±20°). Many `Other` fragments (walls under the
cut, clutter) remain; tables were not verified against ground truth.

## Known limits

- Checked end to end on roomd.fusion's TsdfFusion with its SynthScene (72 rendered
  frames): chair + table detected, a 1 m chair move kept its id via Moving, removal went
  Missing -> Removed from carving evidence; no real Quest capture yet.
- Synthetic tests pass up to 6 mm point noise; at 8 mm two touching objects of
  similar height (bed + nightstand) can merge.
- Yaw-only boxes; no ICP refinement (R15 mentions ICP): pose comes from the
  min-area rectangle of the 2 cm raster (~1 cm / ~3° jitter).
- A chair whose seat is fully under a table is not detected until pulled out.
- Seat states have no hysteresis; a single noisy frame can flip Blocked.
- Beds are not sittable by default (`bed_sittable`); SeatGenerator would put the
  seat in the middle of the mattress.
- Revival of a Removed object is by kind + distance only; two identical chairs
  taken out and brought back may swap ids.
