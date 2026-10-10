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
   above the floor; cells at `structure_min_h` (1.5 m) and above are structure (walls, loft
   bed, beams, tall cabinets) and ignored. The fusion heightmap only scans up to
   `FusionConfig.obstacle_max_height` (1.7 m: character height 1.5–1.6 m + margin, so a
   loft underside at ~1.55 m blocks walking and a ~1.9 m beam does not), so the sink uses
   min(1.5, cap − 0.05): the cut always sits below the cap, where walls and tall cabinets
   are indistinguishable, and above every seat and table top (≤ 1.25 m backrests).
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
   becomes `Missing` after 2 frames of "gone" evidence, `Removed` 5 s later. Gone = the
   surface slab is visible and free **and** the heightmap shows floor (top below half the
   surface height) over ≥ 60 % of the known probe cells; never while a seat is
   `OccupiedByUser` or the user's head is within 0.45 m of the footprint (the fusion body
   mask hides what the user stands at or sits on). A `Removed` tombstone takes part in the
   assignment again (gate `revive_dist` = 1 m) and is revived by one observation.
8. Seats: `OccupiedByUser` when the head is 1.1–1.3 m above the floor within 0.35 m of
   the seat (freezes the object pose); `Blocked` when > 40 % of the seat area is > 8 cm
   above the seat for 2 consecutive frames **and** that share is ≥ 0.25 above the
   baseline taken at the first clear look at the published pose (the object's own
   armrests / backrest under a shifted area; baseline capped at 0.6); object
   Moving/Missing/Removed propagate to its seats.

All thresholds live in `SemanticsParams` (`params.py`) for on-device tuning.

## Real tap (2026-10-09-room2.rktap, 129 s)

Room (PM): ~10 m², loft bed over the desks (underside ~1.55 m), ceiling beam, 2 desks with a
basket drawer unit between them, wardrobe, glass cabinet, suitcase, floor fan, one office
chair. PM sat on the chair under the desk at the start, moved it ~1.3 m at t≈74 s and sat
on it at t≈95–102 s. Replayed through RoomService + SemanticsSink + TsdfFusionSink.

| | chair ids | id kept across move | OccupiedByUser | false chairs | Table ids |
|---|---|---|---|---|---|
| old 5 cm seat gate, 1.85 m cut | 2 | yes, plus a duplicate id at the new spot | yes | 1 (short-lived) | 3 |
| defaults (1.5 m cut, 0.10 m gate, fusion cap 1.7 m) | 1 | yes (Missing t=86, Present t=92) | yes (t=95–102) | 0 | 2 |

With the defaults: the desk zone under the loft is one Table at (2.26, −1.01), 2.07 × 0.57 m
(both desks and the basket unit merge); a second Table id first appears at (−0.13, 0.25) and
ends at (1.67, −1.10) (0.94 × 0.51 m, overlapping the desk zone): the 3 m gate lets table ids
wander across the room.

Real-data rules (each from a failure on this tap): heightmap segmentation (points too
sparse), overhang refill + free-space probe (desk under the loft), refilled cells are never a
backrest and seats seen mostly under an overhang are not seats, backrest ≤ 0.8 × seat,
seat side ≥ 0.35 m and depth ≤ 1.1 m, chair/couch top ≤ 1.32 m, table ≥ 0.30 m² with side
≥ 0.40 m, a user sitting ends Moving.

### Seat robustness (live 2026-10-10: "the room has no available seat")

Live roomd model at the time: the real office chair `Removed` while in the room, two
narrow/tall pieces (`0.31 m` wide; `1.39 m` tall) published as chairs with `Blocked` seats.
Replaying room2 with candidate dumps showed the causes:

- The free-slab probe alone sent a present chair `Missing` (base: t=58 s, one second after
  creation, while the PM stood at it): the published seat height is noisy (0.39–0.47 m on
  this chair) and the body mask leaves the slab carved. Now the heightmap must also show
  floor, and the user's spot never counts.
- Tall false chairs passed the old rules and, while the real chair was Moving/Missing, the
  3 m gate let it take their observations (base t=77–85 s: `Moving` with no real motion).
  The real chair reads 1.23–1.27 m tall here (p90 of its backrest cells 1.22–1.26); the
  false pieces 1.34–1.48 m, or 0.30–0.37 m wide → top ≤ 1.32 m, side ≥ 0.35 m.
- After the user stood up (t≈105 s) the re-measured box put the armrests (0.61–0.79 m)
  and backrest into the seat area: 55 % high cells, `Blocked` until the end. The per-pose
  baseline keeps it `Available`.

| room2, defaults | chair ids | false chairs | Missing while present | id kept across move | OccupiedByUser | seat after t=57 s (polls Available / Moving / Missing / Occupied / Blocked) |
|---|---|---|---|---|---|---|
| before | 1 | 0 tracks (Chair candidates at 14, 36, 57–84, 103 s) | t=58 s | yes (Moving 77–86, Missing 86–92) | t=95–102 | 20 / 13 / 6 / 7 / 23 |
| after  | 1 | 0 tracks (no tall/narrow Chair candidates) | none | yes (Missing 79–81, Present at the new spot from 81) | t=95–102 | 52 / 8 / 2 / 7 / 0 | Geometry-only `Other` objects are not published by the sink.

## Known limits

- Checked end to end on roomd.fusion's TsdfFusion with its SynthScene (72 rendered
  frames): chair + table detected, a 1 m chair move kept its id via Moving, removal went
  Missing -> Removed from carving evidence; no real Quest capture yet.
- Synthetic tests pass up to 6 mm point noise; at 8 mm two touching objects of
  similar height (bed + nightstand) can merge.
- Yaw-only boxes; no ICP refinement (R15 mentions ICP): pose comes from the
  min-area rectangle of the 2 cm raster (~1 cm / ~3° jitter).
- A chair whose seat is fully under a table is not detected until pulled out; while the
  user sits on it the fusion body mask hides it too, so the chair the PM sat on at the
  start of the tap is first seen at t≈60 s (OccupiedByUser cannot fire before that).
- Seats seen only under an overhang (loft desk zone) are never offered.
- A re-measured pose can shift ~9 cm (room2 t≈105 s, after the user stood up) so the seat
  area covers the armrests/backrest; the per-pose baseline keeps such a seat Available,
  but a pile already on the seat when a pose is published raises the baseline (cap 0.6):
  it then needs > 0.85 high cells. Pose anchoring (backrest edge) is still not done.
- `seat_max_top` (1.32 m) is tuned to one chair: a high-back gaming chair taller than that
  would not be offered.
- Table ids can jump to another table-like candidate within `gate_dist` (3 m); a per-kind
  gate (≈1 m for tables) is the next step.
- Only Blocked is debounced (2 frames); Available / OccupiedByUser apply at once.
- Beds are not sittable by default (`bed_sittable`); SeatGenerator would put the
  seat in the middle of the mattress.
- Revival of a Removed object is by kind + distance (1 m) + seat height only; two identical
  chairs taken out and brought back may swap ids. A chair carried > 1 m while Removed gets
  a new id after 3 observations.
