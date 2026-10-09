//! XR_FB_scene probe: load the room layout and every labelled scene anchor (bounding boxes,
//! poses) plus the XR_META_spatial_entity_mesh triangle count of the global mesh.

use crate::xrctx::{XrCtx, xr_result_json};
use openxr::{self as xr, sys};
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    ffi::CStr,
    ptr,
    time::{Duration, Instant},
};

const QUERY_TIMEOUT: Duration = Duration::from_secs(10);
const STATUS_TIMEOUT: Duration = Duration::from_secs(3);
const RECOGNIZED_LABELS: &CStr = c"TABLE,COUCH,FLOOR,CEILING,WALL_FACE,WINDOW_FRAME,DOOR_FRAME,\
STORAGE,BED,SCREEN,LAMP,PLANT,WALL_ART,GLOBAL_MESH,INVISIBLE_WALL_FACE,OTHER";

struct Fns {
    query: xr::raw::SpatialEntityQueryFB,
    entity: xr::raw::SpatialEntityFB,
    scene: xr::raw::SceneFB,
    mesh: Option<xr::raw::SpatialEntityMeshMETA>,
}

pub fn run(ctx: &mut XrCtx) -> Value {
    let exts = ctx.instance.exts();
    let (Some(query), Some(entity), Some(scene)) = (
        exts.fb_spatial_entity_query,
        exts.fb_spatial_entity,
        exts.fb_scene,
    ) else {
        return json!({"ok": false, "error": "needs XR_FB_scene + XR_FB_spatial_entity + XR_FB_spatial_entity_query"});
    };
    let fns = Fns { query, entity, scene, mesh: exts.meta_spatial_entity_mesh };
    let mut out = Map::new();

    // 1. Room layout
    match query_spaces(ctx, &fns, sys::SpaceComponentTypeFB::ROOM_LAYOUT, 16) {
        Ok((rooms, complete)) => {
            out.insert("room_query_complete".into(), complete);
            let list: Vec<Value> = rooms.iter().map(|r| room_layout(ctx, &fns, r)).collect();
            out.insert("rooms".into(), json!(list));
        }
        Err(e) => {
            out.insert("rooms_error".into(), e);
        }
    }

    // 2. Every anchor with semantic labels (walls, floor, furniture, global mesh, ...)
    let anchors = match query_spaces(ctx, &fns, sys::SpaceComponentTypeFB::SEMANTIC_LABELS, 1024) {
        Ok((anchors, complete)) => {
            out.insert("anchor_query_complete".into(), complete);
            anchors
        }
        Err(e) => {
            out.insert("anchors_error".into(), e);
            Vec::new()
        }
    };

    let mut entries: Vec<Map<String, Value>> = anchors.iter().map(|a| describe(ctx, &fns, a)).collect();

    // Locate needs the LOCATABLE component; enable it where it is off and wait for the events.
    let pending = enable_locatable(&fns, &anchors, &mut entries);
    if pending > 0 {
        wait_status_events(ctx, pending);
    }
    for (a, e) in anchors.iter().zip(entries.iter_mut()) {
        e.insert("pose_in_base".into(), locate(ctx, a.space));
    }

    let mut label_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut mesh_triangles = Vec::new();
    for e in &entries {
        let label = e.get("labels").and_then(Value::as_str).unwrap_or("?").to_string();
        if let Some(t) = e.get("mesh").and_then(|m| m.get("triangles")).and_then(Value::as_u64) {
            mesh_triangles.push(json!({"label": label, "triangles": t}));
        }
        *label_counts.entry(label).or_default() += 1;
    }
    let global_mesh_triangles: u64 = entries
        .iter()
        .filter(|e| e.get("labels").and_then(Value::as_str).is_some_and(|l| l.contains("GLOBAL_MESH")))
        .filter_map(|e| e.get("mesh")?.get("triangles")?.as_u64())
        .sum();
    out.insert("anchor_count".into(), json!(entries.len()));
    out.insert("label_counts".into(), json!(label_counts));
    out.insert("meshes".into(), json!(mesh_triangles));
    out.insert("global_mesh_triangles".into(), json!(global_mesh_triangles));
    out.insert("anchors".into(), json!(entries));
    let has_room = out.get("rooms").and_then(Value::as_array).is_some_and(|r| !r.is_empty());
    out.insert("ok".into(), json!(has_room || !entries.is_empty()));
    Value::Object(out)
}

/// xrQuerySpacesFB(LOAD, component filter) and collect results through the event pump.
fn query_spaces(
    ctx: &mut XrCtx,
    fns: &Fns,
    component: sys::SpaceComponentTypeFB,
    max: u32,
) -> Result<(Vec<sys::SpaceQueryResultFB>, Value), Value> {
    // Same filter chain as Meta's XrSceneModel sample: component + local storage.
    let storage = sys::SpaceStorageLocationFilterInfoFB {
        ty: sys::SpaceStorageLocationFilterInfoFB::TYPE,
        next: ptr::null(),
        location: sys::SpaceStorageLocationFB::LOCAL,
    };
    let filter = sys::SpaceComponentFilterInfoFB {
        ty: sys::SpaceComponentFilterInfoFB::TYPE,
        next: if ctx.instance.exts().fb_spatial_entity_storage.is_some() {
            &storage as *const _ as *const _
        } else {
            ptr::null()
        },
        component_type: component,
    };
    let info = sys::SpaceQueryInfoFB {
        ty: sys::SpaceQueryInfoFB::TYPE,
        next: ptr::null(),
        query_action: sys::SpaceQueryActionFB::LOAD,
        max_result_count: max,
        timeout: sys::Duration::NONE,
        filter: &filter as *const _ as *const sys::SpaceFilterInfoBaseHeaderFB,
        exclude_filter: ptr::null(),
    };
    let mut request = sys::AsyncRequestIdFB::from_raw(0);
    let r = unsafe {
        (fns.query.query_spaces)(
            ctx.session.as_raw(),
            &info as *const _ as *const sys::SpaceQueryInfoBaseHeaderFB,
            &mut request,
        )
    };
    if r.into_raw() < 0 {
        return Err(json!({"step": "xrQuerySpacesFB", "component": component.into_raw(), "error": xr_result_json(r)}));
    }

    let mut results = Vec::new();
    let mut complete = Value::Null;
    let deadline = Instant::now() + QUERY_TIMEOUT;
    while complete.is_null() && !ctx.exit && Instant::now() < deadline {
        for ev in ctx.tick(|_| {}) {
            match ev.ty {
                sys::StructureType::EVENT_DATA_SPACE_QUERY_RESULTS_AVAILABLE_FB => {
                    let e = unsafe { &*(&ev as *const _ as *const sys::EventDataSpaceQueryResultsAvailableFB) };
                    if e.request_id == request {
                        retrieve(ctx, fns, request, &mut results)?;
                    }
                }
                sys::StructureType::EVENT_DATA_SPACE_QUERY_COMPLETE_FB => {
                    let e = unsafe { &*(&ev as *const _ as *const sys::EventDataSpaceQueryCompleteFB) };
                    if e.request_id == request {
                        complete = xr_result_json(e.result);
                    }
                }
                _ => {}
            }
        }
    }
    if complete.is_null() {
        complete = json!({"result": "TIMEOUT", "after_s": QUERY_TIMEOUT.as_secs()});
    }
    Ok((results, complete))
}

fn retrieve(
    ctx: &XrCtx,
    fns: &Fns,
    request: sys::AsyncRequestIdFB,
    out: &mut Vec<sys::SpaceQueryResultFB>,
) -> Result<(), Value> {
    let session = ctx.session.as_raw();
    let mut results = sys::SpaceQueryResultsFB {
        ty: sys::SpaceQueryResultsFB::TYPE,
        next: ptr::null_mut(),
        result_capacity_input: 0,
        result_count_output: 0,
        results: ptr::null_mut(),
    };
    let r = unsafe { (fns.query.retrieve_space_query_results)(session, request, &mut results) };
    if r.into_raw() < 0 {
        return Err(json!({"step": "xrRetrieveSpaceQueryResultsFB(count)", "error": xr_result_json(r)}));
    }
    let mut buf: Vec<sys::SpaceQueryResultFB> =
        vec![unsafe { std::mem::zeroed() }; results.result_count_output as usize];
    results.result_capacity_input = buf.len() as u32;
    results.results = buf.as_mut_ptr();
    let r = unsafe { (fns.query.retrieve_space_query_results)(session, request, &mut results) };
    if r.into_raw() < 0 {
        return Err(json!({"step": "xrRetrieveSpaceQueryResultsFB", "error": xr_result_json(r)}));
    }
    buf.truncate(results.result_count_output as usize);
    out.extend(buf);
    Ok(())
}

fn uuid_hex(u: &sys::UuidEXT) -> String {
    u.data.iter().map(|b| format!("{b:02x}")).collect()
}

fn room_layout(ctx: &XrCtx, fns: &Fns, room: &sys::SpaceQueryResultFB) -> Value {
    let session = ctx.session.as_raw();
    let mut layout: sys::RoomLayoutFB = unsafe { std::mem::zeroed() };
    layout.ty = sys::RoomLayoutFB::TYPE;
    let r = unsafe { (fns.scene.get_space_room_layout)(session, room.space, &mut layout) };
    if r.into_raw() < 0 {
        return json!({"uuid": uuid_hex(&room.uuid), "error": xr_result_json(r)});
    }
    let mut walls: Vec<sys::UuidEXT> =
        vec![unsafe { std::mem::zeroed() }; layout.wall_uuid_count_output as usize];
    layout.wall_uuid_capacity_input = walls.len() as u32;
    layout.wall_uuids = walls.as_mut_ptr();
    let r = unsafe { (fns.scene.get_space_room_layout)(session, room.space, &mut layout) };
    if r.into_raw() < 0 {
        return json!({"uuid": uuid_hex(&room.uuid), "error": xr_result_json(r)});
    }
    walls.truncate(layout.wall_uuid_count_output as usize);
    json!({
        "uuid": uuid_hex(&room.uuid),
        "floor_uuid": uuid_hex(&layout.floor_uuid),
        "ceiling_uuid": uuid_hex(&layout.ceiling_uuid),
        "wall_uuids": walls.iter().map(uuid_hex).collect::<Vec<_>>(),
    })
}

fn component_enabled(fns: &Fns, space: sys::Space, c: sys::SpaceComponentTypeFB) -> Result<bool, sys::Result> {
    let mut status: sys::SpaceComponentStatusFB = unsafe { std::mem::zeroed() };
    status.ty = sys::SpaceComponentStatusFB::TYPE;
    let r = unsafe { (fns.entity.get_space_component_status)(space, c, &mut status) };
    if r.into_raw() < 0 { Err(r) } else { Ok(status.enabled.into()) }
}

fn describe(ctx: &XrCtx, fns: &Fns, a: &sys::SpaceQueryResultFB) -> Map<String, Value> {
    let session = ctx.session.as_raw();
    let mut e = Map::new();
    e.insert("uuid".into(), json!(uuid_hex(&a.uuid)));

    // Supported components
    let mut count = 0u32;
    let mut comps = Vec::new();
    unsafe {
        if (fns.entity.enumerate_space_supported_components)(a.space, 0, &mut count, ptr::null_mut()).into_raw() >= 0 {
            comps = vec![sys::SpaceComponentTypeFB::from_raw(0); count as usize];
            (fns.entity.enumerate_space_supported_components)(a.space, count, &mut count, comps.as_mut_ptr());
            comps.truncate(count as usize);
        }
    }
    let comp_status: Map<String, Value> = comps
        .iter()
        .map(|&c| {
            let v = match component_enabled(fns, a.space, c) {
                Ok(b) => json!(b),
                Err(r) => xr_result_json(r),
            };
            (component_name(c), v)
        })
        .collect();
    e.insert("components".into(), Value::Object(comp_status));

    // Semantic labels (the support info makes newer labels such as GLOBAL_MESH visible)
    let support = sys::SemanticLabelsSupportInfoFB {
        ty: sys::SemanticLabelsSupportInfoFB::TYPE,
        next: ptr::null(),
        flags: sys::SemanticLabelsSupportFlagsFB::ACCEPT_DESK_TO_TABLE_MIGRATION
            | sys::SemanticLabelsSupportFlagsFB::ACCEPT_INVISIBLE_WALL_FACE,
        recognized_labels: RECOGNIZED_LABELS.as_ptr(),
    };
    // Older XR_FB_scene versions may reject the support info; retry without it.
    match semantic_labels(fns, session, a.space, &support as *const _ as *const _)
        .or_else(|_| semantic_labels(fns, session, a.space, ptr::null()))
    {
        Ok(l) => e.insert("labels".into(), json!(l)),
        Err(r) => e.insert("labels_error".into(), xr_result_json(r)),
    };

    if comps.contains(&sys::SpaceComponentTypeFB::BOUNDED_3D) {
        let mut b: sys::Rect3DfFB = unsafe { std::mem::zeroed() };
        let r = unsafe { (fns.scene.get_space_bounding_box3_d)(session, a.space, &mut b) };
        e.insert("bbox3d".into(), if r.into_raw() >= 0 {
            json!({"offset": [b.offset.x, b.offset.y, b.offset.z],
                   "extent": [b.extent.width, b.extent.height, b.extent.depth]})
        } else {
            xr_result_json(r)
        });
    }
    if comps.contains(&sys::SpaceComponentTypeFB::BOUNDED_2D) {
        let mut b: sys::Rect2Df = unsafe { std::mem::zeroed() };
        let r = unsafe { (fns.scene.get_space_bounding_box2_d)(session, a.space, &mut b) };
        e.insert("bbox2d".into(), if r.into_raw() >= 0 {
            json!({"offset": [b.offset.x, b.offset.y], "extent": [b.extent.width, b.extent.height]})
        } else {
            xr_result_json(r)
        });
    }
    if comps.contains(&sys::SpaceComponentTypeFB::TRIANGLE_MESH_M) {
        e.insert("mesh".into(), triangle_mesh(fns, a.space));
    }
    e
}

fn semantic_labels(
    fns: &Fns,
    session: sys::Session,
    space: sys::Space,
    next: *const std::ffi::c_void,
) -> Result<String, sys::Result> {
    let mut labels = sys::SemanticLabelsFB {
        ty: sys::SemanticLabelsFB::TYPE,
        next,
        buffer_capacity_input: 0,
        buffer_count_output: 0,
        buffer: ptr::null_mut(),
    };
    let r = unsafe { (fns.scene.get_space_semantic_labels)(session, space, &mut labels) };
    if r.into_raw() < 0 {
        return Err(r);
    }
    let mut buf = vec![0 as std::ffi::c_char; labels.buffer_count_output as usize];
    labels.buffer_capacity_input = buf.len() as u32;
    labels.buffer = buf.as_mut_ptr();
    let r = unsafe { (fns.scene.get_space_semantic_labels)(session, space, &mut labels) };
    if r.into_raw() < 0 {
        return Err(r);
    }
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn triangle_mesh(fns: &Fns, space: sys::Space) -> Value {
    let Some(mesh_fns) = fns.mesh else {
        return json!({"error": "XR_META_spatial_entity_mesh not enabled"});
    };
    let info = sys::SpaceTriangleMeshGetInfoMETA {
        ty: sys::SpaceTriangleMeshGetInfoMETA::TYPE,
        next: ptr::null(),
    };
    let mut m: sys::SpaceTriangleMeshMETA = unsafe { std::mem::zeroed() };
    m.ty = sys::SpaceTriangleMeshMETA::TYPE;
    let r = unsafe { (mesh_fns.get_space_triangle_mesh)(space, &info, &mut m) };
    if r.into_raw() < 0 {
        return json!({"error": xr_result_json(r)});
    }
    let mut verts = vec![sys::Vector3f { x: 0.0, y: 0.0, z: 0.0 }; m.vertex_count_output as usize];
    let mut idx = vec![0u32; m.index_count_output as usize];
    m.vertex_capacity_input = verts.len() as u32;
    m.vertices = verts.as_mut_ptr();
    m.index_capacity_input = idx.len() as u32;
    m.indices = idx.as_mut_ptr();
    let r = unsafe { (mesh_fns.get_space_triangle_mesh)(space, &info, &mut m) };
    if r.into_raw() < 0 {
        return json!({"error": xr_result_json(r)});
    }
    let (lo, hi) = verts.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), v| {
        ([lo[0].min(v.x), lo[1].min(v.y), lo[2].min(v.z)], [hi[0].max(v.x), hi[1].max(v.y), hi[2].max(v.z)])
    });
    json!({
        "vertices": m.vertex_count_output,
        "triangles": m.index_count_output / 3,
        "vertex_aabb_local": if verts.is_empty() { Value::Null } else { json!([lo, hi]) },
    })
}

fn enable_locatable(
    fns: &Fns,
    anchors: &[sys::SpaceQueryResultFB],
    entries: &mut [Map<String, Value>],
) -> usize {
    let mut pending = 0;
    for (a, e) in anchors.iter().zip(entries.iter_mut()) {
        if component_enabled(fns, a.space, sys::SpaceComponentTypeFB::LOCATABLE) != Ok(false) {
            continue;
        }
        let info = sys::SpaceComponentStatusSetInfoFB {
            ty: sys::SpaceComponentStatusSetInfoFB::TYPE,
            next: ptr::null(),
            component_type: sys::SpaceComponentTypeFB::LOCATABLE,
            enabled: true.into(),
            timeout: sys::Duration::NONE,
        };
        let mut req = sys::AsyncRequestIdFB::from_raw(0);
        let r = unsafe { (fns.entity.set_space_component_status)(a.space, &info, &mut req) };
        e.insert("enable_locatable".into(), xr_result_json(r));
        if r.into_raw() >= 0 {
            pending += 1;
        }
    }
    pending
}

fn wait_status_events(ctx: &mut XrCtx, mut pending: usize) {
    let deadline = Instant::now() + STATUS_TIMEOUT;
    while pending > 0 && !ctx.exit && Instant::now() < deadline {
        for ev in ctx.tick(|_| {}) {
            if ev.ty == sys::StructureType::EVENT_DATA_SPACE_SET_STATUS_COMPLETE_FB {
                pending = pending.saturating_sub(1);
            }
        }
    }
}

fn locate(ctx: &XrCtx, space: sys::Space) -> Value {
    let Some(base) = ctx.space.as_ref() else {
        return json!({"error": "no base space"});
    };
    let mut loc: sys::SpaceLocation = unsafe { std::mem::zeroed() };
    loc.ty = sys::SpaceLocation::TYPE;
    let r = unsafe { (ctx.instance.fp().locate_space)(space, base.as_raw(), ctx.last_time, &mut loc) };
    if r.into_raw() < 0 {
        return json!({"error": xr_result_json(r)});
    }
    let (o, p) = (loc.pose.orientation, loc.pose.position);
    json!({
        "flags": loc.location_flags.into_raw(),
        "q_xyzw": [o.x, o.y, o.z, o.w],
        "p_xyz": [p.x, p.y, p.z],
    })
}

fn component_name(c: sys::SpaceComponentTypeFB) -> String {
    match c.into_raw() {
        0 => "LOCATABLE",
        1 => "STORABLE",
        2 => "SHARABLE",
        3 => "BOUNDED_2D",
        4 => "BOUNDED_3D",
        5 => "SEMANTIC_LABELS",
        6 => "ROOM_LAYOUT",
        7 => "SPACE_CONTAINER",
        1000269000 => "TRIANGLE_MESH_META",
        _ => return format!("component_{}", c.into_raw()),
    }
    .to_string()
}
