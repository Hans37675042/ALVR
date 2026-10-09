// Quest Space Setup scene import (XR_FB_scene + XR_FB_spatial_entity_* + XR_META_spatial_entity_mesh).
//
// Flow, following Meta's XrSceneModel sample:
// 1. xrQuerySpacesFB filtered by ROOM_LAYOUT -> room anchors (floor/ceiling/walls + container)
// 2. xrQuerySpacesFB by the container UUIDs -> every anchor of the room (incl. GLOBAL_MESH)
// 3. enable LOCATABLE where needed and wait for XrEventDataSpaceSetStatusCompleteFB
// 4. read labels / bounds / triangle mesh and locate each anchor in STAGE
//
// All calls are asynchronous through OpenXR events; lib.rs forwards them to the loader from its
// xrPollEvent loop and calls `update()` once per frame.

use alvr_common::{error, info, warn};
use alvr_packets::{SceneAnchor, SceneMesh, SceneRoom, SceneSnapshot, SceneUuid};
use openxr::{self as xr, raw, sys};
use std::{
    ffi::{CString, c_char},
    ptr,
    time::{Duration, Instant},
};

#[cfg(target_os = "android")]
pub const USE_SCENE_PERMISSION: &str = "com.oculus.permission.USE_SCENE";
const QUERY_TIMEOUT: Duration = Duration::from_secs(10);
// Anchors are usually locatable right after loading; give the runtime a few frames otherwise.
const MAX_LOCATE_RETRY_FRAMES: u32 = 45;
// GLOBAL_MESH is not in XR_FB_scene's original label set; listing it keeps it from being
// reported as OTHER.
const RECOGNIZED_LABELS: &str = "TABLE,COUCH,FLOOR,CEILING,WALL_FACE,WINDOW_FRAME,DOOR_FRAME,\
    STORAGE,BED,SCREEN,LAMP,PLANT,WALL_ART,INVISIBLE_WALL_FACE,GLOBAL_MESH,OTHER";

fn xr_ok(result: sys::Result) -> xr::Result<()> {
    if result.into_raw() >= 0 {
        Ok(())
    } else {
        Err(result)
    }
}

fn uuid_is_valid(uuid: &sys::UuidEXT) -> bool {
    uuid.data.iter().any(|&b| b != 0)
}

#[derive(Clone, Copy)]
struct SceneFns {
    entity: raw::SpatialEntityFB,
    query: raw::SpatialEntityQueryFB,
    scene: raw::SceneFB,
    container: raw::SpatialEntityContainerFB,
    mesh: Option<raw::SpatialEntityMeshMETA>,
}

struct Anchor {
    space: sys::Space,
    uuid: SceneUuid,
    // Pending xrSetSpaceComponentStatusFB(LOCATABLE) request
    locatable_request: Option<sys::AsyncRequestIdFB>,
}

enum State {
    Idle,
    QueryingRooms {
        request: sys::AsyncRequestIdFB,
        deadline: Instant,
        rooms: Vec<SceneRoom>,
        anchor_uuids: Vec<SceneUuid>,
    },
    QueryingAnchors {
        request: sys::AsyncRequestIdFB,
        deadline: Instant,
        rooms: Vec<SceneRoom>,
        anchors: Vec<Anchor>,
        query_done: bool,
        locate_retries: u32,
    },
    // None: the query failed or timed out, nothing is sent and the server keeps its cache
    Done(Option<SceneSnapshot>),
}

pub struct SceneLoader {
    session: xr::Session<xr::OpenGlEs>,
    fns: SceneFns,
    stage: xr::Space,
    state: State,
    // A request arrived while a query was running: query again once it finishes
    requery: bool,
    // Spaces loaded by the queries; destroyed once the snapshot is built
    loaded_spaces: Vec<sys::Space>,
}

impl SceneLoader {
    /// None when the runtime lacks the scene extensions (they must be enabled on the instance).
    pub fn new(session: xr::Session<xr::OpenGlEs>) -> Option<Self> {
        let exts = session.instance().exts();
        let fns = SceneFns {
            entity: exts.fb_spatial_entity?,
            query: exts.fb_spatial_entity_query?,
            scene: exts.fb_scene?,
            container: exts.fb_spatial_entity_container?,
            mesh: exts.meta_spatial_entity_mesh,
        };
        if fns.mesh.is_none() {
            warn!("[SCENE] XR_META_spatial_entity_mesh unavailable: no global mesh");
        }

        let stage = session
            .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
            .ok()?;

        Some(Self {
            session,
            fns,
            stage,
            state: State::Idle,
            requery: false,
            loaded_spaces: Vec::new(),
        })
    }

    pub fn update_reference_space(&mut self) {
        if let Ok(stage) = self
            .session
            .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
        {
            self.stage = stage;
        }
    }

    /// Start a scene query; if one is running, run another one after it.
    pub fn request(&mut self) {
        if !matches!(self.state, State::Idle) {
            self.requery = true;
            return;
        }

        // Without the permission the runtime reports no rooms, which must not be mistaken for an
        // empty room
        #[cfg(target_os = "android")]
        if !alvr_system_info::has_permission(USE_SCENE_PERMISSION) {
            warn!("[SCENE] USE_SCENE permission not granted, scene query skipped");
            alvr_system_info::try_get_permission(USE_SCENE_PERMISSION);
            return;
        }

        let component_filter = sys::SpaceComponentFilterInfoFB {
            ty: sys::SpaceComponentFilterInfoFB::TYPE,
            next: ptr::null(),
            component_type: sys::SpaceComponentTypeFB::ROOM_LAYOUT,
        };
        match self.query_spaces(&component_filter as *const _ as _, 64) {
            Ok(request) => {
                info!("[SCENE] querying room layout");
                self.state = State::QueryingRooms {
                    request,
                    deadline: Instant::now() + QUERY_TIMEOUT,
                    rooms: Vec::new(),
                    anchor_uuids: Vec::new(),
                };
            }
            Err(e) => error!("[SCENE] xrQuerySpacesFB(ROOM_LAYOUT) failed: {e:?}"),
        }
    }

    fn query_spaces(
        &self,
        filter: *const sys::SpaceFilterInfoBaseHeaderFB,
        max_results: u32,
    ) -> xr::Result<sys::AsyncRequestIdFB> {
        let info = sys::SpaceQueryInfoFB {
            ty: sys::SpaceQueryInfoFB::TYPE,
            next: ptr::null(),
            query_action: sys::SpaceQueryActionFB::LOAD,
            max_result_count: max_results,
            timeout: sys::Duration::NONE,
            filter,
            exclude_filter: ptr::null(),
        };
        let mut request = sys::AsyncRequestIdFB::default();
        let res = unsafe {
            (self.fns.query.query_spaces)(
                self.session.as_raw(),
                &info as *const _ as _,
                &mut request,
            )
        };
        xr_ok(res).map(|_| request)
    }

    fn retrieve_results(&self, request: sys::AsyncRequestIdFB) -> xr::Result<Vec<sys::SpaceQueryResultFB>> {
        let mut results = sys::SpaceQueryResultsFB {
            ty: sys::SpaceQueryResultsFB::TYPE,
            next: ptr::null_mut(),
            result_capacity_input: 0,
            result_count_output: 0,
            results: ptr::null_mut(),
        };
        let retrieve = self.fns.query.retrieve_space_query_results;
        unsafe {
            xr_ok(retrieve(self.session.as_raw(), request, &mut results))?;
            let mut buffer = vec![
                sys::SpaceQueryResultFB {
                    space: <sys::Space as sys::Handle>::NULL,
                    uuid: sys::UuidEXT { data: [0; 16] },
                };
                results.result_count_output as usize
            ];
            results.result_capacity_input = buffer.len() as u32;
            results.results = buffer.as_mut_ptr();
            xr_ok(retrieve(self.session.as_raw(), request, &mut results))?;
            buffer.truncate(results.result_count_output as usize);

            Ok(buffer)
        }
    }

    fn component_enabled(&self, space: sys::Space, component: sys::SpaceComponentTypeFB) -> bool {
        let mut status = sys::SpaceComponentStatusFB {
            ty: sys::SpaceComponentStatusFB::TYPE,
            next: ptr::null_mut(),
            enabled: sys::FALSE,
            change_pending: sys::FALSE,
        };
        let res =
            unsafe { (self.fns.entity.get_space_component_status)(space, component, &mut status) };

        xr_ok(res).is_ok() && status.enabled.into() && !bool::from(status.change_pending)
    }

    fn component_supported(&self, space: sys::Space, component: sys::SpaceComponentTypeFB) -> bool {
        let enumerate = self.fns.entity.enumerate_space_supported_components;
        unsafe {
            let mut count = 0;
            if xr_ok(enumerate(space, 0, &mut count, ptr::null_mut())).is_err() {
                return false;
            }
            let mut types = vec![sys::SpaceComponentTypeFB::LOCATABLE; count as usize];
            if xr_ok(enumerate(space, count, &mut count, types.as_mut_ptr())).is_err() {
                return false;
            }
            types.truncate(count as usize);

            types.contains(&component)
        }
    }

    fn read_room(&self, space: sys::Space, uuid: SceneUuid) -> Option<(SceneRoom, Vec<SceneUuid>)> {
        let get_layout = self.fns.scene.get_space_room_layout;
        let mut layout = sys::RoomLayoutFB {
            ty: sys::RoomLayoutFB::TYPE,
            next: ptr::null(),
            floor_uuid: sys::UuidEXT { data: [0; 16] },
            ceiling_uuid: sys::UuidEXT { data: [0; 16] },
            wall_uuid_capacity_input: 0,
            wall_uuid_count_output: 0,
            wall_uuids: ptr::null_mut(),
        };
        let mut walls = Vec::new();
        unsafe {
            if let Err(e) = xr_ok(get_layout(self.session.as_raw(), space, &mut layout)) {
                warn!("[SCENE] xrGetSpaceRoomLayoutFB failed: {e:?}");
                return None;
            }
            if layout.wall_uuid_count_output > 0 {
                walls = vec![sys::UuidEXT { data: [0; 16] }; layout.wall_uuid_count_output as usize];
                layout.wall_uuid_capacity_input = walls.len() as u32;
                layout.wall_uuids = walls.as_mut_ptr();
                if xr_ok(get_layout(self.session.as_raw(), space, &mut layout)).is_err() {
                    walls.clear();
                }
                walls.truncate(layout.wall_uuid_count_output as usize);
            }
        }

        let get_container = self.fns.container.get_space_container;
        let mut container = sys::SpaceContainerFB {
            ty: sys::SpaceContainerFB::TYPE,
            next: ptr::null(),
            uuid_capacity_input: 0,
            uuid_count_output: 0,
            uuids: ptr::null_mut(),
        };
        let mut contained = Vec::new();
        unsafe {
            if xr_ok(get_container(self.session.as_raw(), space, &mut container)).is_ok()
                && container.uuid_count_output > 0
            {
                contained =
                    vec![sys::UuidEXT { data: [0; 16] }; container.uuid_count_output as usize];
                container.uuid_capacity_input = contained.len() as u32;
                container.uuids = contained.as_mut_ptr();
                if xr_ok(get_container(self.session.as_raw(), space, &mut container)).is_err() {
                    contained.clear();
                }
                contained.truncate(container.uuid_count_output as usize);
            }
        }

        let room = SceneRoom {
            uuid,
            floor: uuid_is_valid(&layout.floor_uuid).then_some(layout.floor_uuid.data),
            ceiling: uuid_is_valid(&layout.ceiling_uuid).then_some(layout.ceiling_uuid.data),
            walls: walls.iter().filter(|u| uuid_is_valid(u)).map(|u| u.data).collect(),
        };
        let mut anchor_uuids: Vec<SceneUuid> = contained.iter().map(|u| u.data).collect();
        // Older runtimes may not list the layout planes in the container
        for plane in room.floor.iter().chain(&room.ceiling).chain(&room.walls) {
            if !anchor_uuids.contains(plane) {
                anchor_uuids.push(*plane);
            }
        }

        Some((room, anchor_uuids))
    }

    pub fn on_query_results(&mut self, request: sys::AsyncRequestIdFB) {
        match &self.state {
            State::QueryingRooms { request: r, .. } if *r == request => {}
            State::QueryingAnchors { request: r, .. } if *r == request => {}
            _ => return,
        }
        let results = match self.retrieve_results(request) {
            Ok(results) => results,
            Err(e) => {
                error!("[SCENE] xrRetrieveSpaceQueryResultsFB failed: {e:?}, nothing sent");
                self.finish(None);
                return;
            }
        };
        self.loaded_spaces.extend(results.iter().map(|r| r.space));

        match &self.state {
            State::QueryingRooms { .. } => {
                let mut found = Vec::new();
                for result in &results {
                    if let Some(room) = self.read_room(result.space, result.uuid.data) {
                        found.push(room);
                    }
                }
                if let State::QueryingRooms {
                    rooms,
                    anchor_uuids,
                    ..
                } = &mut self.state
                {
                    for (room, uuids) in found {
                        for uuid in uuids {
                            if !anchor_uuids.contains(&uuid) {
                                anchor_uuids.push(uuid);
                            }
                        }
                        rooms.push(room);
                    }
                }
            }
            State::QueryingAnchors { .. } => {
                let mut new_anchors = Vec::new();
                for result in &results {
                    let mut anchor = Anchor {
                        space: result.space,
                        uuid: result.uuid.data,
                        locatable_request: None,
                    };
                    if !self.component_enabled(result.space, sys::SpaceComponentTypeFB::LOCATABLE)
                        && self.component_supported(
                            result.space,
                            sys::SpaceComponentTypeFB::LOCATABLE,
                        )
                    {
                        anchor.locatable_request = self.enable_locatable(result.space);
                    }
                    new_anchors.push(anchor);
                }
                if let State::QueryingAnchors { anchors, .. } = &mut self.state {
                    anchors.extend(new_anchors);
                }
            }
            State::Idle | State::Done(_) => (),
        }
    }

    fn enable_locatable(&self, space: sys::Space) -> Option<sys::AsyncRequestIdFB> {
        let info = sys::SpaceComponentStatusSetInfoFB {
            ty: sys::SpaceComponentStatusSetInfoFB::TYPE,
            next: ptr::null(),
            component_type: sys::SpaceComponentTypeFB::LOCATABLE,
            enabled: sys::TRUE,
            timeout: sys::Duration::NONE,
        };
        let mut request = sys::AsyncRequestIdFB::default();
        let res = unsafe { (self.fns.entity.set_space_component_status)(space, &info, &mut request) };
        match res {
            r if r.into_raw() >= 0 => Some(request),
            sys::Result::ERROR_SPACE_COMPONENT_STATUS_ALREADY_SET_FB => None,
            e => {
                warn!("[SCENE] enabling LOCATABLE failed: {e:?}");
                None
            }
        }
    }

    pub fn on_query_complete(&mut self, request: sys::AsyncRequestIdFB, result: sys::Result) {
        match &mut self.state {
            State::QueryingRooms {
                request: r,
                rooms,
                anchor_uuids,
                ..
            } if *r == request => {
                if result.into_raw() < 0 {
                    warn!("[SCENE] room query failed with {result:?}, nothing sent");
                    self.finish(None);
                    return;
                }
                let rooms = std::mem::take(rooms);
                let mut uuids = std::mem::take(anchor_uuids);
                if uuids.is_empty() {
                    info!("[SCENE] no Space Setup room found on the headset");
                    self.finish(Some(SceneSnapshot {
                        rooms,
                        ..Default::default()
                    }));
                    return;
                }

                let mut uuid_buffer: Vec<sys::UuidEXT> =
                    uuids.drain(..).map(|data| sys::UuidEXT { data }).collect();
                let location_filter = sys::SpaceStorageLocationFilterInfoFB {
                    ty: sys::SpaceStorageLocationFilterInfoFB::TYPE,
                    next: ptr::null(),
                    location: sys::SpaceStorageLocationFB::LOCAL,
                };
                let uuid_filter = sys::SpaceUuidFilterInfoFB {
                    ty: sys::SpaceUuidFilterInfoFB::TYPE,
                    next: &location_filter as *const _ as _,
                    uuid_count: uuid_buffer.len() as u32,
                    uuids: uuid_buffer.as_mut_ptr(),
                };
                info!(
                    "[SCENE] {} rooms, querying {} anchors",
                    rooms.len(),
                    uuid_buffer.len()
                );
                match self.query_spaces(&uuid_filter as *const _ as _, uuid_buffer.len() as u32) {
                    Ok(request) => {
                        self.state = State::QueryingAnchors {
                            request,
                            deadline: Instant::now() + QUERY_TIMEOUT,
                            rooms,
                            anchors: Vec::new(),
                            query_done: false,
                            locate_retries: 0,
                        };
                    }
                    Err(e) => {
                        error!("[SCENE] xrQuerySpacesFB(uuids) failed: {e:?}, nothing sent");
                        self.finish(None);
                    }
                }
            }
            State::QueryingAnchors {
                request: r,
                query_done,
                ..
            } if *r == request => {
                if result.into_raw() < 0 {
                    warn!("[SCENE] anchor query failed with {result:?}, nothing sent");
                    self.finish(None);
                    return;
                }
                *query_done = true;
            }
            _ => (),
        }
    }

    pub fn on_set_status_complete(&mut self, request: sys::AsyncRequestIdFB, result: sys::Result) {
        if let State::QueryingAnchors { anchors, .. } = &mut self.state
            && let Some(anchor) = anchors
                .iter_mut()
                .find(|a| a.locatable_request == Some(request))
        {
            if result.into_raw() < 0 {
                warn!("[SCENE] LOCATABLE for an anchor failed: {result:?}");
            }
            anchor.locatable_request = None;
        }
    }

    /// Space Setup finished (it is launched by the user, never by ALVR while streaming): the
    /// scene may have changed.
    pub fn on_capture_complete(&self, result: sys::Result) {
        info!("[SCENE] Space Setup finished: {result:?}");
    }

    /// Call once per frame. Returns a finished snapshot at most once per query: complete, or
    /// empty when the headset has no room. Failed or timed out queries return nothing.
    pub fn update(&mut self, time: xr::Time) -> Option<SceneSnapshot> {
        let now = Instant::now();
        let snapshot = match &self.state {
            State::Idle => return None,
            State::Done(_) => {
                let State::Done(snapshot) = std::mem::replace(&mut self.state, State::Idle) else {
                    unreachable!()
                };
                snapshot
            }
            State::QueryingRooms { deadline, .. } => {
                if now <= *deadline {
                    return None;
                }
                warn!("[SCENE] room query timed out, nothing sent");
                None
            }
            State::QueryingAnchors {
                deadline,
                query_done,
                anchors,
                locate_retries,
                ..
            } => {
                let timed_out = now > *deadline;
                let loaded = *query_done && anchors.iter().all(|a| a.locatable_request.is_none());
                if !loaded && !timed_out {
                    return None;
                }
                let all_located = anchors.iter().all(|a| {
                    !self.component_enabled(a.space, sys::SpaceComponentTypeFB::LOCATABLE)
                        || self.locate(a.space, time).is_some()
                });
                if timed_out {
                    warn!("[SCENE] anchor query timed out, nothing sent");
                    None
                } else if !all_located && *locate_retries < MAX_LOCATE_RETRY_FRAMES {
                    if let State::QueryingAnchors { locate_retries, .. } = &mut self.state {
                        *locate_retries += 1;
                    }
                    return None;
                } else {
                    Some(self.build_snapshot(time))
                }
            }
        };

        self.finish_state();

        snapshot
    }

    // Used from event handlers: the result is handed out by the next update()
    fn finish(&mut self, snapshot: Option<SceneSnapshot>) {
        self.state = State::Done(snapshot);
    }

    fn finish_state(&mut self) {
        for space in self.loaded_spaces.drain(..) {
            unsafe {
                (self.session.instance().fp().destroy_space)(space);
            }
        }
        self.state = State::Idle;
        if std::mem::take(&mut self.requery) {
            self.request();
        }
    }

    fn locate(&self, space: sys::Space, time: xr::Time) -> Option<alvr_common::Pose> {
        let mut location = sys::SpaceLocation {
            ty: sys::SpaceLocation::TYPE,
            next: ptr::null_mut(),
            location_flags: sys::SpaceLocationFlags::EMPTY,
            pose: xr::Posef::IDENTITY,
        };
        let res = unsafe {
            (self.session.instance().fp().locate_space)(space, self.stage.as_raw(), time, &mut location)
        };
        let valid =
            sys::SpaceLocationFlags::POSITION_VALID | sys::SpaceLocationFlags::ORIENTATION_VALID;

        (xr_ok(res).is_ok() && location.location_flags.contains(valid))
            .then(|| crate::from_xr_pose(location.pose))
    }

    fn semantic_labels(&self, space: sys::Space) -> String {
        let get_labels = self.fns.scene.get_space_semantic_labels;
        let recognized = CString::new(RECOGNIZED_LABELS).unwrap();
        let support = sys::SemanticLabelsSupportInfoFB {
            ty: sys::SemanticLabelsSupportInfoFB::TYPE,
            next: ptr::null(),
            flags: sys::SemanticLabelsSupportFlagsFB::MULTIPLE_SEMANTIC_LABELS
                | sys::SemanticLabelsSupportFlagsFB::ACCEPT_DESK_TO_TABLE_MIGRATION
                | sys::SemanticLabelsSupportFlagsFB::ACCEPT_INVISIBLE_WALL_FACE,
            recognized_labels: recognized.as_ptr(),
        };

        // Retry without the support info for runtimes that reject it
        for next in [&support as *const _ as *const _, ptr::null()] {
            let mut labels = sys::SemanticLabelsFB {
                ty: sys::SemanticLabelsFB::TYPE,
                next,
                buffer_capacity_input: 0,
                buffer_count_output: 0,
                buffer: ptr::null_mut(),
            };
            unsafe {
                if xr_ok(get_labels(self.session.as_raw(), space, &mut labels)).is_err() {
                    continue;
                }
                let mut buffer = vec![0 as c_char; labels.buffer_count_output as usize];
                labels.buffer_capacity_input = buffer.len() as u32;
                labels.buffer = buffer.as_mut_ptr();
                if xr_ok(get_labels(self.session.as_raw(), space, &mut labels)).is_err() {
                    continue;
                }
                let bytes = buffer
                    .iter()
                    .take(labels.buffer_count_output as usize)
                    .map(|&c| c as u8)
                    .take_while(|&c| c != 0)
                    .collect::<Vec<_>>();

                return String::from_utf8_lossy(&bytes).into_owned();
            }
        }

        String::new()
    }

    fn bounding_box_2d(&self, space: sys::Space) -> Option<[f32; 4]> {
        let mut rect = sys::Rect2Df::default();
        let res = unsafe {
            (self.fns.scene.get_space_bounding_box2_d)(self.session.as_raw(), space, &mut rect)
        };
        xr_ok(res).ok().map(|_| {
            [
                rect.offset.x,
                rect.offset.y,
                rect.extent.width,
                rect.extent.height,
            ]
        })
    }

    fn boundary_2d(&self, space: sys::Space) -> Option<Vec<[f32; 2]>> {
        let get_boundary = self.fns.scene.get_space_boundary2_d;
        let mut boundary = sys::Boundary2DFB {
            ty: sys::Boundary2DFB::TYPE,
            next: ptr::null(),
            vertex_capacity_input: 0,
            vertex_count_output: 0,
            vertices: ptr::null_mut(),
        };
        unsafe {
            xr_ok(get_boundary(self.session.as_raw(), space, &mut boundary)).ok()?;
            let mut vertices = vec![sys::Vector2f::default(); boundary.vertex_count_output as usize];
            boundary.vertex_capacity_input = vertices.len() as u32;
            boundary.vertices = vertices.as_mut_ptr();
            xr_ok(get_boundary(self.session.as_raw(), space, &mut boundary)).ok()?;
            vertices.truncate(boundary.vertex_count_output as usize);

            Some(vertices.iter().map(|v| [v.x, v.y]).collect())
        }
    }

    fn bounding_box_3d(&self, space: sys::Space) -> Option<[f32; 6]> {
        let mut rect = sys::Rect3DfFB {
            offset: sys::Offset3DfFB {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            extent: sys::Extent3DfFB {
                width: 0.0,
                height: 0.0,
                depth: 0.0,
            },
        };
        let res = unsafe {
            (self.fns.scene.get_space_bounding_box3_d)(self.session.as_raw(), space, &mut rect)
        };
        xr_ok(res).ok().map(|_| {
            [
                rect.offset.x,
                rect.offset.y,
                rect.offset.z,
                rect.extent.width,
                rect.extent.height,
                rect.extent.depth,
            ]
        })
    }

    fn triangle_mesh(&self, space: sys::Space) -> Option<(Vec<[f32; 3]>, Vec<u32>)> {
        let get_mesh = self.fns.mesh?.get_space_triangle_mesh;
        let info = sys::SpaceTriangleMeshGetInfoMETA {
            ty: sys::SpaceTriangleMeshGetInfoMETA::TYPE,
            next: ptr::null(),
        };
        let mut mesh = sys::SpaceTriangleMeshMETA {
            ty: sys::SpaceTriangleMeshMETA::TYPE,
            next: ptr::null_mut(),
            vertex_capacity_input: 0,
            vertex_count_output: 0,
            vertices: ptr::null_mut(),
            index_capacity_input: 0,
            index_count_output: 0,
            indices: ptr::null_mut(),
        };
        unsafe {
            xr_ok(get_mesh(space, &info, &mut mesh)).ok()?;
            let mut vertices = vec![sys::Vector3f::default(); mesh.vertex_count_output as usize];
            let mut indices = vec![0u32; mesh.index_count_output as usize];
            mesh.vertex_capacity_input = vertices.len() as u32;
            mesh.vertices = vertices.as_mut_ptr();
            mesh.index_capacity_input = indices.len() as u32;
            mesh.indices = indices.as_mut_ptr();
            xr_ok(get_mesh(space, &info, &mut mesh)).ok()?;
            vertices.truncate(mesh.vertex_count_output as usize);
            indices.truncate(mesh.index_count_output as usize);

            Some((vertices.iter().map(|v| [v.x, v.y, v.z]).collect(), indices))
        }
    }

    fn build_snapshot(&self, time: xr::Time) -> SceneSnapshot {
        let State::QueryingAnchors { rooms, anchors, .. } = &self.state else {
            return SceneSnapshot::default();
        };

        let mut snapshot = SceneSnapshot {
            rooms: rooms.clone(),
            ..Default::default()
        };
        for anchor in anchors {
            let space = anchor.space;
            let enabled = |c| self.component_enabled(space, c);

            let pose = enabled(sys::SpaceComponentTypeFB::LOCATABLE)
                .then(|| self.locate(space, time))
                .flatten();
            let labels = if enabled(sys::SpaceComponentTypeFB::SEMANTIC_LABELS) {
                self.semantic_labels(space)
            } else {
                String::new()
            };
            let bounded_2d = enabled(sys::SpaceComponentTypeFB::BOUNDED_2D);

            if enabled(sys::SpaceComponentTypeFB::TRIANGLE_MESH_M) {
                match (pose, self.triangle_mesh(space)) {
                    (Some(pose), Some((vertices, indices))) => snapshot.meshes.push(SceneMesh {
                        anchor_uuid: anchor.uuid,
                        pose,
                        vertices,
                        indices,
                    }),
                    (None, _) => warn!("[SCENE] mesh anchor {labels} not located, mesh skipped"),
                    (_, None) => warn!("[SCENE] xrGetSpaceTriangleMeshMETA failed for {labels}"),
                }
            }

            snapshot.anchors.push(SceneAnchor {
                uuid: anchor.uuid,
                bbox2d: bounded_2d.then(|| self.bounding_box_2d(space)).flatten(),
                boundary2d: bounded_2d.then(|| self.boundary_2d(space)).flatten(),
                bbox3d: enabled(sys::SpaceComponentTypeFB::BOUNDED_3D)
                    .then(|| self.bounding_box_3d(space))
                    .flatten(),
                labels,
                pose,
            });
        }

        let located = snapshot.anchors.iter().filter(|a| a.pose.is_some()).count();
        let triangles: usize = snapshot.meshes.iter().map(|m| m.indices.len() / 3).sum();
        info!(
            "[SCENE] snapshot: {} rooms, {} anchors ({located} located), {} meshes / {triangles} triangles",
            snapshot.rooms.len(),
            snapshot.anchors.len(),
            snapshot.meshes.len(),
        );

        snapshot
    }
}

impl Drop for SceneLoader {
    fn drop(&mut self) {
        for space in self.loaded_spaces.drain(..) {
            unsafe {
                (self.session.instance().fp().destroy_space)(space);
            }
        }
    }
}
