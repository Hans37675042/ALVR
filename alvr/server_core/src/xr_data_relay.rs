use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::TcpStream,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use alvr_common::{Pose, info, warn};
use alvr_packets::{CameraFrameHeader, DepthFrameHeader, SceneSnapshot, scene_uuid_string};
use serde_json::json;

pub const DEFAULT_VIEWER_PORT: u16 = 9944;
const RETRY_INTERVAL: Duration = Duration::from_secs(5);
// Message type 1 (legacy depth layout with a single head pose and FOV angles mislabeled as
// intrinsics) is no longer sent; viewers must parse MSG_DEPTH_FRAME_V2.
const MSG_CAMERA_FRAME: u32 = 2;
const MSG_DEPTH_FRAME_V2: u32 = 3;
pub const MSG_ROOM_SNAPSHOT: u32 = 4;
pub const MSG_PLAYSPACE_CHANGED: u32 = 5;
const MSG_STREAM_CONTROL: u32 = 100;
pub const MSG_ROOM_REQUEST: u32 = 101;

#[derive(Clone, Debug)]
pub struct XrStreamFlags {
    pub depth_enabled: bool,
    pub camera_enabled: bool,
}

impl Default for XrStreamFlags {
    fn default() -> Self {
        Self {
            depth_enabled: false,
            camera_enabled: false,
        }
    }
}

pub type StreamControlCallback = Box<dyn Fn(bool, bool) + Send + Sync>;
/// Called with the `recapture` flag of a MSG_ROOM_REQUEST.
pub type SceneRequestCallback = Box<dyn Fn(bool) + Send + Sync>;

/// Latest scene received from the client, kept in client space so it can be re-sent with
/// whatever recentering is current.
struct SceneCache {
    snapshot: Option<(u32, SceneSnapshot)>,
    next_snapshot_id: u32,
    recenter_pose: Pose,
}

struct Shared {
    connection: Mutex<Option<TcpStream>>,
    stream_flags: Mutex<XrStreamFlags>,
    control_callback: Mutex<Option<StreamControlCallback>>,
    scene_request_callback: Mutex<Option<SceneRequestCallback>>,
    scene: Mutex<SceneCache>,
}

impl Shared {
    fn broadcast(&self, msg_type: u32, payload: &[u8]) {
        let mut conn = self.connection.lock().unwrap();

        if let Some(stream) = conn.as_mut() {
            let mut header = [0u8; 8];
            header[0..4].copy_from_slice(&msg_type.to_le_bytes());
            header[4..8].copy_from_slice(&(payload.len() as u32).to_le_bytes());

            if stream.write_all(&header).is_err() || stream.write_all(payload).is_err() {
                warn!("XR Data Relay: viewer disconnected, will reconnect");
                *conn = None;
            }
        }
    }

    // The scene lock is held while sending so that a concurrent recenter or new snapshot cannot
    // be overtaken by an older one.
    fn send_cached_scene(&self) {
        let scene = self.scene.lock().unwrap();
        if let Some((id, snapshot)) = &scene.snapshot {
            let payload = encode_room_snapshot(*id, scene.recenter_pose, snapshot);
            self.broadcast(MSG_ROOM_SNAPSHOT, &payload);
        }
    }
}

pub struct XrDataRelay {
    shared: Arc<Shared>,
    _connector_thread: thread::JoinHandle<()>,
}

impl XrDataRelay {
    pub fn new(port: u16) -> Self {
        let viewer_addr = format!("127.0.0.1:{port}");
        let shared = Arc::new(Shared {
            connection: Mutex::new(None),
            stream_flags: Mutex::new(XrStreamFlags::default()),
            control_callback: Mutex::new(None),
            scene_request_callback: Mutex::new(None),
            scene: Mutex::new(SceneCache {
                snapshot: None,
                next_snapshot_id: 1,
                recenter_pose: Pose::IDENTITY,
            }),
        });

        let connector_thread = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                loop {
                    // Only try to connect if we don't already have a connection
                    if shared.connection.lock().unwrap().is_some() {
                        thread::sleep(RETRY_INTERVAL);
                        continue;
                    }

                    match TcpStream::connect(&viewer_addr) {
                        Ok(stream) => {
                            info!("XR Data Relay connected to viewer at {viewer_addr}");
                            stream.set_nodelay(true).ok();
                            stream.set_write_timeout(Some(Duration::from_millis(500))).ok();

                            // Spawn reader thread for incoming commands from viewer
                            if let Ok(reader_stream) = stream.try_clone() {
                                let shared = Arc::clone(&shared);
                                thread::spawn(move || {
                                    read_viewer_commands(reader_stream, &shared);
                                });
                            }

                            *shared.connection.lock().unwrap() = Some(stream);

                            // A new viewer has no room model yet
                            shared.send_cached_scene();
                        }
                        Err(_) => {
                            // Viewer not running yet, retry silently
                        }
                    }

                    thread::sleep(RETRY_INTERVAL);
                }
            })
        };

        Self {
            shared,
            _connector_thread: connector_thread,
        }
    }

    pub fn set_control_callback(&self, cb: StreamControlCallback) {
        *self.shared.control_callback.lock().unwrap() = Some(cb);
    }

    pub fn set_scene_request_callback(&self, cb: SceneRequestCallback) {
        *self.shared.scene_request_callback.lock().unwrap() = Some(cb);
    }

    pub fn send_depth_frame(&self, header: &DepthFrameHeader, timing: &DepthTiming, data: &[u8]) {
        if !self.shared.stream_flags.lock().unwrap().depth_enabled {
            return;
        }
        let payload = encode_depth_frame(header, timing, data);
        self.shared.broadcast(MSG_DEPTH_FRAME_V2, &payload);
    }

    pub fn send_camera_frame(&self, header: &CameraFrameHeader, data: &[u8]) {
        if !self.shared.stream_flags.lock().unwrap().camera_enabled {
            return;
        }
        let payload = encode_camera_frame(header, data);
        self.shared.broadcast(MSG_CAMERA_FRAME, &payload);
    }

    /// Cache a scene snapshot (client space) and send it to the viewer, recentered. Scene
    /// messages are not gated by the stream flags. Returns the assigned snapshot id.
    pub fn set_scene_snapshot(&self, snapshot: SceneSnapshot) -> u32 {
        let id = {
            let mut scene = self.shared.scene.lock().unwrap();
            let id = scene.next_snapshot_id;
            scene.next_snapshot_id = scene.next_snapshot_id.wrapping_add(1).max(1);
            scene.snapshot = Some((id, snapshot));
            id
        };
        self.shared.send_cached_scene();

        id
    }

    /// Mirror of `TrackingManager::recenter_pose`'s transform. Sends MSG_PLAYSPACE_CHANGED and
    /// re-sends the cached snapshot in the new space.
    pub fn set_recenter_pose(&self, recenter_pose: Pose) {
        self.shared.scene.lock().unwrap().recenter_pose = recenter_pose;
        self.shared
            .broadcast(MSG_PLAYSPACE_CHANGED, &encode_playspace_changed(recenter_pose));
        self.shared.send_cached_scene();
    }
}

fn read_viewer_commands(mut stream: TcpStream, shared: &Shared) {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();

    loop {
        let mut header = [0u8; 8];
        match stream.read_exact(&mut header) {
            Ok(()) => {}
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut => {
                // Check if the write side is still connected
                if shared.connection.lock().unwrap().is_none() {
                    return;
                }
                continue;
            }
            Err(_) => {
                info!("XR Data Relay: viewer reader disconnected");
                *shared.connection.lock().unwrap() = None;
                return;
            }
        }

        let msg_type = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let payload_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            if stream.read_exact(&mut payload).is_err() {
                *shared.connection.lock().unwrap() = None;
                return;
            }
        }

        if msg_type == MSG_STREAM_CONTROL && payload_len >= 2 {
            let stream_id = payload[0];
            let enabled = payload[1] != 0;

            let mut f = shared.stream_flags.lock().unwrap();
            match stream_id {
                0 => f.depth_enabled = enabled,
                1 => f.camera_enabled = enabled,
                _ => {}
            }
            let snapshot = f.clone();
            drop(f);

            info!(
                "XR Data Relay: stream control update — depth={}, camera={}",
                snapshot.depth_enabled, snapshot.camera_enabled
            );

            if let Some(cb) = shared.control_callback.lock().unwrap().as_ref() {
                cb(snapshot.depth_enabled, snapshot.camera_enabled);
            }
        } else if msg_type == MSG_ROOM_REQUEST {
            let Some(recapture) = parse_room_request(&payload) else {
                warn!("XR Data Relay: empty MSG_ROOM_REQUEST ignored");
                continue;
            };
            info!("XR Data Relay: room request from viewer (recapture={recapture})");

            // Answer right away from the cache; the client query refreshes it afterwards
            if !recapture {
                shared.send_cached_scene();
            }
            if let Some(cb) = shared.scene_request_callback.lock().unwrap().as_ref() {
                cb(recapture);
            }
        }
    }
}

/// MSG_ROOM_REQUEST payload: `u8 recapture`.
pub fn parse_room_request(payload: &[u8]) -> Option<bool> {
    payload.first().map(|&b| b != 0)
}

/// Bring every anchor and mesh pose of a client-space snapshot into the recentered space.
/// Bounds and mesh vertices are anchor-local and stay unchanged.
pub fn recenter_scene(snapshot: &SceneSnapshot, recenter_pose: Pose) -> SceneSnapshot {
    let mut out = snapshot.clone();
    for anchor in &mut out.anchors {
        anchor.pose = anchor.pose.map(|pose| recenter_pose * pose);
    }
    for mesh in &mut out.meshes {
        mesh.pose = recenter_pose * mesh.pose;
    }

    out
}

fn pose_position_first(pose: &Pose) -> [f32; 7] {
    [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
}

fn put_pose_position_first(buf: &mut Vec<u8>, pose: &Pose) {
    for v in pose_position_first(pose) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Scene JSON of MSG_ROOM_SNAPSHOT (see CONTRACT-roomd.md). Poses are written as given,
/// `[px, py, pz, qx, qy, qz, qw]`; UUIDs use `scene_uuid_string`.
pub fn scene_snapshot_json(snapshot: &SceneSnapshot) -> String {
    let rooms = snapshot
        .rooms
        .iter()
        .map(|room| {
            json!({
                "uuid": scene_uuid_string(&room.uuid),
                "floor": room.floor.as_ref().map(scene_uuid_string),
                "ceiling": room.ceiling.as_ref().map(scene_uuid_string),
                "walls": room.walls.iter().map(scene_uuid_string).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let anchors = snapshot
        .anchors
        .iter()
        .map(|anchor| {
            json!({
                "uuid": scene_uuid_string(&anchor.uuid),
                "labels": anchor.labels,
                "pose": anchor.pose.as_ref().map(pose_position_first),
                "bbox2d": anchor.bbox2d,
                "boundary2d": anchor.boundary2d,
                "bbox3d": anchor.bbox3d,
            })
        })
        .collect::<Vec<_>>();

    json!({ "rooms": rooms, "anchors": anchors }).to_string()
}

/// Size of the fixed MSG_ROOM_SNAPSHOT header; `json_len` follows at this offset.
pub const ROOM_SNAPSHOT_HEADER_SIZE: u32 = 40;
const ROOM_SNAPSHOT_VERSION: u32 = 1;
const PLAYSPACE_CHANGED_VERSION: u32 = 1;

/// MSG_ROOM_SNAPSHOT payload (all little-endian). `snapshot` is in client space; the
/// recentering is applied here.
///
/// | offset | type      | field                                                        |
/// |--------|-----------|--------------------------------------------------------------|
/// | 0      | u32       | header_size (40)                                             |
/// | 4      | u32       | version (1)                                                  |
/// | 8      | u32       | snapshot_id                                                  |
/// | 12     | f32 x 7   | recenter_pose px py pz qx qy qz qw                           |
/// | 40     | u32       | json_len                                                     |
/// | 44     | u8[]      | UTF-8 JSON (`scene_snapshot_json`, recentered poses)         |
/// |        | u32       | mesh_count                                                   |
/// |        | per mesh  | u8[16] anchor_uuid, f32 x 7 pose (px..qw, recentered),       |
/// |        |           | u32 vcount, u32 icount, f32 x 3 [vcount], u32 [icount]       |
pub fn encode_room_snapshot(snapshot_id: u32, recenter_pose: Pose, snapshot: &SceneSnapshot) -> Vec<u8> {
    let recentered = recenter_scene(snapshot, recenter_pose);
    let json = scene_snapshot_json(&recentered);
    let mesh_bytes: usize = recentered
        .meshes
        .iter()
        .map(|m| 52 + m.vertices.len() * 12 + m.indices.len() * 4)
        .sum();

    let mut buf =
        Vec::with_capacity(ROOM_SNAPSHOT_HEADER_SIZE as usize + 8 + json.len() + mesh_bytes);
    buf.extend_from_slice(&ROOM_SNAPSHOT_HEADER_SIZE.to_le_bytes());
    buf.extend_from_slice(&ROOM_SNAPSHOT_VERSION.to_le_bytes());
    buf.extend_from_slice(&snapshot_id.to_le_bytes());
    put_pose_position_first(&mut buf, &recenter_pose);
    debug_assert_eq!(buf.len(), ROOM_SNAPSHOT_HEADER_SIZE as usize);

    buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
    buf.extend_from_slice(json.as_bytes());

    buf.extend_from_slice(&(recentered.meshes.len() as u32).to_le_bytes());
    for mesh in &recentered.meshes {
        buf.extend_from_slice(&mesh.anchor_uuid);
        put_pose_position_first(&mut buf, &mesh.pose);
        buf.extend_from_slice(&(mesh.vertices.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(mesh.indices.len() as u32).to_le_bytes());
        for vertex in &mesh.vertices {
            for v in vertex {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for index in &mesh.indices {
            buf.extend_from_slice(&index.to_le_bytes());
        }
    }

    buf
}

/// MSG_PLAYSPACE_CHANGED payload: `u32 version, f32 x 7 recenter_pose (px..qw)`.
pub fn encode_playspace_changed(recenter_pose: Pose) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);
    buf.extend_from_slice(&PLAYSPACE_CHANGED_VERSION.to_le_bytes());
    put_pose_position_first(&mut buf, &recenter_pose);

    buf
}

/// Current server wall-clock time as nanoseconds since the Unix epoch.
pub fn unix_now_ns() -> i128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0)
}

/// Estimates `server_unix_ns - client_xr_ns` from (client send time, server receive time) pairs.
///
/// Each sample equals the true clock offset plus that packet's one-way latency, so the minimum
/// over a sliding window is the best estimate (the residual bias is the minimum one-way latency,
/// typically a few ms over USB/Wi-Fi). The window lets the estimate follow clock drift and
/// headset sleep/resume.
pub struct ClientClockOffsetEstimator {
    samples: VecDeque<(Instant, i128)>,
    window: Duration,
}

impl ClientClockOffsetEstimator {
    pub fn new(window: Duration) -> Self {
        Self {
            samples: VecDeque::new(),
            window,
        }
    }

    /// Record one sample and return the current offset estimate in nanoseconds.
    pub fn report(
        &mut self,
        now: Instant,
        server_receive_unix_ns: i128,
        client_send_ns: i128,
    ) -> i128 {
        self.samples
            .push_back((now, server_receive_unix_ns - client_send_ns));
        while let Some(&(t, _)) = self.samples.front() {
            if self.samples.len() > 1 && now.saturating_duration_since(t) > self.window {
                self.samples.pop_front();
            } else {
                break;
            }
        }

        self.samples.iter().map(|&(_, offset)| offset).min().unwrap()
    }
}

/// Server-side timing attached to a relayed depth frame.
#[derive(Clone, Copy, Debug)]
pub struct DepthTiming {
    /// `header.timestamp` mapped to the server clock (Unix epoch ns).
    pub server_timestamp_unix_ns: u64,
    /// Estimated `server_unix_ns - client_xr_ns`.
    pub clock_offset_ns: i64,
    /// When the server received the frame (Unix epoch ns).
    pub server_receive_unix_ns: u64,
}

impl DepthTiming {
    pub fn new(
        estimator: &mut ClientClockOffsetEstimator,
        header: &DepthFrameHeader,
        now: Instant,
        server_receive_unix_ns: i128,
    ) -> Self {
        let offset = estimator.report(
            now,
            server_receive_unix_ns,
            header.client_send_time.as_nanos() as i128,
        );
        let server_timestamp = header.timestamp.as_nanos() as i128 + offset;

        Self {
            server_timestamp_unix_ns: server_timestamp.max(0) as u64,
            clock_offset_ns: offset as i64,
            server_receive_unix_ns: server_receive_unix_ns.max(0) as u64,
        }
    }
}

fn put_pose(buf: &mut Vec<u8>, pose: &alvr_common::Pose) {
    // qx, qy, qz, qw, px, py, pz
    for v in [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Size of the MSG_DEPTH_FRAME_V2 header. New fields are only ever appended, and the header
/// starts with its own size, so viewers can skip fields they do not know.
pub const DEPTH_FRAME_V2_HEADER_SIZE: u32 = 176;

/// MSG_DEPTH_FRAME_V2 payload layout (all little-endian):
///
/// | offset | type      | field                                                        |
/// |--------|-----------|--------------------------------------------------------------|
/// | 0      | u32       | header_size (bytes, including this field; pixel data follows) |
/// | 4      | u64       | client_timestamp_ns (client XR time of the requested frame)  |
/// | 12     | f32 x 7   | view_pose[0] qx qy qz qw px py pz (recentered stage space)   |
/// | 40     | f32 x 7   | view_pose[1]                                                 |
/// | 68     | u32       | width                                                        |
/// | 72     | u32       | height (stacked: view 0 top half, view 1 bottom half)        |
/// | 76     | f32       | near_z (m)                                                   |
/// | 80     | f32       | far_z (m, may be +inf)                                       |
/// | 84     | u32       | format (0 = RawD16, 1 = H264, 2 = Lz4D16)                    |
/// | 88     | f32 x 4   | fov_angles[0] left right up down (radians)                   |
/// | 104    | f32 x 4   | fov_angles[1]                                                |
/// | 120    | f32 x 4   | intrinsics[0] fx fy cx cy (pixels, per-view image, y down)   |
/// | 136    | f32 x 4   | intrinsics[1]                                                |
/// | 152    | u64       | server_timestamp_unix_ns (client_timestamp on server clock)  |
/// | 160    | i64       | clock_offset_ns (server_unix_ns - client_xr_ns, estimated)   |
/// | 168    | u64       | server_receive_unix_ns                                       |
/// | 176    | u8[]      | pixel data                                                   |
pub fn encode_depth_frame(header: &DepthFrameHeader, timing: &DepthTiming, data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(DEPTH_FRAME_V2_HEADER_SIZE as usize + data.len());

    buf.extend_from_slice(&DEPTH_FRAME_V2_HEADER_SIZE.to_le_bytes());
    buf.extend_from_slice(&(header.timestamp.as_nanos() as u64).to_le_bytes());

    for pose in &header.view_poses {
        put_pose(&mut buf, pose);
    }

    buf.extend_from_slice(&header.width.to_le_bytes());
    buf.extend_from_slice(&header.height.to_le_bytes());
    buf.extend_from_slice(&header.near_z.to_le_bytes());
    buf.extend_from_slice(&header.far_z.to_le_bytes());

    let format_id: u32 = match &header.format {
        alvr_packets::DepthFrameFormat::RawD16 => 0,
        alvr_packets::DepthFrameFormat::H264 => 1,
        alvr_packets::DepthFrameFormat::Lz4D16 => 2,
    };
    buf.extend_from_slice(&format_id.to_le_bytes());

    for fov in &header.fov_angles {
        for val in fov {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

    for view in 0..2 {
        for val in header.view_intrinsics(view) {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

    buf.extend_from_slice(&timing.server_timestamp_unix_ns.to_le_bytes());
    buf.extend_from_slice(&timing.clock_offset_ns.to_le_bytes());
    buf.extend_from_slice(&timing.server_receive_unix_ns.to_le_bytes());

    debug_assert_eq!(buf.len(), DEPTH_FRAME_V2_HEADER_SIZE as usize);

    buf.extend_from_slice(data);
    buf
}

fn encode_camera_frame(header: &CameraFrameHeader, data: &[u8]) -> Vec<u8> {
    // Header layout:
    // timestamp(8) + pose(7*4=28) + width(4) + height(4) + format(4) + intrinsics(4*4=16) = 64 bytes
    // Then raw pixel data
    let mut buf = Vec::with_capacity(64 + data.len());

    buf.extend_from_slice(&header.timestamp.as_nanos().to_le_bytes()[..8]);

    // pose: qx, qy, qz, qw, px, py, pz
    buf.extend_from_slice(&header.head_pose.orientation.x.to_le_bytes());
    buf.extend_from_slice(&header.head_pose.orientation.y.to_le_bytes());
    buf.extend_from_slice(&header.head_pose.orientation.z.to_le_bytes());
    buf.extend_from_slice(&header.head_pose.orientation.w.to_le_bytes());
    buf.extend_from_slice(&header.head_pose.position.x.to_le_bytes());
    buf.extend_from_slice(&header.head_pose.position.y.to_le_bytes());
    buf.extend_from_slice(&header.head_pose.position.z.to_le_bytes());

    buf.extend_from_slice(&header.width.to_le_bytes());
    buf.extend_from_slice(&header.height.to_le_bytes());
    let format_id: u32 = match &header.format {
        alvr_packets::CameraFrameFormat::Jpeg => 0,
        alvr_packets::CameraFrameFormat::Nv12 => 1,
        alvr_packets::CameraFrameFormat::Rgb8 => 2,
        alvr_packets::CameraFrameFormat::H264 => 3,
    };
    buf.extend_from_slice(&format_id.to_le_bytes());

    for val in &header.intrinsics {
        buf.extend_from_slice(&val.to_le_bytes());
    }

    buf.extend_from_slice(data);
    buf
}


#[cfg(test)]
mod tests {
    use super::*;
    use alvr_common::{Pose, glam::{Quat, Vec3}};
    use alvr_packets::{
        DepthFrameFormat, SceneAnchor, SceneMesh, SceneRoom, SceneSnapshot, SceneUuid,
        scene_uuid_string,
    };

    fn f32_at(buf: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn u32_at(buf: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn sample_header() -> DepthFrameHeader {
        DepthFrameHeader {
            timestamp: Duration::from_nanos(123_456_789),
            client_send_time: Duration::from_nanos(100_000_000),
            view_poses: [
                Pose {
                    orientation: Quat::IDENTITY,
                    position: Vec3::new(-0.03, 1.6, 0.0),
                },
                Pose {
                    orientation: Quat::from_rotation_y(0.1),
                    position: Vec3::new(0.03, 1.6, 0.0),
                },
            ],
            width: 320,
            height: 640,
            near_z: 0.1,
            far_z: f32::INFINITY,
            format: DepthFrameFormat::Lz4D16,
            fov_angles: [[-0.9, 0.6, 0.7, -0.8], [-0.6, 0.9, 0.7, -0.8]],
        }
    }

    #[test]
    fn depth_v2_layout_matches_documented_offsets() {
        let header = sample_header();
        let data = [1u8, 2, 3];
        let timing = DepthTiming {
            server_timestamp_unix_ns: 1_700_000_000_000_000_000,
            clock_offset_ns: -5,
            server_receive_unix_ns: 1_700_000_000_100_000_000,
        };
        let buf = encode_depth_frame(&header, &timing, &data);

        assert_eq!(buf.len(), DEPTH_FRAME_V2_HEADER_SIZE as usize + data.len());
        assert_eq!(u32_at(&buf, 0), DEPTH_FRAME_V2_HEADER_SIZE);
        assert_eq!(
            u64::from_le_bytes(buf[4..12].try_into().unwrap()),
            123_456_789
        );

        // view_pose[0].px and view_pose[1].qy / px
        assert_eq!(f32_at(&buf, 12 + 4 * 4), -0.03);
        assert_eq!(f32_at(&buf, 40 + 4), header.view_poses[1].orientation.y);
        assert_eq!(f32_at(&buf, 40 + 4 * 4), 0.03);

        assert_eq!(u32_at(&buf, 68), 320);
        assert_eq!(u32_at(&buf, 72), 640);
        assert_eq!(f32_at(&buf, 76), 0.1);
        assert!(f32_at(&buf, 80).is_infinite());
        assert_eq!(u32_at(&buf, 84), 2);

        assert_eq!(f32_at(&buf, 88), -0.9);
        assert_eq!(f32_at(&buf, 104), -0.6);

        let intr1 = header.view_intrinsics(1);
        assert_eq!(f32_at(&buf, 120), header.view_intrinsics(0)[0]);
        assert_eq!(f32_at(&buf, 136 + 8), intr1[2]);

        assert_eq!(
            u64::from_le_bytes(buf[152..160].try_into().unwrap()),
            timing.server_timestamp_unix_ns
        );
        assert_eq!(i64::from_le_bytes(buf[160..168].try_into().unwrap()), -5);
        assert_eq!(
            u64::from_le_bytes(buf[168..176].try_into().unwrap()),
            timing.server_receive_unix_ns
        );

        assert_eq!(&buf[DEPTH_FRAME_V2_HEADER_SIZE as usize..], &data);
    }

    #[test]
    fn clock_offset_uses_minimum_latency_sample_in_window() {
        let mut est = ClientClockOffsetEstimator::new(Duration::from_secs(10));
        let t0 = Instant::now();
        // true offset 1_000_000 ns; one-way latencies 30 ms, 4 ms, 12 ms
        assert_eq!(est.report(t0, 1_000_000 + 30_000_000, 0), 31_000_000);
        assert_eq!(
            est.report(t0 + Duration::from_secs(1), 1_000_000 + 4_000_000 + 1_000, 1_000),
            5_000_000
        );
        assert_eq!(
            est.report(t0 + Duration::from_secs(2), 1_000_000 + 12_000_000 + 2_000, 2_000),
            5_000_000
        );
    }

    #[test]
    fn clock_offset_forgets_samples_outside_window() {
        let mut est = ClientClockOffsetEstimator::new(Duration::from_secs(10));
        let t0 = Instant::now();
        est.report(t0, 5, 0); // stale sample with a very low offset
        let offset = est.report(t0 + Duration::from_secs(11), 1_000, 0);
        assert_eq!(offset, 1_000);
    }

    fn uuid(n: u8) -> SceneUuid {
        let mut u = [0u8; 16];
        u[0] = n;
        u[15] = 0xab;
        u
    }

    fn sample_scene() -> SceneSnapshot {
        SceneSnapshot {
            rooms: vec![SceneRoom {
                uuid: uuid(1),
                floor: Some(uuid(2)),
                ceiling: None,
                walls: vec![uuid(3), uuid(4)],
            }],
            anchors: vec![
                SceneAnchor {
                    uuid: uuid(5),
                    labels: "TABLE".into(),
                    pose: Some(Pose {
                        orientation: Quat::from_rotation_y(0.5),
                        position: Vec3::new(1.0, 0.7, -2.0),
                    }),
                    bbox2d: Some([-0.5, -0.25, 1.0, 0.5]),
                    boundary2d: Some(vec![[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25]]),
                    bbox3d: Some([-0.5, -0.25, -0.7, 1.0, 0.5, 0.7]),
                },
                SceneAnchor {
                    uuid: uuid(6),
                    labels: "GLOBAL_MESH".into(),
                    pose: None,
                    bbox2d: None,
                    boundary2d: None,
                    bbox3d: None,
                },
            ],
            meshes: vec![SceneMesh {
                anchor_uuid: uuid(6),
                pose: Pose {
                    orientation: Quat::IDENTITY,
                    position: Vec3::new(0.0, 0.0, 0.5),
                },
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                indices: vec![0, 1, 2],
            }],
        }
    }

    fn sample_recenter() -> Pose {
        Pose {
            orientation: Quat::from_rotation_y(1.0),
            position: Vec3::new(0.3, 0.0, -0.4),
        }
    }

    fn pose_at(buf: &[u8], offset: usize) -> [f32; 7] {
        std::array::from_fn(|i| f32_at(buf, offset + 4 * i))
    }

    fn pose_array(pose: Pose) -> [f32; 7] {
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    }

    fn assert_pose_close(a: [f32; 7], b: [f32; 7]) {
        for i in 0..7 {
            assert!((a[i] - b[i]).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn recentered_scene_moves_anchor_and_mesh_poses_only() {
        let scene = sample_scene();
        let recenter = sample_recenter();
        let out = recenter_scene(&scene, recenter);

        let expected = recenter * scene.anchors[0].pose.unwrap();
        assert_pose_close(pose_array(out.anchors[0].pose.unwrap()), pose_array(expected));
        assert!(out.anchors[1].pose.is_none());
        assert_pose_close(
            pose_array(out.meshes[0].pose),
            pose_array(recenter * scene.meshes[0].pose),
        );

        // Bounds and mesh vertices are anchor-local and must not change
        assert_eq!(out.anchors[0].bbox3d, scene.anchors[0].bbox3d);
        assert_eq!(out.anchors[0].boundary2d, scene.anchors[0].boundary2d);
        assert_eq!(out.meshes[0].vertices, scene.meshes[0].vertices);
        assert_eq!(out.rooms[0].walls, scene.rooms[0].walls);
    }

    #[test]
    fn scene_json_matches_contract() {
        let scene = sample_scene();
        let json: serde_json::Value =
            serde_json::from_str(&scene_snapshot_json(&scene)).unwrap();

        let room = &json["rooms"][0];
        assert_eq!(room["uuid"], scene_uuid_string(&uuid(1)));
        assert_eq!(room["floor"], scene_uuid_string(&uuid(2)));
        assert!(room["ceiling"].is_null());
        assert_eq!(room["walls"].as_array().unwrap().len(), 2);
        assert_eq!(room["walls"][1], scene_uuid_string(&uuid(4)));

        let table = &json["anchors"][0];
        assert_eq!(table["uuid"], "05000000-0000-0000-0000-0000000000ab");
        assert_eq!(table["labels"], "TABLE");
        let pose = table["pose"].as_array().unwrap();
        assert_eq!(pose.len(), 7);
        // [px, py, pz, qx, qy, qz, qw]
        assert!((pose[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((pose[2].as_f64().unwrap() + 2.0).abs() < 1e-6);
        assert!((pose[6].as_f64().unwrap() - 0.25f64.cos()).abs() < 1e-6);
        assert_eq!(table["bbox2d"].as_array().unwrap().len(), 4);
        assert_eq!(table["boundary2d"].as_array().unwrap().len(), 3);
        assert_eq!(table["boundary2d"][1].as_array().unwrap().len(), 2);
        assert_eq!(table["bbox3d"].as_array().unwrap().len(), 6);

        let mesh_anchor = &json["anchors"][1];
        assert_eq!(mesh_anchor["labels"], "GLOBAL_MESH");
        for key in ["pose", "bbox2d", "boundary2d", "bbox3d"] {
            assert!(mesh_anchor[key].is_null(), "{key}");
        }
    }

    #[test]
    fn room_snapshot_layout_matches_contract() {
        let scene = sample_scene();
        let recenter = sample_recenter();
        let buf = encode_room_snapshot(7, recenter, &scene);

        assert_eq!(u32_at(&buf, 0), ROOM_SNAPSHOT_HEADER_SIZE);
        assert_eq!(ROOM_SNAPSHOT_HEADER_SIZE, 40);
        assert_eq!(u32_at(&buf, 4), 1); // version
        assert_eq!(u32_at(&buf, 8), 7); // snapshot_id
        assert_pose_close(pose_at(&buf, 12), pose_array(recenter));

        let json_len = u32_at(&buf, 40) as usize;
        let json: serde_json::Value = serde_json::from_slice(&buf[44..44 + json_len]).unwrap();
        // Poses in the JSON are already recentered
        let table_pose: Vec<f32> = json["anchors"][0]["pose"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_pose_close(
            table_pose.try_into().unwrap(),
            pose_array(recenter * scene.anchors[0].pose.unwrap()),
        );

        let mut off = 44 + json_len;
        assert_eq!(u32_at(&buf, off), 1); // mesh_count
        off += 4;
        assert_eq!(&buf[off..off + 16], &uuid(6));
        off += 16;
        assert_pose_close(pose_at(&buf, off), pose_array(recenter * scene.meshes[0].pose));
        off += 28;
        assert_eq!(u32_at(&buf, off), 3); // vcount
        assert_eq!(u32_at(&buf, off + 4), 3); // icount
        off += 8;
        assert_eq!(f32_at(&buf, off + 12), 1.0); // vertex 1 x
        assert_eq!(f32_at(&buf, off + 28), 1.0); // vertex 2 y
        off += 36;
        assert_eq!(u32_at(&buf, off + 8), 2); // index 2
        off += 12;
        assert_eq!(buf.len(), off);
    }

    #[test]
    fn empty_room_snapshot_has_empty_lists_and_no_meshes() {
        let buf = encode_room_snapshot(1, Pose::IDENTITY, &SceneSnapshot::default());
        let json_len = u32_at(&buf, 40) as usize;
        let json: serde_json::Value = serde_json::from_slice(&buf[44..44 + json_len]).unwrap();
        assert_eq!(json["rooms"].as_array().unwrap().len(), 0);
        assert_eq!(json["anchors"].as_array().unwrap().len(), 0);
        assert_eq!(u32_at(&buf, 44 + json_len), 0);
        assert_eq!(buf.len(), 48 + json_len);
    }

    #[test]
    fn playspace_changed_layout_matches_contract() {
        let recenter = sample_recenter();
        let buf = encode_playspace_changed(recenter);
        assert_eq!(buf.len(), 32);
        assert_eq!(u32_at(&buf, 0), 1); // version
        assert_pose_close(pose_at(&buf, 4), pose_array(recenter));
    }

    #[test]
    fn room_request_payload_parses_recapture_flag() {
        assert_eq!(parse_room_request(&[0]), Some(false));
        assert_eq!(parse_room_request(&[1]), Some(true));
        assert_eq!(parse_room_request(&[]), None);
    }

    fn tcp_pair() -> (TcpStream, TcpStream) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let a = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (b, _) = listener.accept().unwrap();
        (a, b)
    }

    #[test]
    fn stale_reader_cannot_clear_newer_connection() {
        let slot = ConnectionSlot::default();
        let (old, mut old_peer) = tcp_pair();
        let old_gen = slot.replace(old);
        let (new, _new_peer) = tcp_pair();
        let new_gen = slot.replace(new);

        assert!(!slot.clear_if(old_gen));
        assert!(slot.is_current(new_gen));
        assert!(slot.clear_if(new_gen));
        assert!(!slot.is_current(new_gen));

        // The replaced socket was shut down, so its viewer sees EOF
        old_peer
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut buf = [0u8; 1];
        assert_eq!(old_peer.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn failed_write_shuts_the_socket_down() {
        let slot = ConnectionSlot::default();
        let (stream, mut peer) = tcp_pair();
        let generation = slot.replace(stream);
        slot.drop_connection();
        assert!(!slot.is_current(generation));
        peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut buf = [0u8; 1];
        assert_eq!(peer.read(&mut buf).unwrap(), 0);
    }

    fn read_msg(stream: &mut TcpStream) -> (u32, Vec<u8>) {
        let mut header = [0u8; 8];
        stream.read_exact(&mut header).unwrap();
        let len = u32_at(&header, 4) as usize;
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).unwrap();
        (u32_at(&header, 0), payload)
    }

    #[test]
    fn relay_resends_scene_on_connect_and_recenter_and_forwards_requests() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let relay = XrDataRelay::new(port);
        let requests = Arc::new(Mutex::new(Vec::new()));
        relay.set_scene_request_callback(Box::new({
            let requests = Arc::clone(&requests);
            move |recapture| requests.lock().unwrap().push(recapture)
        }));
        // The snapshot may arrive before or while the viewer connects; it must be delivered
        relay.set_scene_snapshot(sample_scene());

        let (mut viewer, _) = listener.accept().unwrap();
        viewer
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let (ty, payload) = read_msg(&mut viewer);
        assert_eq!(ty, MSG_ROOM_SNAPSHOT);
        let first_id = u32_at(&payload, 8);
        assert_pose_close(pose_at(&payload, 12), pose_array(Pose::IDENTITY));

        relay.set_recenter_pose(sample_recenter());
        // Skip a possible duplicate of the first snapshot (sent by both connect and set)
        let mut msg = read_msg(&mut viewer);
        while msg.0 == MSG_ROOM_SNAPSHOT && pose_at(&msg.1, 12) == pose_array(Pose::IDENTITY) {
            msg = read_msg(&mut viewer);
        }
        assert_eq!(msg.0, MSG_PLAYSPACE_CHANGED);
        assert_pose_close(pose_at(&msg.1, 4), pose_array(sample_recenter()));
        let (ty, payload) = read_msg(&mut viewer);
        assert_eq!(ty, MSG_ROOM_SNAPSHOT);
        assert_eq!(u32_at(&payload, 8), first_id);
        assert_pose_close(pose_at(&payload, 12), pose_array(sample_recenter()));

        // Viewer asks for a Space Setup recapture
        let mut request = Vec::new();
        request.extend_from_slice(&MSG_ROOM_REQUEST.to_le_bytes());
        request.extend_from_slice(&1u32.to_le_bytes());
        request.push(1);
        viewer.write_all(&request).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while requests.lock().unwrap().is_empty() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(*requests.lock().unwrap(), vec![true]);
    }

    #[test]
    fn depth_timing_maps_client_timestamp_to_server_clock() {
        let mut est = ClientClockOffsetEstimator::new(Duration::from_secs(10));
        let header = sample_header(); // timestamp 123_456_789, send 100_000_000
        let recv = 2_000_000_000_i128;
        let timing = DepthTiming::new(&mut est, &header, Instant::now(), recv);
        let offset = recv - 100_000_000;
        assert_eq!(timing.clock_offset_ns as i128, offset);
        assert_eq!(timing.server_timestamp_unix_ns as i128, 123_456_789 + offset);
        assert_eq!(timing.server_receive_unix_ns as i128, recv);
    }
}
