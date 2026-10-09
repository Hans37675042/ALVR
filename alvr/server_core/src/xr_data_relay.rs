use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::TcpStream,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use alvr_common::{info, warn};
use alvr_packets::{CameraFrameHeader, DepthFrameHeader};

pub const DEFAULT_VIEWER_PORT: u16 = 9944;
const RETRY_INTERVAL: Duration = Duration::from_secs(5);
// Message type 1 (legacy depth layout with a single head pose and FOV angles mislabeled as
// intrinsics) is no longer sent; viewers must parse MSG_DEPTH_FRAME_V2.
const MSG_CAMERA_FRAME: u32 = 2;
const MSG_DEPTH_FRAME_V2: u32 = 3;
const MSG_STREAM_CONTROL: u32 = 100;

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

pub struct XrDataRelay {
    connection: Arc<Mutex<Option<TcpStream>>>,
    stream_flags: Arc<Mutex<XrStreamFlags>>,
    _connector_thread: thread::JoinHandle<()>,
    control_callback: Arc<Mutex<Option<StreamControlCallback>>>,
}

impl XrDataRelay {
    pub fn new(port: u16) -> Self {
        let viewer_addr = format!("127.0.0.1:{port}");
        let connection: Arc<Mutex<Option<TcpStream>>> = Arc::new(Mutex::new(None));
        let stream_flags: Arc<Mutex<XrStreamFlags>> = Arc::new(Mutex::new(XrStreamFlags::default()));
        let control_callback: Arc<Mutex<Option<StreamControlCallback>>> = Arc::new(Mutex::new(None));

        let connector_thread = {
            let connection = Arc::clone(&connection);
            let stream_flags = Arc::clone(&stream_flags);
            let control_callback = Arc::clone(&control_callback);
            thread::spawn(move || {
                loop {
                    // Only try to connect if we don't already have a connection
                    {
                        let conn = connection.lock().unwrap();
                        if conn.is_some() {
                            drop(conn);
                            thread::sleep(RETRY_INTERVAL);
                            continue;
                        }
                    }

                    match TcpStream::connect(&viewer_addr) {
                        Ok(stream) => {
                            info!("XR Data Relay connected to viewer at {viewer_addr}");
                            stream.set_nodelay(true).ok();
                            stream.set_write_timeout(Some(Duration::from_millis(500))).ok();

                            // Spawn reader thread for incoming commands from viewer
                            if let Ok(reader_stream) = stream.try_clone() {
                                let flags = Arc::clone(&stream_flags);
                                let cb = Arc::clone(&control_callback);
                                let conn_ref = Arc::clone(&connection);
                                thread::spawn(move || {
                                    read_viewer_commands(reader_stream, flags, cb, conn_ref);
                                });
                            }

                            *connection.lock().unwrap() = Some(stream);
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
            connection,
            stream_flags,
            _connector_thread: connector_thread,
            control_callback,
        }
    }

    pub fn set_control_callback(&self, cb: StreamControlCallback) {
        *self.control_callback.lock().unwrap() = Some(cb);
    }

    #[allow(dead_code)]
    pub fn stream_flags(&self) -> Arc<Mutex<XrStreamFlags>> {
        Arc::clone(&self.stream_flags)
    }

    pub fn send_depth_frame(&self, header: &DepthFrameHeader, timing: &DepthTiming, data: &[u8]) {
        if !self.stream_flags.lock().unwrap().depth_enabled {
            return;
        }
        let payload = encode_depth_frame(header, timing, data);
        self.broadcast(MSG_DEPTH_FRAME_V2, &payload);
    }

    pub fn send_camera_frame(&self, header: &CameraFrameHeader, data: &[u8]) {
        if !self.stream_flags.lock().unwrap().camera_enabled {
            return;
        }
        let payload = encode_camera_frame(header, data);
        self.broadcast(MSG_CAMERA_FRAME, &payload);
    }

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
}

fn read_viewer_commands(
    mut stream: TcpStream,
    flags: Arc<Mutex<XrStreamFlags>>,
    control_callback: Arc<Mutex<Option<StreamControlCallback>>>,
    connection: Arc<Mutex<Option<TcpStream>>>,
) {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();

    loop {
        let mut header = [0u8; 8];
        match stream.read_exact(&mut header) {
            Ok(()) => {}
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock
                || e.kind() == std::io::ErrorKind::TimedOut => {
                // Check if the write side is still connected
                let conn = connection.lock().unwrap();
                if conn.is_none() {
                    return;
                }
                continue;
            }
            Err(_) => {
                info!("XR Data Relay: viewer reader disconnected");
                *connection.lock().unwrap() = None;
                return;
            }
        }

        let msg_type = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let payload_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            if stream.read_exact(&mut payload).is_err() {
                *connection.lock().unwrap() = None;
                return;
            }
        }

        if msg_type == MSG_STREAM_CONTROL && payload_len >= 2 {
            let stream_id = payload[0];
            let enabled = payload[1] != 0;

            let mut f = flags.lock().unwrap();
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

            if let Some(cb) = control_callback.lock().unwrap().as_ref() {
                cb(snapshot.depth_enabled, snapshot.camera_enabled);
            }
        }
    }
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
/// | 12     | f32 x 7   | view_pose[0] qx qy qz qw px py pz (client stage space)       |
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
    use alvr_packets::DepthFrameFormat;

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
