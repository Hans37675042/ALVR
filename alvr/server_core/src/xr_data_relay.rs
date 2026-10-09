use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use alvr_common::{info, warn};
use alvr_packets::{CameraFrameHeader, DepthFrameHeader};

pub const DEFAULT_VIEWER_PORT: u16 = 9944;
const RETRY_INTERVAL: Duration = Duration::from_secs(5);
const MSG_DEPTH_FRAME: u32 = 1;
const MSG_CAMERA_FRAME: u32 = 2;
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

    pub fn send_depth_frame(&self, header: &DepthFrameHeader, data: &[u8]) {
        if !self.stream_flags.lock().unwrap().depth_enabled {
            return;
        }
        let payload = encode_depth_frame(header, data);
        self.broadcast(MSG_DEPTH_FRAME, &payload);
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

fn encode_depth_frame(header: &DepthFrameHeader, data: &[u8]) -> Vec<u8> {
    // Header layout (matches test viewer parser):
    // timestamp(8) + pose(7*4=28) + width(4) + height(4) + near_z(4) + far_z(4) + format(4) + intrinsics(8*4=32) = 88 bytes
    // Then raw pixel data
    let mut buf = Vec::with_capacity(88 + data.len());

    // timestamp as nanos u64
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
    buf.extend_from_slice(&header.near_z.to_le_bytes());
    buf.extend_from_slice(&header.far_z.to_le_bytes());

    // depth format: 0=RawD16, 1=H264, 2=Lz4D16
    let format_id: u32 = match &header.format {
        alvr_packets::DepthFrameFormat::RawD16 => 0,
        alvr_packets::DepthFrameFormat::H264 => 1,
        alvr_packets::DepthFrameFormat::Lz4D16 => 2,
    };
    buf.extend_from_slice(&format_id.to_le_bytes());

    // intrinsics: [eye0: left,right,up,down] [eye1: left,right,up,down]
    for eye in &header.intrinsics {
        for val in eye {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }

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

