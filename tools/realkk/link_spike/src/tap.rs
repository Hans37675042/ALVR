//! .rktap writer and MSG_DEPTH_FRAME_V2 encoder, byte compatible with tools/realkk/rktap.py and
//! depth_listener.parse_depth_v2 (layout documented in alvr/server_core/src/xr_data_relay.rs).

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

pub const MSG_DEPTH_FRAME_V2: u32 = 3;
pub const MSG_MARKER: u32 = 0xFFFF_0001;
pub const DEPTH_FRAME_V2_HEADER_SIZE: u32 = 176;
pub const FORMAT_RAW_D16: u32 = 0;

const MAGIC: &[u8] = b"RKTAP";
const VERSION: u32 = 1;

/// One stacked two-view depth frame (view 0 top half, view 1 bottom half).
#[derive(Clone, Debug)]
pub struct DepthFrame {
    pub client_timestamp_ns: u64,
    /// qx qy qz qw px py pz per view
    pub view_poses: [[f32; 7]; 2],
    pub width: u32,
    /// Stacked height (2 x per-view height).
    pub height: u32,
    pub near_z: f32,
    pub far_z: f32,
    pub format: u32,
    /// left right up down (radians) per view
    pub fov_angles: [[f32; 4]; 2],
    pub server_timestamp_unix_ns: u64,
    pub clock_offset_ns: i64,
    pub server_receive_unix_ns: u64,
}

/// Same formula as alvr_packets::fov_to_pinhole_intrinsics: [fx, fy, cx, cy] in pixels for a
/// top-down image.
pub fn fov_to_pinhole_intrinsics(fov: [f32; 4], width: u32, height: u32) -> [f32; 4] {
    let [left, right, up, down] = fov;
    let (tan_l, tan_r, tan_u, tan_d) = (left.tan(), right.tan(), up.tan(), down.tan());
    let (w, h) = (width as f32, height as f32);

    let fx = w / (tan_r - tan_l);
    let fy = h / (tan_u - tan_d);
    [fx, fy, -tan_l * fx, tan_u * fy]
}

pub fn encode_depth_frame_v2(frame: &DepthFrame, data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(DEPTH_FRAME_V2_HEADER_SIZE as usize + data.len());
    buf.extend_from_slice(&DEPTH_FRAME_V2_HEADER_SIZE.to_le_bytes());
    buf.extend_from_slice(&frame.client_timestamp_ns.to_le_bytes());
    for pose in &frame.view_poses {
        for v in pose {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf.extend_from_slice(&frame.width.to_le_bytes());
    buf.extend_from_slice(&frame.height.to_le_bytes());
    buf.extend_from_slice(&frame.near_z.to_le_bytes());
    buf.extend_from_slice(&frame.far_z.to_le_bytes());
    buf.extend_from_slice(&frame.format.to_le_bytes());
    for fov in &frame.fov_angles {
        for v in fov {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    for fov in &frame.fov_angles {
        for v in fov_to_pinhole_intrinsics(*fov, frame.width, frame.height / 2) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf.extend_from_slice(&frame.server_timestamp_unix_ns.to_le_bytes());
    buf.extend_from_slice(&frame.clock_offset_ns.to_le_bytes());
    buf.extend_from_slice(&frame.server_receive_unix_ns.to_le_bytes());
    debug_assert_eq!(buf.len(), DEPTH_FRAME_V2_HEADER_SIZE as usize);
    buf.extend_from_slice(data);
    buf
}

pub struct TapWriter {
    out: BufWriter<File>,
}

impl TapWriter {
    pub fn create(path: &Path, metadata: &serde_json::Value) -> io::Result<Self> {
        let mut out = BufWriter::new(File::create(path)?);
        let meta = serde_json::to_vec(metadata)?;
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(meta.len() as u32).to_le_bytes())?;
        out.write_all(&meta)?;
        Ok(Self { out })
    }

    pub fn write(&mut self, recv_ns: u64, msg_type: u32, payload: &[u8]) -> io::Result<()> {
        self.out.write_all(&recv_ns.to_le_bytes())?;
        self.out.write_all(&msg_type.to_le_bytes())?;
        self.out.write_all(&(payload.len() as u32).to_le_bytes())?;
        self.out.write_all(payload)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

impl Drop for TapWriter {
    fn drop(&mut self) {
        let _ = self.out.flush();
    }
}
