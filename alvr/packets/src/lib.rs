use alvr_common::{
    BodySkeleton, ConnectionState, DeviceMotion, LogSeverity, Pose, ViewParams,
    anyhow::Result,
    glam::{Quat, UVec2, Vec2},
    semver::Version,
};
use alvr_session::{
    ClientsidePostProcessingConfig, CodecType, PassthroughMode, PerformanceLevel, SessionConfig,
    Settings, XrDataConfig,
};
use serde::{Deserialize, Serialize};
use serde_json as json;
use std::{
    collections::HashSet,
    fmt::{self, Debug},
    net::IpAddr,
    time::Duration,
};

pub const TRACKING: u16 = 0;
pub const HAPTICS: u16 = 1;
pub const AUDIO: u16 = 2;
pub const VIDEO: u16 = 3;
pub const STATISTICS: u16 = 4;
pub const DEPTH: u16 = 5;
pub const CAMERA: u16 = 6;

#[derive(Serialize, Deserialize, Clone)]
pub struct VideoStreamingCapabilitiesExt {
    // Nothing for now
}

#[derive(Serialize, Deserialize, Clone)]
pub struct VideoStreamingCapabilities {
    pub default_view_resolution: UVec2,
    pub max_view_resolution: UVec2,
    pub refresh_rates: Vec<f32>,
    pub microphone_sample_rate: u32,
    pub foveated_encoding: bool,
    pub encoder_high_profile: bool,
    pub encoder_10_bits: bool,
    pub encoder_av1: bool,
    pub prefer_10bit: bool,
    pub preferred_encoding_gamma: f32,
    pub prefer_hdr: bool,
    pub ext_str: String,
}

impl VideoStreamingCapabilities {
    pub fn with_ext(self, ext: VideoStreamingCapabilitiesExt) -> Self {
        Self {
            ext_str: json::to_string(&ext).unwrap(),
            ..self
        }
    }

    pub fn ext(&self) -> Result<VideoStreamingCapabilitiesExt> {
        let _ext_json = json::from_str::<json::Value>(&self.ext_str)?;

        // decode values here

        Ok(VideoStreamingCapabilitiesExt {})
    }
}

#[derive(Serialize, Deserialize)]
pub struct ConnectionAcceptedInfo {
    pub client_protocol_id: u64,
    pub platform_string: String,
    pub server_ip: IpAddr,
    pub streaming_capabilities: Option<VideoStreamingCapabilities>,
}

#[derive(Serialize, Deserialize)]
pub enum ClientConnectionResult {
    ConnectionAccepted(Box<ConnectionAcceptedInfo>),
    ClientStandby,
}

#[derive(Serialize, Deserialize)]
pub struct NegotiatedStreamingConfigExt {
    // Nothing for now
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NegotiatedStreamingConfig {
    pub view_resolution: UVec2,
    pub refresh_rate_hint: f32,
    pub game_audio_sample_rate: u32,
    pub enable_foveated_encoding: bool,
    pub encoding_gamma: f32,
    pub enable_hdr: bool,
    pub wired: bool,
    pub ext_str: String,
}

impl NegotiatedStreamingConfig {
    pub fn with_ext(self, ext: NegotiatedStreamingConfigExt) -> Self {
        Self {
            ext_str: json::to_string(&ext).unwrap(),
            ..self
        }
    }

    pub fn ext(&self) -> Result<NegotiatedStreamingConfigExt> {
        let _ext_json = json::from_str::<json::Value>(&self.ext_str)?;

        // decode values here

        Ok(NegotiatedStreamingConfigExt {})
    }
}

#[derive(Serialize, Deserialize)]
pub struct StreamConfigPacket {
    pub session: String, // JSON session that allows for extrapolation
    pub negotiated: NegotiatedStreamingConfig,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StreamConfig {
    pub server_version: Version,
    pub settings: Settings,
    pub negotiated_config: NegotiatedStreamingConfig,
}

impl StreamConfigPacket {
    pub fn new(session: &SessionConfig, negotiated: NegotiatedStreamingConfig) -> Result<Self> {
        Ok(Self {
            session: json::to_string(session)?,
            negotiated,
        })
    }

    pub fn to_stream_config(self) -> Result<StreamConfig> {
        let mut session_config = SessionConfig::default();
        session_config.merge_from_json(&json::from_str(&self.session)?)?;
        let settings = session_config.to_settings();

        Ok(StreamConfig {
            server_version: session_config.server_version,
            settings,
            negotiated_config: self.negotiated,
        })
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DecoderInitializationConfig {
    pub codec: CodecType,
    pub config_buffer: Vec<u8>, // e.g. SPS + PPS NALs
    pub ext_str: String,
}

#[derive(Serialize, Deserialize)]
pub enum ServerControlPacket {
    StartStream,
    DecoderConfig(DecoderInitializationConfig),
    Restarting,
    KeepAlive,
    RealTimeConfig(RealTimeConfig),
    XrStreamControl {
        depth_enabled: bool,
        camera_enabled: bool,
    },
    Reserved(String),
    ReservedBuffer(Vec<u8>),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BatteryInfo {
    pub device_id: u64,
    pub gauge_value: f32, // range [0, 1]
    pub is_plugged: bool,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ButtonValue {
    Binary(bool),
    Scalar(f32),
}

#[derive(Serialize, Deserialize)]
pub struct ButtonEntry {
    pub path_id: u64,
    pub value: ButtonValue,
}

#[derive(Serialize, Deserialize)]
pub enum ClientControlPacket {
    PlayspaceSync(Option<Vec2>),
    RequestIdr,
    KeepAlive,
    StreamReady, // This flag notifies the server the client streaming socket is ready listening
    LocalViewParams([ViewParams; 2]), // In relation to head
    Battery(BatteryInfo),
    Buttons(Vec<ButtonEntry>),
    ActiveInteractionProfile {
        device_id: u64,
        profile_id: u64,
        input_ids: HashSet<u64>,
    },
    Log {
        level: LogSeverity,
        message: String,
    },
    ProximityState(bool),
    Reserved(String),
    ReservedBuffer(Vec<u8>),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum FaceExpressions {
    Fb(Vec<f32>),   // 70 values
    Pico(Vec<f32>), // 52 values
    Htc {
        eye: Option<Vec<f32>>, // 14 values
        lip: Option<Vec<f32>>, // 37 values
    },
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct FaceData {
    // Can be used for foveated eye tracking
    pub eyes_combined: Option<Quat>,
    // Should be used only for social presence
    pub eyes_social: [Option<Quat>; 2],

    pub face_expressions: Option<FaceExpressions>,
}

#[derive(Serialize, Deserialize)]
pub struct TrackingData {
    pub poll_timestamp: Duration,
    pub device_motions: Vec<(u64, DeviceMotion)>,
    pub hand_skeletons: [Option<[Pose; 26]>; 2],
    pub face: FaceData,
    pub body: Option<BodySkeleton>,
}

#[derive(Serialize, Deserialize)]
pub struct VideoPacketHeader {
    pub timestamp: Duration,
    pub global_view_params: [ViewParams; 2],
    pub is_idr: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Haptics {
    pub device_id: u64,
    pub duration: Duration,
    pub frequency: f32,
    pub amplitude: f32,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum DepthFrameFormat {
    RawD16,
    H264,
    Lz4D16,
}

/// Header of one environment depth frame (XR_META_environment_depth).
///
/// The payload contains both depth views stacked vertically: view 0 (left) in the top half,
/// view 1 (right) in the bottom half, each `width` x `height / 2`, rows ordered top-down.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DepthFrameHeader {
    /// Client XR time (predicted display time) the depth image was acquired for. This is in the
    /// client's clock domain; the server maps it to its own clock before relaying.
    pub timestamp: Duration,
    /// Client XR time ("now") right before the frame was handed to the network layer. The server
    /// uses it to estimate the client-to-server clock offset.
    pub client_send_time: Duration,
    /// Pose of each depth view (0 = left, 1 = right), as returned by
    /// xrAcquireEnvironmentDepthImageMETA in the client's stage reference space.
    /// The depth cameras are not the eye cameras, so each view must be unprojected with its own
    /// pose and FOV.
    pub view_poses: [Pose; 2],
    pub width: u32,
    /// Total height of the stacked image (two views).
    pub height: u32,
    pub near_z: f32,
    pub far_z: f32,
    pub format: DepthFrameFormat,
    /// Per-view OpenXR `XrFovf` half-angles in radians: [left, right, up, down]
    /// (left and down are usually negative). These are angles, not pinhole intrinsics; use
    /// [`fov_to_pinhole_intrinsics`] to convert.
    pub fov_angles: [[f32; 4]; 2],
}

impl DepthFrameHeader {
    /// Pinhole intrinsics (fx, fy, cx, cy) in pixels for one view of this frame.
    pub fn view_intrinsics(&self, view: usize) -> [f32; 4] {
        fov_to_pinhole_intrinsics(self.fov_angles[view], self.width, self.height / 2)
    }
}

/// Convert OpenXR FOV half-angles [left, right, up, down] (radians) into pinhole intrinsics
/// [fx, fy, cx, cy] (pixels) for an image of `width` x `height` whose rows are ordered top-down
/// (pixel y grows downwards) and whose pixel centers sit at half-integer coordinates.
///
/// A view-space point (x, y, z) with z < 0 in front of the camera (OpenXR convention) maps to
/// u = fx * (x / -z) + cx, v = fy * (-y / -z) + cy.
pub fn fov_to_pinhole_intrinsics(fov: [f32; 4], width: u32, height: u32) -> [f32; 4] {
    let [left, right, up, down] = fov;
    let (tan_l, tan_r, tan_u, tan_d) = (left.tan(), right.tan(), up.tan(), down.tan());
    let (w, h) = (width as f32, height as f32);

    let fx = w / (tan_r - tan_l);
    let fy = h / (tan_u - tan_d);
    let cx = -tan_l * fx;
    let cy = tan_u * fy;

    [fx, fy, cx, cy]
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CameraFrameHeader {
    pub timestamp: Duration,
    pub head_pose: Pose,
    pub width: u32,
    pub height: u32,
    pub format: CameraFrameFormat,
    // Camera intrinsics (fx, fy, cx, cy)
    pub intrinsics: [f32; 4],
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum CameraFrameFormat {
    Jpeg,
    Nv12,
    Rgb8,
    H264,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum PathSegment {
    Name(String),
    Index(usize),
}

impl Debug for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathSegment::Name(name) => write!(f, "{name}"),
            PathSegment::Index(index) => write!(f, "[{index}]"),
        }
    }
}

impl From<&str> for PathSegment {
    fn from(value: &str) -> Self {
        PathSegment::Name(value.to_owned())
    }
}

impl From<String> for PathSegment {
    fn from(value: String) -> Self {
        PathSegment::Name(value)
    }
}

impl From<usize> for PathSegment {
    fn from(value: usize) -> Self {
        PathSegment::Index(value)
    }
}

// todo: support indices
pub fn parse_path(path: &str) -> Vec<PathSegment> {
    path.split('.').map(|s| s.into()).collect()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ClientConnectionsAction {
    AddIfMissing {
        trusted: bool,
        manual_ips: Vec<IpAddr>,
    },
    SetDisplayName(String),
    Trust,
    SetManualIps(Vec<IpAddr>),
    RemoveEntry,
    UpdateCurrentIp(Option<IpAddr>),
    SetConnectionState(ConnectionState),
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct ClientStatistics {
    pub target_timestamp: Duration, // identifies the frame
    pub frame_interval: Duration,
    pub video_decode: Duration,
    pub video_decoder_queue: Duration,
    pub rendering: Duration,
    pub vsync_queue: Duration,
    pub total_pipeline_latency: Duration,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PathValuePair {
    pub path: Vec<PathSegment>,
    pub value: json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum FirewallRulesAction {
    Add,
    Remove,
}

// Note: server sends a packet to the client at low frequency, binary encoding, without ensuring
// compatibility between different versions, even if within the same major version.
#[derive(Serialize, Deserialize, PartialEq, Clone)]
pub struct RealTimeConfig {
    pub passthrough: Option<PassthroughMode>,
    pub clientside_post_processing: Option<ClientsidePostProcessingConfig>,
    pub cpu_performance_level: Option<PerformanceLevel>,
    pub gpu_performance_level: Option<PerformanceLevel>,
    pub xr_data_config: Option<XrDataConfig>,
    pub ext_str: String,
}

impl RealTimeConfig {
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            passthrough: settings.video.passthrough.clone().into_option(),
            clientside_post_processing: settings
                .video
                .clientside_post_processing
                .clone()
                .into_option(),
            cpu_performance_level: settings.headset.performance_level.clone().cpu.into_option(),
            gpu_performance_level: settings.headset.performance_level.clone().gpu.into_option(),
            xr_data_config: settings.video.xr_data.clone().into_option(),
            ext_str: String::new(), // No extensions for now
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn symmetric_fov_gives_centered_principal_point() {
        let a = std::f32::consts::FRAC_PI_4;
        let [fx, fy, cx, cy] = fov_to_pinhole_intrinsics([-a, a, a, -a], 320, 240);
        assert!(approx(fx, 160.0) && approx(fy, 120.0));
        assert!(approx(cx, 160.0) && approx(cy, 120.0));
    }

    #[test]
    fn fov_edges_project_to_image_borders() {
        // Asymmetric FOV like a real depth view
        let fov = [-0.9_f32, 0.6, 0.7, -0.8];
        let (w, h) = (320, 320);
        let [fx, fy, cx, cy] = fov_to_pinhole_intrinsics(fov, w, h);

        // Ray along the left/up edges (z = -1): x = tan(left), y = tan(up)
        let u_left = fx * fov[0].tan() + cx;
        let u_right = fx * fov[1].tan() + cx;
        let v_top = fy * -fov[2].tan() + cy;
        let v_bottom = fy * -fov[3].tan() + cy;

        assert!(approx(u_left, 0.0), "{u_left}");
        assert!(approx(u_right, w as f32), "{u_right}");
        assert!(approx(v_top, 0.0), "{v_top}");
        assert!(approx(v_bottom, h as f32), "{v_bottom}");
    }

    #[test]
    fn view_intrinsics_use_per_view_height() {
        let a = std::f32::consts::FRAC_PI_4;
        let header = DepthFrameHeader {
            timestamp: Duration::ZERO,
            client_send_time: Duration::ZERO,
            view_poses: [Pose::IDENTITY; 2],
            width: 320,
            height: 640, // two stacked 320x320 views
            near_z: 0.1,
            far_z: f32::INFINITY,
            format: DepthFrameFormat::Lz4D16,
            fov_angles: [[-a, a, a, -a], [-a, 0.5, a, -a]],
        };
        let [_, fy, _, cy] = header.view_intrinsics(0);
        assert!(approx(fy, 160.0) && approx(cy, 160.0));
        let [fx1, _, cx1, _] = header.view_intrinsics(1);
        assert!(approx(fx1, 320.0 / (0.5_f32.tan() + 1.0)));
        assert!(approx(cx1, fx1));
    }
}
