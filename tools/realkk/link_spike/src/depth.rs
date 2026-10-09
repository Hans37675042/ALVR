//! XR_META_environment_depth probe: create the provider, read depth frames back through D3D11
//! and record them as MSG_DEPTH_FRAME_V2 messages in a .rktap file.

use crate::{
    d3d::{D3d, texture_desc_json},
    tap::{DepthFrame, FORMAT_RAW_D16, MSG_DEPTH_FRAME_V2, TapWriter, encode_depth_frame_v2,
          fov_to_pinhole_intrinsics},
    xrctx::{unix_now_ns, xr_result_json},
};
use openxr::{self as xr, sys, sys::Handle as _};
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, ffi::c_void, ptr};

/// XR_ENVIRONMENT_DEPTH_NOT_AVAILABLE_META (a success code).
const ENVIRONMENT_DEPTH_NOT_AVAILABLE_META: i32 = 1000291000;

pub struct DepthProbe {
    fns: xr::raw::EnvironmentDepthMETA,
    provider: sys::EnvironmentDepthProviderMETA,
    swapchain: sys::EnvironmentDepthSwapchainMETA,
    pub width: u32,
    pub height: u32,
    images: Vec<*mut c_void>,
    started: bool,
    writer: Option<TapWriter>,

    acquire_calls: u64,
    acquire_ok: u64,
    not_available: u64,
    acquire_errors: BTreeMap<String, u64>,
    readback_errors: BTreeMap<String, u64>,
    frames_written: u64,
    index_changes: u64,
    last_index: Option<u32>,
    last_key: Option<u64>,
    last_pixel_hash: Option<u64>,
    pixel_changes: u64,
    frame_times_ns: Vec<u64>,
    near_far: Vec<(f32, f32)>,
    first_frame: Option<Value>,
    last_frame: Option<Value>,
    texture_desc: Value,
}

/// Each creation step is recorded in `steps`; the first failure aborts creation.
pub fn create(
    instance: &xr::Instance,
    session: sys::Session,
    with_d3d_images: bool,
    steps: &mut Map<String, Value>,
) -> Option<DepthProbe> {
    let fns = *instance.exts().meta_environment_depth.as_ref()?;
    let mut check = |name: &str, r: sys::Result| -> bool {
        steps.insert(name.into(), xr_result_json(r));
        r.into_raw() >= 0
    };

    let mut provider = sys::EnvironmentDepthProviderMETA::NULL;
    let info = sys::EnvironmentDepthProviderCreateInfoMETA {
        ty: sys::EnvironmentDepthProviderCreateInfoMETA::TYPE,
        next: ptr::null(),
        create_flags: sys::EnvironmentDepthProviderCreateFlagsMETA::EMPTY,
    };
    if !check("create_provider", unsafe {
        (fns.create_environment_depth_provider)(session, &info, &mut provider)
    }) {
        return None;
    }

    let mut probe = DepthProbe {
        fns,
        provider,
        swapchain: sys::EnvironmentDepthSwapchainMETA::NULL,
        width: 0,
        height: 0,
        images: Vec::new(),
        started: false,
        writer: None,
        acquire_calls: 0,
        acquire_ok: 0,
        not_available: 0,
        acquire_errors: BTreeMap::new(),
        readback_errors: BTreeMap::new(),
        frames_written: 0,
        index_changes: 0,
        last_index: None,
        last_key: None,
        last_pixel_hash: None,
        pixel_changes: 0,
        frame_times_ns: Vec::new(),
        near_far: Vec::new(),
        first_frame: None,
        last_frame: None,
        texture_desc: Value::Null,
    };

    let sc_info = sys::EnvironmentDepthSwapchainCreateInfoMETA {
        ty: sys::EnvironmentDepthSwapchainCreateInfoMETA::TYPE,
        next: ptr::null(),
        create_flags: sys::EnvironmentDepthSwapchainCreateFlagsMETA::EMPTY,
    };
    if !check("create_swapchain", unsafe {
        (fns.create_environment_depth_swapchain)(provider, &sc_info, &mut probe.swapchain)
    }) {
        return None;
    }

    let mut state = sys::EnvironmentDepthSwapchainStateMETA {
        ty: sys::EnvironmentDepthSwapchainStateMETA::TYPE,
        next: ptr::null_mut(),
        width: 0,
        height: 0,
    };
    if !check("get_swapchain_state", unsafe {
        (fns.get_environment_depth_swapchain_state)(probe.swapchain, &mut state)
    }) {
        return None;
    }
    probe.width = state.width;
    probe.height = state.height;
    steps.insert("swapchain_size".into(), json!([state.width, state.height]));

    if with_d3d_images {
        let r = probe.enumerate_images();
        steps.insert("enumerate_images".into(), xr_result_json(r));
        steps.insert("image_count".into(), json!(probe.images.len()));
        if let Some(&tex) = probe.images.first() {
            probe.texture_desc = texture_desc_json(tex);
            steps.insert("texture_desc".into(), probe.texture_desc.clone());
        }
    }
    Some(probe)
}

impl DepthProbe {
    fn enumerate_images(&mut self) -> sys::Result {
        unsafe {
            let mut count = 0u32;
            let r = (self.fns.enumerate_environment_depth_swapchain_images)(
                self.swapchain,
                0,
                &mut count,
                ptr::null_mut(),
            );
            if r.into_raw() < 0 {
                return r;
            }
            let mut images: Vec<sys::SwapchainImageD3D11KHR> = (0..count)
                .map(|_| sys::SwapchainImageD3D11KHR {
                    ty: sys::SwapchainImageD3D11KHR::TYPE,
                    next: ptr::null_mut(),
                    texture: ptr::null_mut(),
                })
                .collect();
            let r = (self.fns.enumerate_environment_depth_swapchain_images)(
                self.swapchain,
                count,
                &mut count,
                images.as_mut_ptr() as *mut sys::SwapchainImageBaseHeader,
            );
            self.images = images.iter().map(|i| i.texture).collect();
            r
        }
    }

    pub fn start(&mut self, steps: &mut Map<String, Value>, with_d3d_images: bool) -> bool {
        let r = unsafe { (self.fns.start_environment_depth_provider)(self.provider) };
        steps.insert("start_provider".into(), xr_result_json(r));
        self.started = r.into_raw() >= 0;
        if self.started && with_d3d_images {
            // The runtime may only allocate the textures once the provider runs.
            let r = self.enumerate_images();
            steps.insert("enumerate_images_after_start".into(), xr_result_json(r));
            if let Some(&tex) = self.images.first() {
                self.texture_desc = texture_desc_json(tex);
            }
        }
        self.started
    }

    pub fn set_writer(&mut self, writer: TapWriter) {
        self.writer = Some(writer);
    }

    /// Call between xrBeginFrame and xrEndFrame.
    pub fn poll(&mut self, space: sys::Space, time: sys::Time, d3d: Option<&mut D3d>, offset: Option<i64>) {
        let view = sys::EnvironmentDepthImageViewMETA {
            ty: sys::EnvironmentDepthImageViewMETA::TYPE,
            next: ptr::null(),
            fov: sys::Fovf { angle_left: 0.0, angle_right: 0.0, angle_up: 0.0, angle_down: 0.0 },
            pose: xr::Posef::IDENTITY,
        };
        let mut image = sys::EnvironmentDepthImageMETA {
            ty: sys::EnvironmentDepthImageMETA::TYPE,
            next: ptr::null(),
            swapchain_index: 0,
            near_z: 0.0,
            far_z: 0.0,
            views: [view; 2],
        };
        let info = sys::EnvironmentDepthImageAcquireInfoMETA {
            ty: sys::EnvironmentDepthImageAcquireInfoMETA::TYPE,
            next: ptr::null(),
            space,
            display_time: time,
        };
        self.acquire_calls += 1;
        let r = unsafe { (self.fns.acquire_environment_depth_image)(self.provider, &info, &mut image) };
        if r.into_raw() == ENVIRONMENT_DEPTH_NOT_AVAILABLE_META {
            self.not_available += 1;
            return;
        }
        if r.into_raw() < 0 {
            *self.acquire_errors.entry(format!("{r:?} ({})", r.into_raw())).or_default() += 1;
            return;
        }
        self.acquire_ok += 1;
        let recv_ns = unix_now_ns();

        if self.last_index != Some(image.swapchain_index) {
            self.index_changes += 1;
        }
        self.last_index = Some(image.swapchain_index);

        // A frame is new when the runtime hands out another image (index) or capture pose; pixel
        // hashes are not used for this because a half-written or recycled image would inflate fps.
        let key = fnv1a(
            &[image.swapchain_index.to_le_bytes().as_slice(), &pose_bytes(&image.views[0].pose)].concat(),
        );
        if self.last_key == Some(key) {
            return;
        }
        self.last_key = Some(key);

        let pixels = match (d3d, self.images.get(image.swapchain_index as usize)) {
            (Some(d3d), Some(&tex)) => match d3d.read_depth_slices(tex, self.width, self.height) {
                Ok(p) => Some(p),
                Err(e) => {
                    *self.readback_errors.entry(e).or_default() += 1;
                    None
                }
            },
            _ => None,
        };
        if let Some(p) = &pixels {
            let h = fnv1a(p);
            if self.last_pixel_hash != Some(h) {
                self.pixel_changes += 1;
            }
            self.last_pixel_hash = Some(h);
        }
        self.frame_times_ns.push(recv_ns);
        self.near_far.push((image.near_z, image.far_z));

        let frame = DepthFrame {
            client_timestamp_ns: time.as_nanos().max(0) as u64,
            view_poses: [0, 1].map(|i| pose7(&image.views[i].pose)),
            width: self.width,
            height: self.height * 2,
            near_z: image.near_z,
            far_z: image.far_z,
            format: FORMAT_RAW_D16,
            fov_angles: [0, 1].map(|i| {
                let f = image.views[i].fov;
                [f.angle_left, f.angle_right, f.angle_up, f.angle_down]
            }),
            server_timestamp_unix_ns: match offset {
                Some(o) => (time.as_nanos() + o).max(0) as u64,
                None => recv_ns,
            },
            clock_offset_ns: offset.unwrap_or(0),
            server_receive_unix_ns: recv_ns,
        };
        let summary = frame_json(&frame, image.swapchain_index, pixels.as_deref());
        if self.first_frame.is_none() {
            self.first_frame = Some(summary.clone());
        }
        self.last_frame = Some(summary);

        if let (Some(w), Some(p)) = (self.writer.as_mut(), pixels.as_ref()) {
            match w.write(recv_ns, MSG_DEPTH_FRAME_V2, &encode_depth_frame_v2(&frame, p)) {
                Ok(()) => self.frames_written += 1,
                Err(e) => *self.readback_errors.entry(format!("tap write: {e}")).or_default() += 1,
            }
        }
    }

    /// `window_s` is the capture window; fps counts new frames over the whole window so that a
    /// stream that stalls after a burst does not pass. `max_gap_ms` must also stay below
    /// `max_gap_limit_ms`.
    pub fn summary(&mut self, window_s: f64, min_fps: f64, max_gap_limit_ms: f64) -> Value {
        if let Some(w) = self.writer.as_mut() {
            let _ = w.flush();
        }
        let n = self.frame_times_ns.len();
        let span_s = if n > 1 {
            (self.frame_times_ns[n - 1] - self.frame_times_ns[0]) as f64 / 1e9
        } else {
            0.0
        };
        let span_fps = if span_s > 0.0 { (n - 1) as f64 / span_s } else { 0.0 };
        let fps = if window_s > 0.0 { n as f64 / window_s } else { 0.0 };
        let intervals_ms: Vec<f64> =
            self.frame_times_ns.windows(2).map(|w| (w[1] - w[0]) as f64 / 1e6).collect();
        let max_gap_ms = intervals_ms.iter().cloned().fold(0.0, f64::max);
        let (near_min, near_max, far_values) = self.near_far.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, Vec::<String>::new()),
            |(lo, hi, mut fars), &(n, f)| {
                let fs = format!("{f}");
                if !fars.contains(&fs) {
                    fars.push(fs);
                }
                (lo.min(n), hi.max(n), fars)
            },
        );
        json!({
            "resolution_per_view": [self.width, self.height],
            "stacked_resolution": [self.width, self.height * 2],
            "swapchain_images": self.images.len(),
            "texture_desc": self.texture_desc,
            "acquire_calls": self.acquire_calls,
            "acquire_ok": self.acquire_ok,
            "acquire_not_available": self.not_available,
            "acquire_errors": self.acquire_errors,
            "swapchain_index_changes": self.index_changes,
            "new_frames": n,
            "frames_written": self.frames_written,
            "pixels_ok": self.frames_written > 0 && self.readback_errors.is_empty(),
            "pixel_changes": self.pixel_changes,
            "fps": fps,
            "window_s": window_s,
            "span_fps": span_fps,
            "span_s": span_s,
            "max_gap_ms": max_gap_ms,
            "near_z_range": if n > 0 { json!([near_min, near_max]) } else { Value::Null },
            "far_z_values": far_values,
            "readback_errors": self.readback_errors,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "max_gap_limit_ms": max_gap_limit_ms,
            "fps_ok": fps >= min_fps && n > 1 && max_gap_ms < max_gap_limit_ms,
        })
    }
}

impl Drop for DepthProbe {
    fn drop(&mut self) {
        unsafe {
            if self.started {
                (self.fns.stop_environment_depth_provider)(self.provider);
            }
            if self.swapchain != sys::EnvironmentDepthSwapchainMETA::NULL {
                (self.fns.destroy_environment_depth_swapchain)(self.swapchain);
            }
            (self.fns.destroy_environment_depth_provider)(self.provider);
        }
    }
}

fn pose7(p: &sys::Posef) -> [f32; 7] {
    let (o, t) = (p.orientation, p.position);
    [o.x, o.y, o.z, o.w, t.x, t.y, t.z]
}

fn pose_bytes(p: &sys::Posef) -> Vec<u8> {
    pose7(p).iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn fnv1a(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn frame_json(f: &DepthFrame, swapchain_index: u32, pixels: Option<&[u8]>) -> Value {
    let stats = pixels.map(|p| {
        let vals: Vec<u16> = p.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let min = vals.iter().copied().min().unwrap_or(0);
        let max = vals.iter().copied().max().unwrap_or(0);
        let zeros = vals.iter().filter(|&&v| v == 0).count();
        let ones = vals.iter().filter(|&&v| v == u16::MAX).count();
        json!({"d16_min": min, "d16_max": max, "zero_px": zeros, "max_px": ones, "px": vals.len()})
    });
    let intrinsics = [0, 1].map(|i| fov_to_pinhole_intrinsics(f.fov_angles[i], f.width, f.height / 2));
    json!({
        "client_timestamp_ns": f.client_timestamp_ns,
        "swapchain_index": swapchain_index,
        "near_z": f.near_z,
        "far_z": format!("{}", f.far_z),
        "view_poses_qxyzw_pxyz": f.view_poses,
        "fov_lrud_deg": f.fov_angles.map(|a| a.map(f32::to_degrees)),
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "clock_offset_ns": f.clock_offset_ns,
        "pixel_stats": stats,
    })
}
