use crate::{
    graphics::{self, ProjectionLayerAlphaConfig, ProjectionLayerBuilder},
    interaction::{self, InteractionContext, InteractionSourcesConfig},
};
use alvr_client_core::{
    ClientCoreContext,
    depth_pipeline::{
        CapturePacer, SendLink, SlotEvent, SlotRing, StageSet, SubmitOutcome, copy_flipped_rows,
        detailed_gl_check_due, send_link,
    },
    video_decoder::{self, VideoDecoderConfig, VideoDecoderSource},
};
use alvr_common::{
    DETACHED_CONTROLLER_LEFT_ID, DETACHED_CONTROLLER_RIGHT_ID, HAND_LEFT_ID, HAND_RIGHT_ID,
    HEAD_ID, Pose, RelaxedAtomic, ViewParams,
    anyhow::Result,
    error,
    glam::{UVec2, Vec2},
    parking_lot::RwLock,
};
use alvr_graphics::{GraphicsContext, StreamRenderer, StreamViewParams};
use glow::HasContext;
use alvr_packets::{
    DepthFrameHeader, RealTimeConfig, StreamConfig, TrackingData,
};
use alvr_session::{
    ClientsideFoveationConfig, ClientsideFoveationMode, ClientsidePostProcessingConfig, CodecType,
    FoveatedEncodingConfig, MediacodecProperty, PassthroughMode, UpscalingConfig, XrDataConfig,
};
use alvr_system_info::Platform;
use openxr as xr;
use std::{
    ptr,
    rc::Rc,
    sync::Arc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const DECODER_MAX_TIMEOUT_MULTIPLIER: f32 = 0.8;

// Depth stage timings are summarized in the log every this many captures
const DEPTH_PERF_REPORT_INTERVAL: u64 = 50;

fn elapsed_ms(since: Instant) -> f32 {
    since.elapsed().as_secs_f32() * 1000.0
}

pub struct ParsedStreamConfig {
    pub view_resolution: UVec2,
    pub refresh_rate_hint: f32,
    pub encoding_gamma: f32,
    pub enable_hdr: bool,
    pub passthrough: Option<PassthroughMode>,
    pub foveated_encoding_config: Option<FoveatedEncodingConfig>,
    pub clientside_foveation_config: Option<ClientsideFoveationConfig>,
    pub clientside_post_processing: Option<ClientsidePostProcessingConfig>,
    pub upscaling: Option<UpscalingConfig>,
    pub force_software_decoder: bool,
    pub max_buffering_frames: f32,
    pub buffering_history_weight: f32,
    pub decoder_options: Vec<(String, MediacodecProperty)>,
    pub interaction_sources: InteractionSourcesConfig,
    pub xr_data_config: Option<XrDataConfig>,
}

impl ParsedStreamConfig {
    pub fn new(config: &StreamConfig) -> Self {
        Self {
            view_resolution: config.negotiated_config.view_resolution,
            refresh_rate_hint: config.negotiated_config.refresh_rate_hint,
            encoding_gamma: config.negotiated_config.encoding_gamma,
            enable_hdr: config.negotiated_config.enable_hdr,
            passthrough: config.settings.video.passthrough.as_option().cloned(),
            foveated_encoding_config: config
                .negotiated_config
                .enable_foveated_encoding
                .then(|| config.settings.video.foveated_encoding.as_option().cloned())
                .flatten(),
            clientside_foveation_config: config
                .settings
                .video
                .clientside_foveation
                .as_option()
                .cloned(),
            clientside_post_processing: config
                .settings
                .video
                .clientside_post_processing
                .as_option()
                .cloned(),
            upscaling: config.settings.video.upscaling.as_option().cloned(),
            force_software_decoder: config.settings.video.force_software_decoder,
            max_buffering_frames: config.settings.video.max_buffering_frames,
            buffering_history_weight: config.settings.video.buffering_history_weight,
            decoder_options: config.settings.video.mediacodec_extra_options.clone(),
            interaction_sources: InteractionSourcesConfig::new(config),
            xr_data_config: config.settings.video.xr_data.as_option().cloned(),
        }
    }
}

// Readback slots in flight; a capture is skipped (never waited for) when all are pending
const DEPTH_READBACK_SLOTS: usize = 3;
// A slot whose fence has not signaled after this many frames is given up
const DEPTH_READBACK_TIMEOUT_FRAMES: u64 = 30;

/// Everything the depth header needs, taken when the image is acquired so that a frame read
/// back a few frames later still carries the pose and FOV it was captured with.
struct DepthFrameMeta {
    display_time: xr::Time,
    view_poses: [Pose; 2],
    fov_angles: [[f32; 4]; 2],
    near_z: f32,
    far_z: f32,
    // Single view size; views are stacked vertically
    width: u32,
    height: u32,
    enqueued_at: Instant,
}

/// Asynchronous depth readback. Per capture, in front of xrEndFrame and without waiting:
/// 1. Attach the runtime TEXTURE_2D_ARRAY layer as depth on fbo_read
/// 2. BlitFramebuffer depth → local D16 on fbo_write (bypasses the Adreno 740 depth sampling bug)
/// 3. CopyImageSubData D16 → R16UI
/// 4. ReadPixels(RED_INTEGER, UNSIGNED_SHORT) from fbo_r16ui into a pixel pack buffer, then a
///    fence. One to a few frames later, once the fence signaled, the buffer is mapped and copied
///    out with the rows flipped. ReadPixels into client memory would stall the render thread
///    until the GPU caught up.
struct DepthReadback {
    fbo_read: glow::NativeFramebuffer,  // runtime tex layer attached as depth
    fbo_write: glow::NativeFramebuffer, // local D16 attached as depth
    fbo_r16ui: glow::NativeFramebuffer, // R16UI attached as color (for integer readback)
    depth_2d_tex: glow::NativeTexture,  // local D16 blit target
    r16ui_tex: glow::NativeTexture,     // R16UI for integer readback
    pbos: [glow::NativeBuffer; DEPTH_READBACK_SLOTS],
    ring: SlotRing<glow::NativeFence, DepthFrameMeta>,
    tex_width: u32,
    tex_height: u32,
    frame_count: u32,
}

impl DepthReadback {
    unsafe fn new(gl: &glow::Context) -> Self {
        unsafe {
            Self {
                fbo_read: gl.create_framebuffer().unwrap(),
                fbo_write: gl.create_framebuffer().unwrap(),
                fbo_r16ui: gl.create_framebuffer().unwrap(),
                depth_2d_tex: gl.create_texture().unwrap(),
                r16ui_tex: gl.create_texture().unwrap(),
                pbos: [(); DEPTH_READBACK_SLOTS].map(|_| gl.create_buffer().unwrap()),
                ring: SlotRing::new(DEPTH_READBACK_SLOTS),
                tex_width: 0,
                tex_height: 0,
                frame_count: 0,
            }
        }
    }

    /// Bytes between rows in a pack buffer (GL_PACK_ALIGNMENT defaults to 4)
    fn pack_row_stride(width: u32) -> usize {
        (width as usize * 2 + 3) & !3
    }

    /// Bytes of one slot: both views of R16UI
    fn slot_bytes(width: u32, height: u32) -> usize {
        Self::pack_row_stride(width) * height as usize * 2
    }

    /// (Re)creates the blit targets and pack buffers for this image size. Frames still in flight
    /// were read back at the old size and are dropped.
    unsafe fn ensure_size(&mut self, gl: &glow::Context, width: u32, height: u32) {
        if self.tex_width == width && self.tex_height == height {
            return;
        }
        alvr_common::info!("[XR_DATA] Init blit targets: {}x{}", width, height);
        let (w, h) = (width as i32, height as i32);
        unsafe {
            for fence in self.ring.drain() {
                gl.delete_sync(fence);
            }

            // Local D16 depth texture (blit target)
            gl.delete_texture(self.depth_2d_tex);
            self.depth_2d_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(self.depth_2d_tex));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::DEPTH_COMPONENT16, w, h);
            gl.bind_texture(glow::TEXTURE_2D, None);

            // Attach local D16 to fbo_write
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo_write));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::TEXTURE_2D,
                Some(self.depth_2d_tex),
                0,
            );
            let write_status = gl.check_framebuffer_status(glow::FRAMEBUFFER);

            // R16UI texture (CopyImageSubData target for integer readback)
            gl.delete_texture(self.r16ui_tex);
            self.r16ui_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(self.r16ui_tex));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::R16UI, w, h);
            gl.bind_texture(glow::TEXTURE_2D, None);

            // Attach R16UI to fbo_r16ui
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo_r16ui));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(self.r16ui_tex),
                0,
            );
            let r16ui_status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            // Pack buffers, read by the CPU once per fill
            let slot_bytes = Self::slot_bytes(width, height) as i32;
            for &pbo in &self.pbos {
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
                gl.buffer_data_size(glow::PIXEL_PACK_BUFFER, slot_bytes, glow::STREAM_READ);
            }
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);

            self.tex_width = width;
            self.tex_height = height;
            let err = gl.get_error();
            alvr_common::info!(
                "[XR_DATA] Blit targets ready: write_fbo={:#x} r16ui_fbo={:#x} \
                 {DEPTH_READBACK_SLOTS} pack buffers x {slot_bytes} B, GL err={:#x}",
                write_status,
                r16ui_status,
                err,
            );
        }
    }

    /// Queues the copy of both views of `runtime_tex` into the pack buffer of `slot` and returns
    /// the fence that signals when it is filled. Does not wait for the GPU.
    unsafe fn enqueue(
        &mut self,
        gl: &glow::Context,
        runtime_tex: glow::NativeTexture,
        slot: usize,
    ) -> Result<glow::NativeFence, String> {
        let (w, h) = (self.tex_width as i32, self.tex_height as i32);
        let view_bytes = Self::pack_row_stride(self.tex_width) * self.tex_height as usize;
        // glGetError and framebuffer status queries can be costly on some drivers; most captures
        // only check the sticky error flag once at the end
        let detailed = detailed_gl_check_due(self.frame_count as u64);
        let step_error = |gl: &glow::Context| {
            if detailed {
                unsafe { gl.get_error() }
            } else {
                glow::NO_ERROR
            }
        };
        let mut failure = None;
        unsafe {
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(self.pbos[slot]));
            for eye in 0..2u32 {
                // Step 1: Attach runtime TEXTURE_2D_ARRAY layer as depth on fbo_read
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo_read));
                gl.framebuffer_texture_layer(
                    glow::FRAMEBUFFER,
                    glow::DEPTH_ATTACHMENT,
                    Some(runtime_tex),
                    0,
                    eye as i32,
                );
                let read_status = if detailed {
                    gl.check_framebuffer_status(glow::FRAMEBUFFER)
                } else {
                    glow::FRAMEBUFFER_COMPLETE
                };
                let attach_err = step_error(gl);

                // Step 2: Blit depth from runtime FBO → local D16 FBO
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.fbo_read));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.fbo_write));
                gl.blit_framebuffer(0, 0, w, h, 0, 0, w, h, glow::DEPTH_BUFFER_BIT, glow::NEAREST);
                let blit_err = step_error(gl);

                // Step 3: CopyImageSubData from local D16 → R16UI (bitwise, both 16-bit)
                gl.copy_image_sub_data(
                    self.depth_2d_tex,
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    0,
                    self.r16ui_tex,
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    0,
                    w,
                    h,
                    1,
                );
                let copy_err = step_error(gl);

                // Step 4: Integer ReadPixels from R16UI into this view's part of the pack buffer
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo_r16ui));
                gl.read_pixels(
                    0,
                    0,
                    w,
                    h,
                    glow::RED_INTEGER,
                    glow::UNSIGNED_SHORT,
                    glow::PixelPackData::BufferOffset((eye as usize * view_bytes) as u32),
                );
                let read_err = step_error(gl);

                let errors = [attach_err, blit_err, copy_err, read_err];
                if read_status != glow::FRAMEBUFFER_COMPLETE
                    || errors.iter().any(|&e| e != glow::NO_ERROR)
                {
                    failure.get_or_insert(format!(
                        "eye {eye}: fbo status {read_status:#x}, GL errors attach/blit/\
                         copy/read {attach_err:#x}/{blit_err:#x}/{copy_err:#x}/{read_err:#x}"
                    ));
                }
            }
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.frame_count += 1;

            if !detailed {
                let err = gl.get_error();
                if err != glow::NO_ERROR {
                    failure = Some(format!("GL error {err:#x} (step unknown)"));
                }
            }
            if let Some(failure) = failure {
                return Err(failure);
            }
            let fence = gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)?;
            // Make sure the fence reaches the GPU even if nothing else flushes soon
            gl.flush();
            Ok(fence)
        }
    }

    unsafe fn is_signaled(gl: &glow::Context, fence: &glow::NativeFence) -> bool {
        unsafe { gl.get_sync_status(*fence) == glow::SIGNALED }
    }

    /// Copies a filled slot (whose fence signaled) into `out`, top-down rows, views stacked.
    unsafe fn read_slot(
        &self,
        gl: &glow::Context,
        slot: usize,
        meta: &DepthFrameMeta,
        out: &mut [u8],
    ) -> Result<(), String> {
        if meta.width != self.tex_width || meta.height != self.tex_height {
            return Err("pack buffer was resized".into());
        }
        let stride = Self::pack_row_stride(meta.width);
        let len = Self::slot_bytes(meta.width, meta.height);
        unsafe {
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(self.pbos[slot]));
            let ptr = gl.map_buffer_range(glow::PIXEL_PACK_BUFFER, 0, len as i32, glow::MAP_READ_BIT);
            let result = if ptr.is_null() {
                Err(format!("map failed, GL err {:#x}", gl.get_error()))
            } else {
                let mapped = std::slice::from_raw_parts(ptr as *const u8, len);
                copy_flipped_rows(
                    mapped,
                    stride,
                    out,
                    meta.width as usize * 2,
                    meta.height as usize,
                    2,
                );
                gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
                Ok(())
            };
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            result
        }
    }

    unsafe fn destroy(mut self, gl: &glow::Context) {
        unsafe {
            for fence in self.ring.drain() {
                gl.delete_sync(fence);
            }
            for pbo in self.pbos {
                gl.delete_buffer(pbo);
            }
            gl.delete_framebuffer(self.fbo_read);
            gl.delete_framebuffer(self.fbo_write);
            gl.delete_framebuffer(self.fbo_r16ui);
            gl.delete_texture(self.depth_2d_tex);
            gl.delete_texture(self.r16ui_tex);
        }
    }
}

pub struct StreamContext {
    core_context: Arc<ClientCoreContext>,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_context: Arc<RwLock<InteractionContext>>,
    stage_reference_space: Arc<xr::Space>,
    view_reference_space: Arc<xr::Space>,
    swapchains: [xr::Swapchain<xr::OpenGlEs>; 2],
    last_good_view_params: [ViewParams; 2],
    input_thread: Option<JoinHandle<()>>,
    input_thread_running: Arc<RelaxedAtomic>,
    config: ParsedStreamConfig,
    target_view_resolution: UVec2,
    renderer: StreamRenderer,
    decoder: Option<(VideoDecoderConfig, VideoDecoderSource)>,
    use_custom_reprojection: bool,
    gfx_ctx: Rc<GraphicsContext>,
    depth_provider: Option<crate::extra_extensions::EnvironmentDepthMeta>,
    depth_init_retries: u32,
    depth_init_start: Instant,
    depth_last_retry: Instant,
    system: xr::SystemId,
    depth_pacer: CapturePacer,
    depth_readback: Option<DepthReadback>,
    // Compression and sending run on a worker; the render thread only hands frames over
    depth_link: Option<SendLink<DepthFrameMeta>>,
    depth_skipped_frames: u64,
    depth_perf: StageSet,
    depth_perf_captures: u64,
    depth_frame_index: u64,
    depth_ring_full: u64,
    camera_capture: Option<crate::camera_capture::CameraCapture>,
    camera_capture_right: Option<crate::camera_capture::CameraCapture>,
    last_camera_capture: Instant,
    xr_depth_enabled: bool,
    xr_camera_enabled: bool,
    #[cfg(target_os = "android")]
    camera_encoder: Option<crate::hw_encoder::HwEncoder>,
}

impl StreamContext {
    pub fn new(
        core_ctx: Arc<ClientCoreContext>,
        xr_session: xr::Session<xr::OpenGlEs>,
        gfx_ctx: Rc<GraphicsContext>,
        interaction_ctx: Arc<RwLock<InteractionContext>>,
        config: ParsedStreamConfig,
        system: xr::SystemId,
    ) -> StreamContext {
        interaction_ctx
            .write()
            .select_sources(&config.interaction_sources);

        let xr_exts = xr_session.instance().exts();

        if xr_exts.fb_display_refresh_rate.is_some() {
            xr_session
                .request_display_refresh_rate(config.refresh_rate_hint)
                .unwrap();
        }

        let foveation_profile = if let Some(config) = &config.clientside_foveation_config
            && xr_exts.fb_swapchain_update_state.is_some()
            && xr_exts.fb_foveation.is_some()
            && xr_exts.fb_foveation_configuration.is_some()
        {
            let level;
            let dynamic;
            match config.mode {
                ClientsideFoveationMode::Static { level: lvl } => {
                    level = lvl;
                    dynamic = false;
                }
                ClientsideFoveationMode::Dynamic { max_level } => {
                    level = max_level;
                    dynamic = true;
                }
            };

            xr_session
                .create_foveation_profile(Some(xr::FoveationLevelProfile {
                    level: xr::FoveationLevelFB::from_raw(level as i32),
                    vertical_offset: config.vertical_offset_deg,
                    dynamic: xr::FoveationDynamicFB::from_raw(dynamic as i32),
                }))
                .ok()
        } else {
            None
        };

        let target_view_resolution = alvr_graphics::compute_target_view_resolution(
            config.view_resolution,
            &config.upscaling,
        );
        let format = graphics::swapchain_format(&gfx_ctx, &xr_session, config.enable_hdr);

        let swapchains = [
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
            graphics::create_swapchain(
                &xr_session,
                &gfx_ctx,
                target_view_resolution,
                format,
                foveation_profile.as_ref(),
            ),
        ];

        let renderer = StreamRenderer::new(
            Rc::clone(&gfx_ctx),
            config.view_resolution,
            target_view_resolution,
            [
                swapchains[0]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
                swapchains[1]
                    .enumerate_images()
                    .unwrap()
                    .iter()
                    .map(|i| *i as _)
                    .collect(),
            ],
            format,
            config.foveated_encoding_config.clone(),
            core_ctx.platform() != Platform::Lynx
                && !((core_ctx.platform().is_pico()
                    || (core_ctx.platform() == Platform::SamsungGalaxyXR))
                    && config.enable_hdr),
            // TODO: Find a driver heuristic for the limited range bug instead?
            core_ctx.platform() != Platform::SamsungGalaxyXR && !config.enable_hdr,
            config.encoding_gamma,
            config.upscaling.clone(),
        );

        {
            let int_ctx = interaction_ctx.read();
            core_ctx.send_active_interaction_profile(
                *HAND_LEFT_ID,
                int_ctx.hands_interaction[0].controllers_profile_id,
                int_ctx.hands_interaction[0].input_ids.clone(),
            );
            core_ctx.send_active_interaction_profile(
                *HAND_RIGHT_ID,
                int_ctx.hands_interaction[1].controllers_profile_id,
                int_ctx.hands_interaction[1].input_ids.clone(),
            );
        }

        let input_thread_running = Arc::new(RelaxedAtomic::new(false));

        let stage_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        let view_reference_space = Arc::new(interaction::get_reference_space(
            &xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        // Request USE_SCENE permission at runtime — required by Meta Quest for both
        // the Depth API (XR_META_environment_depth) and Scene API (XR_FB_scene).
        // Always request because the viewer can toggle these features at runtime.
        #[cfg(target_os = "android")]
        alvr_system_info::try_get_permission("com.oculus.permission.USE_SCENE");

        // Depth provider will be lazily initialized in maybe_capture_depth()
        // because it requires passthrough to be fully running (at least one frame submitted)
        let depth_provider: Option<crate::extra_extensions::EnvironmentDepthMeta> = None;
        let depth_wants_init = config
            .xr_data_config
            .as_ref()
            .is_some_and(|c| c.enable_depth)
            && xr_exts.meta_environment_depth.is_some();
        alvr_common::info!("[XR_DIAG] Depth provider deferred init: wants_init={depth_wants_init}");

        // GL depth readback resources (blit-based, see DepthReadback)
        let depth_readback = if depth_wants_init {
            gfx_ctx.make_current();
            let readback = unsafe { DepthReadback::new(&gfx_ctx.gl_context) };
            alvr_common::info!(
                "[XR_DIAG] GL depth readback pipeline created (blit-based, {DEPTH_READBACK_SLOTS} async pack buffers)"
            );
            Some(readback)
        } else {
            None
        };

        // Exits when `depth_link` is dropped with the StreamContext
        let depth_link = depth_wants_init.then(|| {
            let (link, worker) = send_link::<DepthFrameMeta>(1);
            let core_context = Arc::clone(&core_ctx);
            let xr_instance = xr_session.instance().clone();
            thread::spawn(move || {
                let mut perf = StageSet::default();
                let mut sent = 0u64;
                worker.run(|depth_bytes, meta| {
                    send_depth_frame(&core_context, &xr_instance, &mut perf, &meta, depth_bytes);
                    sent += 1;
                    if sent % DEPTH_PERF_REPORT_INTERVAL == 0 {
                        alvr_common::info!(
                            "[XR_PERF] depth worker {sent} frames: {}",
                            perf.format_and_reset()
                        );
                    }
                });
            });
            link
        });

        let mut this = StreamContext {
            use_custom_reprojection: core_ctx.platform().is_yvr(),
            core_context: core_ctx,
            xr_session,
            interaction_context: interaction_ctx,
            stage_reference_space,
            view_reference_space,
            swapchains,
            last_good_view_params: [ViewParams::DUMMY; 2],
            input_thread: None,
            input_thread_running,
            config,
            target_view_resolution,
            renderer,
            decoder: None,
            gfx_ctx,
            depth_provider,
            depth_init_retries: if depth_wants_init { 0 } else { u32::MAX },
            depth_init_start: Instant::now(),
            depth_last_retry: Instant::now(),
            system,
            depth_pacer: CapturePacer::default(),
            depth_readback,
            depth_link,
            depth_skipped_frames: 0,
            depth_perf: StageSet::default(),
            depth_perf_captures: 0,
            depth_frame_index: 0,
            depth_ring_full: 0,
            camera_capture: None,
            camera_capture_right: None,
            last_camera_capture: Instant::now(),
            xr_depth_enabled: false,
            xr_camera_enabled: false,
            #[cfg(target_os = "android")]
            camera_encoder: None,
        };

        // Request camera permissions early so the user has time to grant
        // before the first capture attempt (the request is non-blocking)
        #[cfg(target_os = "android")]
        {
            alvr_system_info::try_get_permission("android.permission.CAMERA");
            alvr_system_info::try_get_permission("horizonos.permission.PASSTHROUGH_CAMERA_ACCESS");
        }

        this.update_reference_space();

        this
    }

    pub fn uses_passthrough(&self) -> bool {
        self.config.passthrough.is_some()
    }

    pub fn update_reference_space(&mut self) {
        self.input_thread_running.set(false);

        self.stage_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::STAGE,
        ));
        self.view_reference_space = Arc::new(interaction::get_reference_space(
            &self.xr_session,
            xr::ReferenceSpaceType::VIEW,
        ));

        self.core_context.send_playspace(
            self.xr_session
                .reference_space_bounds_rect(xr::ReferenceSpaceType::STAGE)
                .unwrap()
                .map(|a| Vec2::new(a.width, a.height)),
        );

        if let Some(running) = self.input_thread.take() {
            running.join().ok();
        }

        self.input_thread_running.set(true);

        self.input_thread = Some(thread::spawn({
            let core_ctx = Arc::clone(&self.core_context);
            let xr_session = self.xr_session.clone();
            let interaction_ctx = Arc::clone(&self.interaction_context);
            let stage_reference_space = Arc::clone(&self.stage_reference_space);
            let view_reference_space = Arc::clone(&self.view_reference_space);
            let refresh_rate = self.config.refresh_rate_hint;
            let running = Arc::clone(&self.input_thread_running);
            move || {
                stream_input_loop(
                    &core_ctx,
                    xr_session,
                    &interaction_ctx,
                    &stage_reference_space,
                    &view_reference_space,
                    refresh_rate,
                    running,
                )
            }
        }));
    }

    pub fn maybe_initialize_decoder(&mut self, codec: CodecType, config_nal: Vec<u8>) {
        let new_config = VideoDecoderConfig {
            codec,
            force_software_decoder: self.config.force_software_decoder,
            max_buffering_frames: self.config.max_buffering_frames,
            buffering_history_weight: self.config.buffering_history_weight,
            options: self.config.decoder_options.clone(),
            config_buffer: config_nal,
        };

        let maybe_config = if let Some((config, _)) = &self.decoder {
            (new_config != *config).then_some(new_config)
        } else {
            Some(new_config)
        };

        if let Some(config) = maybe_config {
            let (mut sink, source) = video_decoder::create_decoder(config.clone(), {
                let ctx = Arc::clone(&self.core_context);
                move |maybe_timestamp: Result<Duration>| match maybe_timestamp {
                    Ok(timestamp) => ctx.report_frame_decoded(timestamp),
                    Err(e) => ctx.report_fatal_decoder_error(&e.to_string()),
                }
            });
            self.decoder = Some((config, source));

            self.core_context.set_decoder_input_callback(Box::new(
                move |timestamp, buffer| -> bool { sink.push_nal(timestamp, buffer) },
            ));
        }
    }

    pub fn update_real_time_config(&mut self, config: &RealTimeConfig) {
        self.config.passthrough = config.passthrough.clone();
        self.config.clientside_post_processing = config.clientside_post_processing.clone();

        // Hot-reload XR data config if it changed
        if let Some(new_xr) = &config.xr_data_config {
            let old_xr = self.config.xr_data_config.as_ref();
            let changed = old_xr.map_or(true, |old| old != new_xr);
            if changed {
                // Detect what specifically changed to react appropriately
                if let Some(old) = old_xr {
                    // Camera bitrate changed → destroy encoder so it recreates with new bitrate
                    #[cfg(target_os = "android")]
                    if (old.camera_bitrate_mbps - new_xr.camera_bitrate_mbps).abs() > 0.1
                        || old.camera_fps != new_xr.camera_fps
                    {
                        if self.camera_encoder.take().is_some() {
                            alvr_common::info!(
                                "[XR_DATA] Camera encoder destroyed (config changed: bitrate {:.0}→{:.0} Mbps, fps {:.0}→{:.0})",
                                old.camera_bitrate_mbps, new_xr.camera_bitrate_mbps,
                                old.camera_fps, new_xr.camera_fps,
                            );
                        }
                    }

                    // Camera resolution changed → destroy captures, re-init on next frame
                    if old.camera_width != new_xr.camera_width || old.camera_height != new_xr.camera_height {
                        alvr_common::info!(
                            "[XR_DATA] Camera resolution changed: {}x{}→{}x{}, reinitializing...",
                            old.camera_width, old.camera_height,
                            new_xr.camera_width, new_xr.camera_height,
                        );
                        if let Some(cam) = self.camera_capture.take() {
                            cam.destroy_async();
                        }
                        if let Some(cam) = self.camera_capture_right.take() {
                            cam.destroy_async();
                        }
                        #[cfg(target_os = "android")]
                        { self.camera_encoder.take(); }
                    }
                }

                alvr_common::info!(
                    "[XR_DATA] Config updated: depth_fps={:.0}, camera_fps={:.0}, cam={}x{}, bitrate={:.0}Mbps",
                    new_xr.depth_fps, new_xr.camera_fps,
                    new_xr.camera_width, new_xr.camera_height,
                    new_xr.camera_bitrate_mbps,
                );
                self.config.xr_data_config = Some(new_xr.clone());
            }
        }
    }

    pub fn set_xr_stream_flags(&mut self, depth: bool, camera: bool) {
        alvr_common::info!(
            "[XR_DATA] Stream flags updated: depth={depth}, camera={camera}"
        );
        // Destroy camera encoder when camera stream is disabled to free HW resources
        #[cfg(target_os = "android")]
        {
            if !camera && self.xr_camera_enabled {
                if self.camera_encoder.take().is_some() {
                    alvr_common::info!("[XR_DATA] Camera encoder destroyed (stream disabled)");
                }
            }
        }
        // Destroy camera capture when camera stream is disabled.
        // Use destroy_async() to move NDK cleanup off the render thread —
        // ACameraCaptureSession_close / ACameraDevice_close can block for a
        // long time (or indefinitely) on some Android drivers.
        if !camera && self.xr_camera_enabled {
            if let Some(cam) = self.camera_capture.take() {
                alvr_common::info!("[XR_DATA] Camera capture (left) destroying async...");
                cam.destroy_async();
            }
            if let Some(cam) = self.camera_capture_right.take() {
                alvr_common::info!("[XR_DATA] Camera capture (right) destroying async...");
                cam.destroy_async();
            }
        }
        self.xr_depth_enabled = depth;
        self.xr_camera_enabled = camera;
    }

    pub fn render(
        &mut self,
        frame_interval: Duration,
        vsync_time: Duration,
    ) -> (ProjectionLayerBuilder<'_>, Duration) {
        let xr_vsync_time = xr::Time::from_nanos(vsync_time.as_nanos() as _);
        let frame_poll_deadline = Instant::now()
            + Duration::from_secs_f32(
                frame_interval.as_secs_f32() * DECODER_MAX_TIMEOUT_MULTIPLIER,
            );
        let mut frame_result = None;
        if let Some((_, source)) = &mut self.decoder {
            while frame_result.is_none() && Instant::now() < frame_poll_deadline {
                frame_result = source.get_frame();
                thread::sleep(Duration::from_micros(500));
            }
        }

        let (timestamp, view_params, buffer_ptr) =
            if let Some((timestamp, buffer_ptr)) = frame_result {
                let view_params = self.core_context.report_compositor_start(timestamp);

                self.last_good_view_params = view_params;

                (timestamp, view_params, buffer_ptr)
            } else {
                (vsync_time, self.last_good_view_params, ptr::null_mut())
            };

        let left_swapchain_idx = self.swapchains[0].acquire_image().unwrap();
        let right_swapchain_idx = self.swapchains[1].acquire_image().unwrap();

        self.swapchains[0]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();
        self.swapchains[1]
            .wait_image(xr::Duration::INFINITE)
            .unwrap();

        let (flags, maybe_views) = self
            .xr_session
            .locate_views(
                xr::ViewConfigurationType::PRIMARY_STEREO,
                xr_vsync_time,
                &self.stage_reference_space,
            )
            .unwrap();

        let current_headset_views = if flags.contains(xr::ViewStateFlags::ORIENTATION_VALID) {
            maybe_views
        } else {
            vec![crate::default_view(), crate::default_view()]
        };

        // The poses and FoVs we received from the PC runtime, which may differ and/or include
        // altered FoVs based on settings and view conversions done for canting.
        let input_view_params = view_params;
        let mut output_view_params = input_view_params;
        // Avoid passing invalid timestamp to runtime.
        // `timestamp` is generally a current vsync time, but may be repeated if frames are
        // dropped. Some runtimes dislike it if the timestamp is repeated for too long, so after
        // one second we begin presenting a lagged vsync time instead.
        let mut openxr_display_time =
            Duration::max(timestamp, vsync_time.saturating_sub(Duration::from_secs(1)));

        // (shinyquagsire23) I don't entirely trust runtimes to implement CompositionLayerProjectionView
        // correctly, but if we do trust them, avoid doing rotation ourselves. Otherwise, rerender.
        // Ex: YVR/PFDMR has issues with aspect ratio mismatches and passthrough compositing.
        if self.use_custom_reprojection {
            output_view_params = [
                ViewParams {
                    pose: crate::from_xr_pose(current_headset_views[0].pose),
                    fov: crate::from_xr_fov(current_headset_views[0].fov),
                },
                ViewParams {
                    pose: crate::from_xr_pose(current_headset_views[1].pose),
                    fov: crate::from_xr_fov(current_headset_views[1].fov),
                },
            ];

            openxr_display_time = vsync_time;
        }

        self.renderer.render(
            buffer_ptr,
            [
                StreamViewParams {
                    swapchain_index: left_swapchain_idx,
                    input_view_params: input_view_params[0],
                    output_view_params: output_view_params[0],
                },
                StreamViewParams {
                    swapchain_index: right_swapchain_idx,
                    input_view_params: input_view_params[1],
                    output_view_params: output_view_params[1],
                },
            ],
            self.config.passthrough.as_ref(),
        );

        self.swapchains[0].release_image().unwrap();
        self.swapchains[1].release_image().unwrap();

        // Capture depth and camera (must be between xrWaitFrame and xrEndFrame)
        self.maybe_capture_depth(xr_vsync_time);
        self.maybe_capture_camera();

        if !buffer_ptr.is_null()
            && let Some(xr_now) = crate::xr_runtime_now(self.xr_session.instance())
        {
            self.core_context.report_submit(
                timestamp,
                vsync_time.saturating_sub(Duration::from_nanos(xr_now.as_nanos() as u64)),
            );
        }

        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: self.target_view_resolution.x as _,
                height: self.target_view_resolution.y as _,
            },
        };

        let clientside_post_processing = self
            .xr_session
            .instance()
            .exts()
            .fb_composition_layer_settings
            .and(self.config.clientside_post_processing.clone());

        let layer = ProjectionLayerBuilder::new(
            &self.stage_reference_space,
            [
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(output_view_params[0].pose))
                    .fov(crate::to_xr_fov(output_view_params[0].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[0])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(crate::to_xr_pose(output_view_params[1].pose))
                    .fov(crate::to_xr_fov(output_view_params[1].fov))
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[1])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
            ],
            self.config
                .passthrough
                .clone()
                .map(|mode| ProjectionLayerAlphaConfig {
                    premultiplied: matches!(
                        mode,
                        PassthroughMode::Blend {
                            premultiplied_alpha: true,
                            ..
                        } | PassthroughMode::RgbChromaKey(_)
                            | PassthroughMode::HsvChromaKey(_)
                    ),
                }),
            clientside_post_processing,
        );

        (layer, openxr_display_time)
    }

    fn maybe_capture_depth(&mut self, display_time: xr::Time) {
        // Lazy init with retries: always runs based on config (not gated by viewer toggle).
        // This ensures the provider is created early while passthrough is freshly started.
        if self.depth_provider.is_none() && self.depth_init_retries < 10 {
            let elapsed = self.depth_init_start.elapsed();
            let since_last = self.depth_last_retry.elapsed();
            if elapsed < Duration::from_secs(2) || since_last < Duration::from_secs(1) {
                // Still waiting for initial delay or retry interval
            } else {
                self.depth_init_retries += 1;
                self.depth_last_retry = Instant::now();
                alvr_common::info!(
                    "[XR_DATA] Deferred depth init attempt {}/10 ({:.1}s after stream start)...",
                    self.depth_init_retries,
                    elapsed.as_secs_f32(),
                );
                // Runtime needs GL context current to allocate GLES textures for depth swapchain
                self.gfx_ctx.make_current();
                match crate::extra_extensions::EnvironmentDepthMeta::new(
                    self.xr_session.clone(),
                    self.system,
                ) {
                    Ok(provider) => {
                        alvr_common::info!("[XR_DATA] Deferred depth provider created successfully!");
                        self.depth_provider = Some(provider);
                    }
                    Err(e) => {
                        alvr_common::info!("[XR_DATA] Deferred depth provider attempt {} failed: {e:?}", self.depth_init_retries);
                    }
                }
            }
        }

        self.depth_frame_index += 1;
        let frame_start = Instant::now();
        let mut did_work = false;

        // Collect readbacks queued on earlier frames whose GPU work has finished
        if self.depth_readback.as_ref().is_some_and(|r| r.ring.pending() > 0) {
            did_work |= self.collect_depth_readbacks();
        }

        did_work |= self.maybe_enqueue_depth(display_time);

        if did_work {
            self.depth_perf.record("render_thread_ms", elapsed_ms(frame_start));
        }
    }

    /// Maps every finished readback slot and sends its frame. Returns whether any slot finished.
    fn collect_depth_readbacks(&mut self) -> bool {
        let Some(readback) = &mut self.depth_readback else {
            return false;
        };
        self.gfx_ctx.make_current();
        let gl = &self.gfx_ctx.gl_context;
        let mut finished = false;
        loop {
            let event = readback.ring.poll_one(
                self.depth_frame_index,
                DEPTH_READBACK_TIMEOUT_FRAMES,
                |fence| unsafe { DepthReadback::is_signaled(gl, fence) },
            );
            let Some(event) = event else {
                break;
            };
            finished = true;
            match event {
                SlotEvent::Ready {
                    slot,
                    fence,
                    meta,
                    latency_frames,
                } => {
                    unsafe { gl.delete_sync(fence) };
                    self.depth_perf.record("fence_frames", latency_frames as f32);
                    self.depth_perf
                        .record("fence_ms", elapsed_ms(meta.enqueued_at));
                    let Some(link) = &mut self.depth_link else {
                        continue;
                    };
                    let map_start = Instant::now();
                    let mut depth_bytes =
                        link.take_buffer(meta.width as usize * meta.height as usize * 2 * 2);
                    let result = unsafe { readback.read_slot(gl, slot, &meta, &mut depth_bytes) };
                    self.depth_perf.record("map_ms", elapsed_ms(map_start));
                    match result {
                        Ok(()) => {
                            if let SubmitOutcome::Disconnected = link.submit(depth_bytes, meta) {
                                alvr_common::warn!("[XR_DATA] depth send worker is gone");
                            }
                            self.depth_perf_captures += 1;
                            if self.depth_perf_captures % DEPTH_PERF_REPORT_INTERVAL == 0 {
                                alvr_common::info!(
                                    "[XR_PERF] depth {} frames read back, {} skipped, {} ring-full frames, \
                                     {} dropped at the worker, {} buffers: {}",
                                    self.depth_perf_captures,
                                    self.depth_skipped_frames,
                                    self.depth_ring_full,
                                    link.dropped(),
                                    link.allocations(),
                                    self.depth_perf.format_and_reset()
                                );
                            }
                        }
                        Err(failure) => {
                            Self::note_depth_skip(&mut self.depth_skipped_frames, Some(failure))
                        }
                    }
                }
                SlotEvent::Expired { fence, .. } => {
                    unsafe { gl.delete_sync(fence) };
                    Self::note_depth_skip(
                        &mut self.depth_skipped_frames,
                        Some(format!(
                            "GPU readback not done after {DEPTH_READBACK_TIMEOUT_FRAMES} frames"
                        )),
                    );
                }
            }
        }
        finished
    }

    /// Acquires a depth image if a capture is due and queues its GPU readback. Returns whether
    /// GPU work was queued.
    fn maybe_enqueue_depth(&mut self, display_time: xr::Time) -> bool {
        // Only capture/send depth frames when the viewer has enabled depth streaming
        if !self.xr_depth_enabled {
            return false;
        }

        let Some(depth_provider) = &mut self.depth_provider else {
            return false;
        };

        let depth_fps = self
            .config
            .xr_data_config
            .as_ref()
            .map(|c| c.depth_fps)
            .unwrap_or(10.0);
        let depth_interval = Duration::from_secs_f32(1.0 / depth_fps);

        let now = Instant::now();
        if !self.depth_pacer.is_due(now) {
            return false;
        }

        // Start provider if not already started
        if !depth_provider.is_started() {
            alvr_common::info!("[XR_DATA] Starting depth provider...");
            if let Err(e) = depth_provider.start() {
                alvr_common::info!("[XR_DATA] Failed to start depth provider: {e:?}");
                self.depth_pacer.defer(now, depth_interval);
                return false;
            }
            alvr_common::info!("[XR_DATA] Depth provider started");
            // Runtime needs at least one frame after start before acquire works
            self.depth_pacer.defer(now, depth_interval);
            return false;
        }

        // Without a readback there is no depth: skip the frame instead of sending a constant
        // 0x8080 image that viewers would take for real depth.
        let Some(readback) = &mut self.depth_readback else {
            Self::note_depth_skip(&mut self.depth_skipped_frames, None);
            self.depth_pacer.defer(now, depth_interval);
            return false;
        };

        // Never wait for the GPU: with every slot still in flight, try again next frame
        let Some(slot) = readback.ring.free_slot() else {
            self.depth_ring_full += 1;
            return false;
        };

        // Acquire depth image
        let acquire_start = Instant::now();
        let depth_image = match depth_provider.acquire_depth_image(
            self.stage_reference_space.as_raw(),
            display_time,
        ) {
            Ok(Some(image)) => image,
            // No new depth image yet: retry on the next frame
            Ok(None) => return false,
            Err(e) => {
                alvr_common::info!("[XR_DATA] Failed to acquire depth image: {e:?}");
                self.depth_pacer.defer(now, depth_interval);
                return false;
            }
        };
        self.depth_perf.record("acquire_ms", elapsed_ms(acquire_start));

        let width = depth_provider.swapchain_width;
        let height = depth_provider.swapchain_height;
        let swapchain_idx = depth_image.swapchain_index as usize;
        let texture_id = depth_provider
            .swapchain_images
            .get(swapchain_idx)
            .copied()
            .unwrap_or(0);
        let Some(texture_id) = std::num::NonZeroU32::new(texture_id) else {
            Self::note_depth_skip(&mut self.depth_skipped_frames, None);
            self.depth_pacer.on_captured(now, depth_interval);
            return false;
        };

        // Each depth view has its own pose (stage space) and FOV; the depth cameras are not
        // the eye cameras, so both must be forwarded for correct unprojection.
        let meta = DepthFrameMeta {
            display_time,
            view_poses: [0, 1].map(|i| crate::from_xr_pose(depth_image.views[i].pose)),
            fov_angles: [0, 1].map(|i| {
                let fov = depth_image.views[i].fov;
                [fov.angle_left, fov.angle_right, fov.angle_up, fov.angle_down]
            }),
            near_z: depth_image.near_z,
            far_z: depth_image.far_z,
            width,
            height,
            enqueued_at: Instant::now(),
        };

        self.gfx_ctx.make_current();
        let gl = &self.gfx_ctx.gl_context;
        let enqueue_start = Instant::now();
        let result = unsafe {
            readback.ensure_size(gl, width, height);
            readback.enqueue(gl, glow::NativeTexture(texture_id), slot)
        };
        self.depth_perf.record("enqueue_ms", elapsed_ms(enqueue_start));
        match result {
            Ok(fence) => readback.ring.submit(slot, fence, meta, self.depth_frame_index),
            // A failed GL step leaves zeros or stale data in the buffer
            Err(failure) => Self::note_depth_skip(&mut self.depth_skipped_frames, Some(failure)),
        }
        self.depth_pacer.on_captured(now, depth_interval);
        true
    }

    fn note_depth_skip(skipped_frames: &mut u64, failure: Option<String>) {
        *skipped_frames += 1;
        if *skipped_frames <= 3 || *skipped_frames % 100 == 0 {
            alvr_common::warn!(
                "[XR_DATA] depth readback unavailable, frame skipped ({} so far){}",
                skipped_frames,
                failure.map(|f| format!(": {f}")).unwrap_or_default()
            );
        }
    }

    fn maybe_capture_camera(&mut self) {
        if !self.xr_camera_enabled {
            return;
        }
        let camera_fps = self
            .config
            .xr_data_config
            .as_ref()
            .map(|c| c.camera_fps)
            .unwrap_or(15.0);
        let camera_interval = Duration::from_secs_f32(1.0 / camera_fps);

        if self.last_camera_capture.elapsed() < camera_interval {
            return;
        }

        // Lazy init: open left camera (ID 50), then try right camera (ID 51)
        if self.camera_capture.is_none() {
            let cam_w = self.config.xr_data_config.as_ref().map(|c| c.camera_width).unwrap_or(640) as i32;
            let cam_h = self.config.xr_data_config.as_ref().map(|c| c.camera_height).unwrap_or(480) as i32;
            alvr_common::info!("[XR_DATA] Initializing dual camera capture at {cam_w}x{cam_h}...");
            match crate::camera_capture::CameraCapture::new_with_camera_id(cam_w, cam_h, Some("50")) {
                Some(cam) => {
                    alvr_common::info!("[XR_DATA] Left camera (50) opened: {}x{}", cam.width(), cam.height());
                    self.camera_capture = Some(cam);
                }
                None => {
                    // Fallback: auto-select any camera
                    alvr_common::info!("[XR_DATA] Camera 50 unavailable, trying auto-select...");
                    match crate::camera_capture::CameraCapture::new(cam_w, cam_h) {
                        Some(cam) => {
                            alvr_common::info!("[XR_DATA] Fallback camera opened: {}x{}", cam.width(), cam.height());
                            self.camera_capture = Some(cam);
                        }
                        None => {
                            alvr_common::info!("[XR_DATA] Camera capture unavailable");
                            self.last_camera_capture = Instant::now() + Duration::from_secs(5);
                            return;
                        }
                    }
                }
            }
            // Try to open right camera (ID 51) for stereo
            if self.camera_capture_right.is_none() {
                match crate::camera_capture::CameraCapture::new_with_camera_id(cam_w, cam_h, Some("51")) {
                    Some(cam) => {
                        alvr_common::info!("[XR_DATA] Right camera (51) opened: {}x{}", cam.width(), cam.height());
                        self.camera_capture_right = Some(cam);
                    }
                    None => {
                        alvr_common::info!("[XR_DATA] Right camera (51) unavailable, single-eye mode");
                    }
                }
            }
        }

        // Get frames from both cameras
        let left_frame = self.camera_capture.as_ref().and_then(|c| c.get_latest_frame());
        let right_frame = self.camera_capture_right.as_ref().and_then(|c| c.get_latest_frame());

        let Some((l_w, l_h, l_data, l_nv12)) = left_frame else {
            return;
        };

        // Build the frame data: side-by-side if both eyes available
        let (width, height, data, is_nv12) = if let Some((r_w, r_h, r_data, r_nv12)) = right_frame {
            if l_nv12 && r_nv12 && l_h == r_h {
                // Tile side-by-side in NV12: combined_w = l_w + r_w
                let combined_w = (l_w + r_w) as usize;
                let h = l_h as usize;
                let lw = l_w as usize;
                let rw = r_w as usize;

                let y_size = combined_w * h;
                let uv_size = combined_w * (h / 2);
                let mut combined = vec![0u8; y_size + uv_size];

                // Tile Y planes side-by-side
                for row in 0..h {
                    let dst_off = row * combined_w;
                    let l_off = row * lw;
                    let r_off = row * rw;
                    combined[dst_off..dst_off + lw]
                        .copy_from_slice(&l_data[l_off..l_off + lw]);
                    if r_off + rw <= r_data.len() {
                        combined[dst_off + lw..dst_off + combined_w]
                            .copy_from_slice(&r_data[r_off..r_off + rw]);
                    }
                }

                // Tile UV planes side-by-side
                let l_uv_off = lw * h;
                let r_uv_off = rw * h;
                let uv_h = h / 2;
                for row in 0..uv_h {
                    let dst_off = y_size + row * combined_w;
                    let l_off = l_uv_off + row * lw;
                    let r_off = r_uv_off + row * rw;
                    if l_off + lw <= l_data.len() {
                        combined[dst_off..dst_off + lw]
                            .copy_from_slice(&l_data[l_off..l_off + lw]);
                    }
                    if r_off + rw <= r_data.len() {
                        combined[dst_off + lw..dst_off + combined_w]
                            .copy_from_slice(&r_data[r_off..r_off + rw]);
                    }
                }

                (l_w + r_w, l_h, combined, true)
            } else {
                // Mismatched heights or not NV12, just send left
                (l_w, l_h, l_data, l_nv12)
            }
        } else {
            // Single camera only
            (l_w, l_h, l_data, l_nv12)
        };

        use alvr_packets::{CameraFrameFormat, CameraFrameHeader};

        // Try HW encoding on Android: NV12 → H264
        #[cfg(target_os = "android")]
        let (send_data, send_format) = if is_nv12 {
            // Lazily create camera encoder (dimensions may change with stereo)
            if self.camera_encoder.as_ref().map_or(true, |e| e.width() != width || e.height() != height) {
                self.camera_encoder = None; // recreate with new dimensions
                let fps = self.config.xr_data_config.as_ref()
                    .map(|c| c.camera_fps as i32).unwrap_or(15);
                let bitrate_bps = (self.config.xr_data_config.as_ref()
                    .map(|c| c.camera_bitrate_mbps).unwrap_or(15.0) * 1_000_000.0) as i32;
                self.camera_encoder = crate::hw_encoder::HwEncoder::new(
                    width, height,
                    bitrate_bps,
                    fps,
                    1,
                );
                if self.camera_encoder.is_some() {
                    alvr_common::info!("[XR_DATA] HW encoder created: {}x{}", width, height);
                } else {
                    alvr_common::info!("[XR_DATA] HW encoder unavailable, falling back to raw NV12");
                }
            }

            if let Some(ref mut encoder) = self.camera_encoder {
                if let Some(nals) = encoder.encode(&data) {
                    (nals.to_vec(), CameraFrameFormat::H264)
                } else {
                    self.last_camera_capture = Instant::now();
                    return;
                }
            } else {
                (data, CameraFrameFormat::Nv12)
            }
        } else {
            (data, CameraFrameFormat::Rgb8)
        };

        #[cfg(not(target_os = "android"))]
        let (send_data, send_format) = if is_nv12 {
            (data, CameraFrameFormat::Nv12)
        } else {
            (data, CameraFrameFormat::Rgb8)
        };

        let header = CameraFrameHeader {
            timestamp: Duration::from_nanos(0),
            head_pose: alvr_common::Pose::default(),
            width,
            height,
            format: send_format,
            intrinsics: [0.0; 4],
        };
        self.core_context.send_camera_frame(&header, &send_data);
        self.last_camera_capture = Instant::now();
    }

}

impl Drop for StreamContext {
    fn drop(&mut self) {
        self.input_thread_running.set(false);
        self.input_thread.take().unwrap().join().ok();

        // Clean up depth readback GL resources
        if let Some(readback) = self.depth_readback.take() {
            self.gfx_ctx.make_current();
            unsafe { readback.destroy(&self.gfx_ctx.gl_context) };
        }
    }
}

/// Runs on the depth send worker: compresses one frame and sends it with its header.
fn send_depth_frame(
    core_context: &ClientCoreContext,
    xr_instance: &xr::Instance,
    perf: &mut StageSet,
    meta: &DepthFrameMeta,
    depth_bytes: &[u8],
) {
    // LZ4 compress the raw D16 depth bytes (lossless, fast, ~2-4x compression).
    let lz4_start = Instant::now();
    let compressed = lz4_flex::compress_prepend_size(depth_bytes);
    perf.record("lz4_ms", elapsed_ms(lz4_start));
    let (send_data, send_format) = (compressed, alvr_packets::DepthFrameFormat::Lz4D16);

    // Sampled after readback and compression, as close to the send as possible, so that the
    // server-side offset estimate (receive time - send time) only contains network latency.
    let client_send_time = crate::xr_runtime_now(xr_instance)
        .map(crate::from_xr_time)
        .unwrap_or_else(|| crate::from_xr_time(meta.display_time));

    let header = DepthFrameHeader {
        timestamp: crate::from_xr_time(meta.display_time),
        client_send_time,
        view_poses: meta.view_poses,
        width: meta.width,
        height: meta.height * 2,
        near_z: meta.near_z,
        far_z: meta.far_z,
        format: send_format,
        fov_angles: meta.fov_angles,
    };

    let send_start = Instant::now();
    core_context.send_depth_frame(&header, &send_data);
    perf.record("send_ms", elapsed_ms(send_start));
}

fn stream_input_loop(
    core_ctx: &ClientCoreContext,
    xr_session: xr::Session<xr::OpenGlEs>,
    interaction_ctx: &RwLock<InteractionContext>,
    stage_reference_space: &xr::Space,
    view_reference_space: &xr::Space,
    refresh_rate: f32,
    running: Arc<RelaxedAtomic>,
) {
    let mut last_controller_poses = [Pose::IDENTITY; 2];
    let mut last_palm_poses = [Pose::IDENTITY; 2];
    let mut last_view_params = [ViewParams::DUMMY; 2];

    let mut deadline = Instant::now();
    let frame_interval = Duration::from_secs_f32(1.0 / refresh_rate);
    while running.value() {
        let int_ctx = &*interaction_ctx.read();
        // Streaming related inputs are updated here. Make sure every input poll is done in this
        // thread
        if let Err(e) = xr_session.sync_actions(&[(&int_ctx.action_set).into()]) {
            error!("{e}");
            return;
        }

        let Some(now) = crate::xr_runtime_now(xr_session.instance()).map(crate::from_xr_time)
        else {
            error!("Cannot poll tracking: invalid time");
            return;
        };

        let target_time = now + core_ctx.get_total_prediction_offset();

        let Some((head_motion, local_views)) = interaction::get_head_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            view_reference_space,
            now,
            target_time,
            &last_view_params,
        ) else {
            continue;
        };

        if let Some(views) = local_views {
            core_ctx.send_view_params(views);
            last_view_params = views;
        }

        let mut device_motions = Vec::with_capacity(3);

        device_motions.push((*HEAD_ID, head_motion));

        let left_hand_data = crate::interaction::get_hand_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[0],
            &mut last_controller_poses[0],
            &mut last_palm_poses[0],
        );
        let right_hand_data = crate::interaction::get_hand_data(
            &xr_session,
            core_ctx.platform(),
            stage_reference_space,
            now,
            target_time,
            &int_ctx.hands_interaction[1],
            &mut last_controller_poses[1],
            &mut last_palm_poses[1],
        );

        // Note: When multimodal input is enabled, we are sure that when free hands are used
        // (not holding controllers) the controller data is None.
        if (int_ctx.multimodal_hands_enabled || left_hand_data.skeleton_joints.is_none())
            && let Some(motion) = left_hand_data.grip_motion
        {
            device_motions.push((*HAND_LEFT_ID, motion));
        }
        if (int_ctx.multimodal_hands_enabled || right_hand_data.skeleton_joints.is_none())
            && let Some(motion) = right_hand_data.grip_motion
        {
            device_motions.push((*HAND_RIGHT_ID, motion));
        }

        if int_ctx.multimodal_hands_enabled
            && let Some(detached_controller) = left_hand_data.detached_grip_motion
        {
            device_motions.push((*DETACHED_CONTROLLER_LEFT_ID, detached_controller));
        }
        if int_ctx.multimodal_hands_enabled
            && let Some(detached_controller) = right_hand_data.detached_grip_motion
        {
            device_motions.push((*DETACHED_CONTROLLER_RIGHT_ID, detached_controller));
        }

        let face = interaction::get_face_data(
            &xr_session,
            &int_ctx.face_sources,
            view_reference_space,
            now,
        );

        let body = int_ctx
            .body_source
            .as_ref()
            .and_then(|source| interaction::get_body_skeleton(source, stage_reference_space, now));

        if let Some(source) = &int_ctx.body_source {
            device_motions.append(&mut interaction::get_bd_motion_trackers(source, now));
        }

        // Even though the server is already adding the motion-to-photon latency, here we use
        // target_time as the poll_timestamp to compensate for the fact that video frames are sent
        // with the poll timestamp instead of the vsync time. This is to ensure correctness when
        // submitting frames to OpenXR. This won't cause any desync with the server because no time
        // sync step is performed between client and server.
        core_ctx.send_tracking(TrackingData {
            poll_timestamp: target_time,
            device_motions,
            hand_skeletons: [
                left_hand_data.skeleton_joints,
                right_hand_data.skeleton_joints,
            ],
            face,
            body,
        });

        let button_entries = interaction::update_buttons(&xr_session, &int_ctx.button_actions);
        if !button_entries.is_empty() {
            core_ctx.send_buttons(button_entries);
        }

        deadline += frame_interval / 3;
        thread::sleep(deadline.saturating_duration_since(Instant::now()));
    }
}
