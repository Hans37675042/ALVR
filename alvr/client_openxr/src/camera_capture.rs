// Camera capture using Android Camera2 NDK API for Quest 3 passthrough cameras.
// Captures YUV_420_888 frames from the left camera and provides them for streaming.

use std::ffi::{c_char, c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

// ── Camera2 NDK FFI bindings ──────────────────────────────────────────────

// Opaque types
#[repr(C)]
pub struct ACameraManager {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACameraIdList {
    pub num_cameras: c_int,
    pub camera_ids: *const *const c_char,
}
#[repr(C)]
pub struct ACameraDevice {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACameraCaptureSession {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACaptureRequest {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACameraOutputTarget {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACaptureSessionOutputContainer {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACaptureSessionOutput {
    _private: [u8; 0],
}
#[repr(C)]
pub struct AImageReader {
    _private: [u8; 0],
}
#[repr(C)]
pub struct AImage {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ANativeWindow {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACameraMetadata {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ACameraMetadata_const_entry {
    pub tag: u32,
    pub type_: u8,
    pub count: u32,
    pub data: *const c_void,
}

// Callback structs
#[repr(C)]
pub struct ACameraDevice_StateCallbacks {
    pub context: *mut c_void,
    pub on_disconnected: Option<unsafe extern "C" fn(*mut c_void, *mut ACameraDevice)>,
    pub on_error: Option<unsafe extern "C" fn(*mut c_void, *mut ACameraDevice, c_int)>,
}

#[repr(C)]
pub struct ACameraCaptureSession_stateCallbacks {
    pub context: *mut c_void,
    pub on_closed: Option<unsafe extern "C" fn(*mut c_void, *mut ACameraCaptureSession)>,
    pub on_ready: Option<unsafe extern "C" fn(*mut c_void, *mut ACameraCaptureSession)>,
    pub on_active: Option<unsafe extern "C" fn(*mut c_void, *mut ACameraCaptureSession)>,
}

#[repr(C)]
pub struct AImageReader_ImageListener {
    pub context: *mut c_void,
    pub on_image_available: Option<unsafe extern "C" fn(*mut c_void, *mut AImageReader)>,
}

// Constants
const ACAMERA_OK: i32 = 0;
const TEMPLATE_PREVIEW: c_int = 1;
const AIMAGE_FORMAT_YUV_420_888: i32 = 0x23;
// ACAMERA_SCALER_START (section 0x000D << 16) + 10
const ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS: u32 = 0x000D000A;

#[link(name = "camera2ndk")]
unsafe extern "C" {
    fn ACameraManager_create() -> *mut ACameraManager;
    fn ACameraManager_delete(manager: *mut ACameraManager);
    fn ACameraManager_getCameraIdList(
        manager: *mut ACameraManager,
        camera_id_list: *mut *mut ACameraIdList,
    ) -> i32;
    fn ACameraManager_deleteCameraIdList(list: *mut ACameraIdList);
    fn ACameraManager_getCameraCharacteristics(
        manager: *mut ACameraManager,
        camera_id: *const c_char,
        characteristics: *mut *mut ACameraMetadata,
    ) -> i32;
    fn ACameraMetadata_getConstEntry(
        metadata: *const ACameraMetadata,
        tag: u32,
        entry: *mut ACameraMetadata_const_entry,
    ) -> i32;
    fn ACameraMetadata_free(metadata: *mut ACameraMetadata);
    fn ACameraManager_openCamera(
        manager: *mut ACameraManager,
        camera_id: *const c_char,
        callbacks: *mut ACameraDevice_StateCallbacks,
        device: *mut *mut ACameraDevice,
    ) -> i32;
    fn ACameraDevice_close(device: *mut ACameraDevice) -> i32;
    fn ACameraDevice_createCaptureRequest(
        device: *mut ACameraDevice,
        template_id: c_int,
        request: *mut *mut ACaptureRequest,
    ) -> i32;
    fn ACaptureRequest_addTarget(
        request: *mut ACaptureRequest,
        output: *const ACameraOutputTarget,
    ) -> i32;
    fn ACaptureRequest_free(request: *mut ACaptureRequest);
    fn ACameraOutputTarget_create(
        window: *mut ANativeWindow,
        output: *mut *mut ACameraOutputTarget,
    ) -> i32;
    fn ACameraOutputTarget_free(output: *mut ACameraOutputTarget);
    fn ACaptureSessionOutputContainer_create(
        container: *mut *mut ACaptureSessionOutputContainer,
    ) -> i32;
    fn ACaptureSessionOutputContainer_free(container: *mut ACaptureSessionOutputContainer);
    fn ACaptureSessionOutput_create(
        window: *mut ANativeWindow,
        output: *mut *mut ACaptureSessionOutput,
    ) -> i32;
    fn ACaptureSessionOutput_free(output: *mut ACaptureSessionOutput);
    fn ACaptureSessionOutputContainer_add(
        container: *mut ACaptureSessionOutputContainer,
        output: *const ACaptureSessionOutput,
    ) -> i32;
    fn ACameraDevice_createCaptureSession(
        device: *mut ACameraDevice,
        outputs: *mut ACaptureSessionOutputContainer,
        callbacks: *mut ACameraCaptureSession_stateCallbacks,
        session: *mut *mut ACameraCaptureSession,
    ) -> i32;
    fn ACameraCaptureSession_setRepeatingRequest(
        session: *mut ACameraCaptureSession,
        callbacks: *mut c_void, // ACameraCaptureSession_captureCallbacks, nullable
        num_requests: c_int,
        requests: *mut *mut ACaptureRequest,
        sequence_id: *mut c_int,
    ) -> i32;
    fn ACameraCaptureSession_close(session: *mut ACameraCaptureSession);
}

#[link(name = "mediandk")]
unsafe extern "C" {
    fn AImageReader_new(
        width: i32,
        height: i32,
        format: i32,
        max_images: i32,
        reader: *mut *mut AImageReader,
    ) -> i32;
    fn AImageReader_getWindow(
        reader: *mut AImageReader,
        window: *mut *mut ANativeWindow,
    ) -> i32;
    fn AImageReader_setImageListener(
        reader: *mut AImageReader,
        listener: *mut AImageReader_ImageListener,
    ) -> i32;
    fn AImageReader_delete(reader: *mut AImageReader);
    fn AImageReader_acquireLatestImage(
        reader: *mut AImageReader,
        image: *mut *mut AImage,
    ) -> i32;
    fn AImage_getWidth(image: *const AImage, width: *mut i32) -> i32;
    fn AImage_getHeight(image: *const AImage, height: *mut i32) -> i32;
    fn AImage_getPlaneRowStride(
        image: *const AImage,
        plane_idx: c_int,
        row_stride: *mut i32,
    ) -> i32;
    fn AImage_getPlanePixelStride(
        image: *const AImage,
        plane_idx: c_int,
        pixel_stride: *mut i32,
    ) -> i32;
    fn AImage_getPlaneData(
        image: *const AImage,
        plane_idx: c_int,
        data: *mut *const u8,
        data_length: *mut i32,
    ) -> i32;
    fn AImage_delete(image: *mut AImage);
}

// ── High-level wrapper ────────────────────────────────────────────────────

/// Latest captured frame data, shared between callback thread and main thread.
struct SharedFrame {
    data: Vec<u8>,     // NV12 packed data (Y + interleaved UV)
    width: u32,
    height: u32,
    is_nv12: bool,     // true = NV12, false = RGB
    fresh: bool,
}

/// Persistent data kept alive for the camera session.
/// These must not be dropped while the session is active.
struct SessionData {
    _device_callbacks: Box<ACameraDevice_StateCallbacks>,
    _session_callbacks: Box<ACameraCaptureSession_stateCallbacks>,
    _image_listener: Box<AImageReader_ImageListener>,
}

pub struct CameraCapture {
    manager: *mut ACameraManager,
    device: *mut ACameraDevice,
    session: *mut ACameraCaptureSession,
    request: *mut ACaptureRequest,
    output_target: *mut ACameraOutputTarget,
    session_output: *mut ACaptureSessionOutput,
    output_container: *mut ACaptureSessionOutputContainer,
    reader: *mut AImageReader,
    shared: Arc<Mutex<SharedFrame>>,
    _session_data: SessionData,
    capture_width: u32,
    capture_height: u32,
}

// Safety: the raw pointers are only accessed from the thread that created them
// (except the shared Arc<Mutex<>> which is thread-safe)
unsafe impl Send for CameraCapture {}

/// Query the camera's supported YUV_420_888 output resolutions via Camera2 characteristics.
unsafe fn query_supported_yuv_resolutions(
    manager: *mut ACameraManager,
    camera_id: *const c_char,
) -> Vec<(i32, i32)> {
    let mut metadata: *mut ACameraMetadata = ptr::null_mut();
    let status = ACameraManager_getCameraCharacteristics(manager, camera_id, &mut metadata);
    if status != ACAMERA_OK || metadata.is_null() {
        alvr_common::error!("[XR_CAM] getCameraCharacteristics failed: {status}");
        return Vec::new();
    }

    let mut entry = ACameraMetadata_const_entry {
        tag: 0,
        type_: 0,
        count: 0,
        data: ptr::null(),
    };
    let status = ACameraMetadata_getConstEntry(
        metadata,
        ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS,
        &mut entry,
    );

    let mut resolutions = Vec::new();
    if status == ACAMERA_OK && !entry.data.is_null() && entry.count >= 4 {
        // Each configuration is 4 x i32: [format, width, height, input]
        let data = entry.data as *const i32;
        let num_configs = entry.count as usize / 4;
        for i in 0..num_configs {
            let format = *data.add(i * 4);
            let w = *data.add(i * 4 + 1);
            let h = *data.add(i * 4 + 2);
            let input = *data.add(i * 4 + 3);
            if format == AIMAGE_FORMAT_YUV_420_888 && input == 0 {
                resolutions.push((w, h));
            }
        }
    } else {
        alvr_common::error!("[XR_CAM] getConstEntry for STREAM_CONFIGURATIONS failed: {status}");
    }

    ACameraMetadata_free(metadata);
    resolutions
}

/// Pick the supported resolution closest in pixel count to the requested size.
/// Prefers exact match, then closest by total pixels.
fn pick_best_resolution(
    requested_w: i32,
    requested_h: i32,
    supported: &[(i32, i32)],
) -> (i32, i32) {
    if supported.is_empty() {
        return (requested_w, requested_h);
    }

    // Exact match first
    for &(w, h) in supported {
        if w == requested_w && h == requested_h {
            return (w, h);
        }
    }

    // Closest by pixel count
    let requested_pixels = (requested_w as i64) * (requested_h as i64);
    let mut best = supported[0];
    let mut best_diff = i64::MAX;

    for &(w, h) in supported {
        let pixels = (w as i64) * (h as i64);
        let diff = (pixels - requested_pixels).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (w, h);
        }
    }

    best
}

impl CameraCapture {
    /// Try to open the left passthrough camera and start capture.
    /// Returns None if camera access is unavailable.
    pub fn new(width: i32, height: i32) -> Option<Self> {
        Self::new_with_camera_id(width, height, None)
    }

    /// Open a specific camera by ID string (e.g. "50" for left, "51" for right on Quest 3).
    /// If preferred_id is None, auto-selects from preferred IDs [50, 51, 0].
    pub fn new_with_camera_id(width: i32, height: i32, preferred_id: Option<&str>) -> Option<Self> {
        // Request camera permissions at runtime (same pattern as eye/face tracking)
        #[cfg(target_os = "android")]
        {
            alvr_system_info::try_get_permission("android.permission.CAMERA");
            alvr_system_info::try_get_permission("horizonos.permission.PASSTHROUGH_CAMERA_ACCESS");
        }
        unsafe { Self::init_camera(width, height, preferred_id) }
    }

    unsafe fn init_camera(width: i32, height: i32, preferred_id: Option<&str>) -> Option<Self> {
        let manager = ACameraManager_create();
        if manager.is_null() {
            alvr_common::error!("[XR_CAM] Failed to create ACameraManager");
            return None;
        }

        // List cameras
        let mut id_list: *mut ACameraIdList = ptr::null_mut();
        let status = ACameraManager_getCameraIdList(manager, &mut id_list);
        if status != ACAMERA_OK || id_list.is_null() {
            alvr_common::error!("[XR_CAM] getCameraIdList failed: {status}");
            ACameraManager_delete(manager);
            return None;
        }

        let num = (*id_list).num_cameras;
        alvr_common::error!("[XR_CAM] Found {num} cameras");
        if num < 1 {
            ACameraManager_deleteCameraIdList(id_list);
            ACameraManager_delete(manager);
            return None;
        }

        // Log all available camera IDs
        for i in 0..num {
            let cid = *(*id_list).camera_ids.offset(i as isize);
            let cid_str = std::ffi::CStr::from_ptr(cid).to_string_lossy();
            alvr_common::error!("[XR_CAM]   Camera[{i}] id={cid_str}");
        }

        // On Quest 3, passthrough cameras have IDs 50 (left) and 51 (right).
        // Camera ID 1 is an internal sensor that produces all-zero frames.
        // If a specific camera ID is requested, use it; otherwise auto-select.
        let mut camera_id = *(*id_list).camera_ids;
        if let Some(target_id) = preferred_id {
            let mut found = false;
            for i in 0..num {
                let cid = *(*id_list).camera_ids.offset(i as isize);
                let cid_str = std::ffi::CStr::from_ptr(cid).to_string_lossy();
                if cid_str == target_id {
                    camera_id = cid;
                    found = true;
                    break;
                }
            }
            if !found {
                alvr_common::error!("[XR_CAM] Requested camera ID '{target_id}' not found");
                ACameraManager_deleteCameraIdList(id_list);
                ACameraManager_delete(manager);
                return None;
            }
        } else {
            // Auto-select: prefer passthrough cameras, falling back to first available
            let auto_ids = ["50", "51", "0"];
            for preferred in &auto_ids {
                for i in 0..num {
                    let cid = *(*id_list).camera_ids.offset(i as isize);
                    let cid_str = std::ffi::CStr::from_ptr(cid).to_string_lossy();
                    if cid_str == *preferred {
                        camera_id = cid;
                        break;
                    }
                }
                let chosen = std::ffi::CStr::from_ptr(camera_id).to_string_lossy();
                if auto_ids.contains(&chosen.as_ref()) {
                    break;
                }
            }
        }
        let camera_id_str = std::ffi::CStr::from_ptr(camera_id).to_string_lossy().to_string();
        alvr_common::error!("[XR_CAM] Opening camera: {camera_id_str}");

        // Query supported resolutions and pick the best match
        let supported = query_supported_yuv_resolutions(manager, camera_id);
        if supported.is_empty() {
            alvr_common::error!("[XR_CAM] No supported YUV resolutions found, using requested {width}x{height}");
        } else {
            alvr_common::error!("[XR_CAM] Supported YUV resolutions for camera {camera_id_str}:");
            for (w, h) in &supported {
                alvr_common::error!("[XR_CAM]   {w}x{h}");
            }
        }
        let (actual_w, actual_h) = pick_best_resolution(width, height, &supported);
        if actual_w != width || actual_h != height {
            alvr_common::error!(
                "[XR_CAM] Requested {width}x{height} not supported, using closest: {actual_w}x{actual_h}"
            );
        } else {
            alvr_common::error!("[XR_CAM] Using requested resolution: {actual_w}x{actual_h}");
        }

        // Device callbacks (must be kept alive)
        let mut device_callbacks = Box::new(ACameraDevice_StateCallbacks {
            context: ptr::null_mut(),
            on_disconnected: Some(on_disconnected),
            on_error: Some(on_error),
        });

        let mut device: *mut ACameraDevice = ptr::null_mut();
        let status = ACameraManager_openCamera(
            manager,
            camera_id,
            &mut *device_callbacks as *mut _,
            &mut device,
        );
        ACameraManager_deleteCameraIdList(id_list);

        if status != ACAMERA_OK || device.is_null() {
            alvr_common::error!("[XR_CAM] openCamera failed: {status}");
            ACameraManager_delete(manager);
            return None;
        }
        alvr_common::error!("[XR_CAM] Camera opened successfully");

        // Create AImageReader for YUV_420_888 at the actual supported resolution
        let mut reader: *mut AImageReader = ptr::null_mut();
        let status = AImageReader_new(actual_w, actual_h, AIMAGE_FORMAT_YUV_420_888, 4, &mut reader);
        if status != ACAMERA_OK || reader.is_null() {
            alvr_common::error!("[XR_CAM] AImageReader_new failed: {status}");
            ACameraDevice_close(device);
            ACameraManager_delete(manager);
            return None;
        }

        // Set up shared frame buffer
        let shared = Arc::new(Mutex::new(SharedFrame {
            data: Vec::new(),
            width: 0,
            height: 0,
            is_nv12: true,
            fresh: false,
        }));

        // Image listener callback (must be kept alive)
        let shared_for_callback = Arc::clone(&shared);
        let ctx_ptr = Arc::into_raw(shared_for_callback) as *mut c_void;
        let mut image_listener = Box::new(AImageReader_ImageListener {
            context: ctx_ptr,
            on_image_available: Some(on_image_available),
        });

        let status = AImageReader_setImageListener(reader, &mut *image_listener as *mut _);
        if status != ACAMERA_OK {
            alvr_common::error!("[XR_CAM] setImageListener failed: {status}");
            // Reconstruct the Arc to drop it properly
            let _ = Arc::from_raw(ctx_ptr as *const Mutex<SharedFrame>);
            AImageReader_delete(reader);
            ACameraDevice_close(device);
            ACameraManager_delete(manager);
            return None;
        }

        // Get ANativeWindow from reader
        let mut window: *mut ANativeWindow = ptr::null_mut();
        AImageReader_getWindow(reader, &mut window);

        // Create output target
        let mut output_target: *mut ACameraOutputTarget = ptr::null_mut();
        ACameraOutputTarget_create(window, &mut output_target);

        // Create capture request
        let mut request: *mut ACaptureRequest = ptr::null_mut();
        ACameraDevice_createCaptureRequest(device, TEMPLATE_PREVIEW, &mut request);
        ACaptureRequest_addTarget(request, output_target);

        // Create session output
        let mut session_output: *mut ACaptureSessionOutput = ptr::null_mut();
        ACaptureSessionOutput_create(window, &mut session_output);

        let mut output_container: *mut ACaptureSessionOutputContainer = ptr::null_mut();
        ACaptureSessionOutputContainer_create(&mut output_container);
        ACaptureSessionOutputContainer_add(output_container, session_output);

        // Create capture session (must keep callbacks alive)
        let mut session_callbacks = Box::new(ACameraCaptureSession_stateCallbacks {
            context: ptr::null_mut(),
            on_closed: Some(on_session_closed),
            on_ready: Some(on_session_ready),
            on_active: Some(on_session_active),
        });

        let mut session: *mut ACameraCaptureSession = ptr::null_mut();
        let status = ACameraDevice_createCaptureSession(
            device,
            output_container,
            &mut *session_callbacks as *mut _,
            &mut session,
        );
        if status != ACAMERA_OK || session.is_null() {
            alvr_common::error!("[XR_CAM] createCaptureSession failed: {status}");
            ACaptureRequest_free(request);
            ACameraOutputTarget_free(output_target);
            ACaptureSessionOutput_free(session_output);
            ACaptureSessionOutputContainer_free(output_container);
            let _ = Arc::from_raw(ctx_ptr as *const Mutex<SharedFrame>);
            AImageReader_delete(reader);
            ACameraDevice_close(device);
            ACameraManager_delete(manager);
            return None;
        }

        // Start repeating capture
        let mut seq_id: c_int = 0;
        let status = ACameraCaptureSession_setRepeatingRequest(
            session,
            ptr::null_mut(),
            1,
            &mut request as *mut *mut _,
            &mut seq_id,
        );
        if status != ACAMERA_OK {
            alvr_common::error!("[XR_CAM] setRepeatingRequest failed: {status}");
            ACameraCaptureSession_close(session);
            ACaptureRequest_free(request);
            ACameraOutputTarget_free(output_target);
            ACaptureSessionOutput_free(session_output);
            ACaptureSessionOutputContainer_free(output_container);
            let _ = Arc::from_raw(ctx_ptr as *const Mutex<SharedFrame>);
            AImageReader_delete(reader);
            ACameraDevice_close(device);
            ACameraManager_delete(manager);
            return None;
        }

        alvr_common::error!("[XR_CAM] Camera capture started: {}x{} YUV_420_888", actual_w, actual_h);

        Some(CameraCapture {
            manager,
            device,
            session,
            request,
            output_target,
            session_output,
            output_container,
            reader,
            shared,
            _session_data: SessionData {
                _device_callbacks: device_callbacks,
                _session_callbacks: session_callbacks,
                _image_listener: image_listener,
            },
            capture_width: actual_w as u32,
            capture_height: actual_h as u32,
        })
    }

    /// Get the latest camera frame as (width, height, data, is_nv12).
    /// Returns None if no new frame is available.
    pub fn get_latest_frame(&self) -> Option<(u32, u32, Vec<u8>, bool)> {
        let mut frame = self.shared.lock().ok()?;
        if !frame.fresh || frame.data.is_empty() {
            return None;
        }
        frame.fresh = false;
        Some((frame.width, frame.height, frame.data.clone(), frame.is_nv12))
    }

    pub fn width(&self) -> u32 {
        self.capture_width
    }

    pub fn height(&self) -> u32 {
        self.capture_height
    }

    /// Destroy the camera capture on a background thread to avoid blocking the
    /// render loop.  NDK calls like ACameraCaptureSession_close and
    /// ACameraDevice_close can block for a long time (or indefinitely) on some
    /// Android drivers, which freezes ALVR if done on the render thread.
    pub fn destroy_async(self) {
        std::thread::Builder::new()
            .name("cam-destroy".into())
            .spawn(move || {
                drop(self);
            })
            .ok();
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        unsafe {
            alvr_common::error!("[XR_CAM] Shutting down camera capture (drop start)");
            alvr_common::error!("[XR_CAM]   closing capture session...");
            ACameraCaptureSession_close(self.session);
            alvr_common::error!("[XR_CAM]   freeing capture request...");
            ACaptureRequest_free(self.request);
            alvr_common::error!("[XR_CAM]   freeing output target...");
            ACameraOutputTarget_free(self.output_target);
            alvr_common::error!("[XR_CAM]   freeing session output...");
            ACaptureSessionOutput_free(self.session_output);
            alvr_common::error!("[XR_CAM]   freeing output container...");
            ACaptureSessionOutputContainer_free(self.output_container);
            // The image listener context holds an Arc reference.
            // We need to reconstruct and drop it to avoid a leak.
            // The listener is invalidated when the reader is deleted.
            alvr_common::error!("[XR_CAM]   deleting image reader...");
            AImageReader_delete(self.reader);
            alvr_common::error!("[XR_CAM]   closing camera device...");
            ACameraDevice_close(self.device);
            alvr_common::error!("[XR_CAM]   deleting camera manager...");
            ACameraManager_delete(self.manager);
            alvr_common::error!("[XR_CAM] Camera capture shutdown complete");
        }
    }
}

// ── NDK Callbacks ─────────────────────────────────────────────────────────

unsafe extern "C" fn on_disconnected(_ctx: *mut c_void, _device: *mut ACameraDevice) {
    alvr_common::error!("[XR_CAM] Camera disconnected");
}

unsafe extern "C" fn on_error(_ctx: *mut c_void, _device: *mut ACameraDevice, error: c_int) {
    alvr_common::error!("[XR_CAM] Camera error: {error}");
}

unsafe extern "C" fn on_session_closed(_ctx: *mut c_void, _session: *mut ACameraCaptureSession) {
    alvr_common::error!("[XR_CAM] Capture session closed");
}

unsafe extern "C" fn on_session_ready(_ctx: *mut c_void, _session: *mut ACameraCaptureSession) {
    alvr_common::error!("[XR_CAM] Capture session ready");
}

unsafe extern "C" fn on_session_active(_ctx: *mut c_void, _session: *mut ACameraCaptureSession) {
    alvr_common::error!("[XR_CAM] Capture session active");
}

unsafe extern "C" fn on_image_available(ctx: *mut c_void, reader: *mut AImageReader) {
    if ctx.is_null() || reader.is_null() {
        return;
    }

    // Acquire latest image
    let mut image: *mut AImage = ptr::null_mut();
    let status = AImageReader_acquireLatestImage(reader, &mut image);
    if status != ACAMERA_OK || image.is_null() {
        return;
    }

    // Get dimensions
    let mut width: i32 = 0;
    let mut height: i32 = 0;
    AImage_getWidth(image, &mut width);
    AImage_getHeight(image, &mut height);

    // Get Y plane data (plane 0)
    let mut y_data: *const u8 = ptr::null();
    let mut y_len: i32 = 0;
    let mut y_stride: i32 = 0;
    AImage_getPlaneData(image, 0, &mut y_data, &mut y_len);
    AImage_getPlaneRowStride(image, 0, &mut y_stride);

    // Get U plane data (plane 1)
    let mut u_data: *const u8 = ptr::null();
    let mut u_len: i32 = 0;
    let mut u_stride: i32 = 0;
    let mut u_pixel_stride: i32 = 0;
    AImage_getPlaneData(image, 1, &mut u_data, &mut u_len);
    AImage_getPlaneRowStride(image, 1, &mut u_stride);
    AImage_getPlanePixelStride(image, 1, &mut u_pixel_stride);

    // Get V plane data (plane 2)
    let mut v_data: *const u8 = ptr::null();
    let mut v_len: i32 = 0;
    let mut v_stride: i32 = 0;
    let mut v_pixel_stride: i32 = 0;
    AImage_getPlaneData(image, 2, &mut v_data, &mut v_len);
    AImage_getPlaneRowStride(image, 2, &mut v_stride);
    AImage_getPlanePixelStride(image, 2, &mut v_pixel_stride);

    if !y_data.is_null() && y_len > 0 && width > 0 && height > 0 {
        // Log first frame stats for debugging
        static FRAME_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let fc = FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if fc < 5 || fc % 100 == 0 {
            alvr_common::error!("[XR_CAM] on_image_available: frame={fc} {width}x{height} y_stride={y_stride} u_stride={u_stride} u_pxstride={u_pixel_stride}");
        }

        let w = width as usize;
        let h = height as usize;
        let y_row_stride = y_stride as usize;
        let uv_row_stride = u_stride as usize;
        let uv_px_stride = u_pixel_stride.max(1) as usize;

        let has_uv = !u_data.is_null() && u_len > 0 && !v_data.is_null() && v_len > 0;

        // Pack YUV data into NV12 format (Y plane + interleaved UV plane)
        // NV12: width*height Y bytes, then width*height/2 interleaved UV bytes
        let nv12_size = w * h + w * (h / 2);
        let mut nv12_buf = vec![0u8; nv12_size];

        // Copy Y plane (handle row stride)
        for row in 0..h {
            let src_off = row * y_row_stride;
            let dst_off = row * w;
            for col in 0..w {
                nv12_buf[dst_off + col] = if src_off + col < y_len as usize {
                    *y_data.add(src_off + col)
                } else {
                    0
                };
            }
        }

        // Copy UV plane (interleaved NV12 format)
        let uv_offset = w * h;
        let uv_h = h / 2;
        if has_uv && uv_px_stride == 2 {
            // Already NV12-interleaved on Qualcomm (u_pixel_stride == 2)
            // U plane pointer actually points to interleaved UVUV... data
            for row in 0..uv_h {
                let src_off = row * uv_row_stride;
                let dst_off = uv_offset + row * w;
                for col in 0..w {
                    // Interleaved UV: byte at offset col in the UV row
                    let src_idx = src_off + col;
                    nv12_buf[dst_off + col] = if src_idx < u_len as usize {
                        *u_data.add(src_idx)
                    } else {
                        128
                    };
                }
            }
        } else if has_uv {
            // Planar YUV420 (u_pixel_stride == 1): interleave U and V manually
            for row in 0..uv_h {
                let uv_src_off = row * uv_row_stride;
                let dst_off = uv_offset + row * w;
                for col in 0..(w / 2) {
                    let u_idx = uv_src_off + col * uv_px_stride;
                    let v_idx = uv_src_off + col * uv_px_stride;
                    nv12_buf[dst_off + col * 2] = if u_idx < u_len as usize {
                        *u_data.add(u_idx)
                    } else {
                        128
                    };
                    nv12_buf[dst_off + col * 2 + 1] = if v_idx < v_len as usize {
                        *v_data.add(v_idx)
                    } else {
                        128
                    };
                }
            }
        } else {
            // No UV data: fill with 128 (neutral gray)
            nv12_buf[uv_offset..].fill(128);
        }

        // Store in shared frame
        let shared_ptr = ctx as *const Mutex<SharedFrame>;
        if let Ok(mut frame) = (*shared_ptr).lock() {
            frame.data = nv12_buf;
            frame.width = width as u32;
            frame.height = height as u32;
            frame.is_nv12 = true;
            frame.fresh = true;
        }
    }

    AImage_delete(image);
}
