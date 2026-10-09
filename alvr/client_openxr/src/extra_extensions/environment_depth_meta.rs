use crate::extra_extensions::{get_instance_proc, xr_res};
use openxr::{self as xr, sys};
use std::{ffi::c_void, ptr, sync::LazyLock};

#[allow(dead_code)]
pub const META_ENVIRONMENT_DEPTH_EXTENSION_NAME: &str = "XR_META_environment_depth";

// Structure types for XR_META_environment_depth (extension number 292, base 1000291000)
static TYPE_ENVIRONMENT_DEPTH_PROVIDER_CREATE_INFO_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291000));
static TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_CREATE_INFO_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291001));
static TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_STATE_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291002));
static TYPE_ENVIRONMENT_DEPTH_IMAGE_ACQUIRE_INFO_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291003));
static TYPE_ENVIRONMENT_DEPTH_IMAGE_VIEW_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291004));
static TYPE_ENVIRONMENT_DEPTH_IMAGE_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291005));
static TYPE_ENVIRONMENT_DEPTH_HAND_REMOVAL_SET_INFO_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291006));
static TYPE_SYSTEM_ENVIRONMENT_DEPTH_PROPERTIES_META: LazyLock<xr::StructureType> =
    LazyLock::new(|| xr::StructureType::from_raw(1000291007));

// Hand removal on the depth provider. Build with `false` to A/B its cost (Client System latency,
// GPU load) against the extra geometry hands leave in the depth for roomd.
const DEPTH_HAND_REMOVAL: bool = true;

// XR_ENVIRONMENT_DEPTH_NOT_AVAILABLE_META = 1000291000
const ENVIRONMENT_DEPTH_NOT_AVAILABLE_META: i32 = 1000291000;

// Opaque handles (represented as u64)
type XrEnvironmentDepthProviderMETA = u64;
type XrEnvironmentDepthSwapchainMETA = u64;

#[repr(C)]
struct XrSystemEnvironmentDepthPropertiesMETA {
    ty: xr::StructureType,
    next: *mut c_void,
    supports_environment_depth: sys::Bool32,
    supports_hand_removal: sys::Bool32,
}

#[repr(C)]
struct XrEnvironmentDepthProviderCreateInfoMETA {
    ty: xr::StructureType,
    next: *const c_void,
    create_flags: u64,
}

#[repr(C)]
struct XrEnvironmentDepthHandRemovalSetInfoMETA {
    ty: xr::StructureType,
    next: *const c_void,
    enabled: sys::Bool32,
}

#[repr(C)]
struct XrEnvironmentDepthSwapchainCreateInfoMETA {
    ty: xr::StructureType,
    next: *const c_void,
    create_flags: u64,
}

#[repr(C)]
pub struct XrEnvironmentDepthSwapchainStateMETA {
    ty: xr::StructureType,
    next: *mut c_void,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
struct XrEnvironmentDepthImageAcquireInfoMETA {
    ty: xr::StructureType,
    next: *const c_void,
    space: sys::Space,
    display_time: sys::Time,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XrEnvironmentDepthImageViewMETA {
    ty: xr::StructureType,
    next: *const c_void,
    pub fov: sys::Fovf,
    pub pose: sys::Posef,
}

#[repr(C)]
pub struct XrEnvironmentDepthImageMETA {
    ty: xr::StructureType,
    next: *const c_void,
    pub swapchain_index: u32,
    pub near_z: f32,
    pub far_z: f32,
    pub views: [XrEnvironmentDepthImageViewMETA; 2],
}

// Function pointer types
type CreateEnvironmentDepthProviderMETA = unsafe extern "system" fn(
    sys::Session,
    *const XrEnvironmentDepthProviderCreateInfoMETA,
    *mut XrEnvironmentDepthProviderMETA,
) -> sys::Result;

type DestroyEnvironmentDepthProviderMETA =
    unsafe extern "system" fn(XrEnvironmentDepthProviderMETA) -> sys::Result;

type StartEnvironmentDepthProviderMETA =
    unsafe extern "system" fn(XrEnvironmentDepthProviderMETA) -> sys::Result;

type StopEnvironmentDepthProviderMETA =
    unsafe extern "system" fn(XrEnvironmentDepthProviderMETA) -> sys::Result;

type SetEnvironmentDepthHandRemovalMETA = unsafe extern "system" fn(
    XrEnvironmentDepthProviderMETA,
    *const XrEnvironmentDepthHandRemovalSetInfoMETA,
) -> sys::Result;

type CreateEnvironmentDepthSwapchainMETA = unsafe extern "system" fn(
    XrEnvironmentDepthProviderMETA,
    *const XrEnvironmentDepthSwapchainCreateInfoMETA,
    *mut XrEnvironmentDepthSwapchainMETA,
) -> sys::Result;

type DestroyEnvironmentDepthSwapchainMETA =
    unsafe extern "system" fn(XrEnvironmentDepthSwapchainMETA) -> sys::Result;

type GetEnvironmentDepthSwapchainStateMETA = unsafe extern "system" fn(
    XrEnvironmentDepthSwapchainMETA,
    *mut XrEnvironmentDepthSwapchainStateMETA,
) -> sys::Result;

type EnumerateEnvironmentDepthSwapchainImagesMETA = unsafe extern "system" fn(
    XrEnvironmentDepthSwapchainMETA,
    u32,
    *mut u32,
    *mut sys::SwapchainImageBaseHeader,
) -> sys::Result;

type AcquireEnvironmentDepthImageMETA = unsafe extern "system" fn(
    XrEnvironmentDepthProviderMETA,
    *const XrEnvironmentDepthImageAcquireInfoMETA,
    *mut XrEnvironmentDepthImageMETA,
) -> sys::Result;

#[allow(dead_code)]
pub struct EnvironmentDepthMeta {
    session: xr::Session<xr::AnyGraphics>,
    provider: XrEnvironmentDepthProviderMETA,
    swapchain: XrEnvironmentDepthSwapchainMETA,
    pub swapchain_width: u32,
    pub swapchain_height: u32,
    pub swapchain_images: Vec<u32>, // OpenGL ES texture names
    started: bool,

    // Function pointers
    destroy_provider: DestroyEnvironmentDepthProviderMETA,
    start_provider: StartEnvironmentDepthProviderMETA,
    stop_provider: StopEnvironmentDepthProviderMETA,
    destroy_swapchain: DestroyEnvironmentDepthSwapchainMETA,
    get_swapchain_state: GetEnvironmentDepthSwapchainStateMETA,
    enumerate_swapchain_images: EnumerateEnvironmentDepthSwapchainImagesMETA,
    acquire_image: AcquireEnvironmentDepthImageMETA,
}

impl EnvironmentDepthMeta {
    pub fn new<G>(
        session: xr::Session<G>,
        system: xr::SystemId,
    ) -> xr::Result<Self> {
        // Check that the extension was enabled on the instance
        if !session.instance().exts().meta_environment_depth.is_some() {
            return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
        }

        // Check system support
        let props_result = super::get_props(
            &session,
            system,
            XrSystemEnvironmentDepthPropertiesMETA {
                ty: *TYPE_SYSTEM_ENVIRONMENT_DEPTH_PROPERTIES_META,
                next: ptr::null_mut(),
                supports_environment_depth: sys::FALSE,
                supports_hand_removal: sys::FALSE,
            },
        );
        match &props_result {
            Ok(props) => {
                alvr_common::error!(
                    "[XR_DIAG] Depth system props: supports_depth={}, supports_hand_removal={}, ty={}",
                    props.supports_environment_depth.into_raw(),
                    props.supports_hand_removal.into_raw(),
                    props.ty.into_raw(),
                );
            }
            Err(e) => {
                alvr_common::error!("[XR_DIAG] get_props for depth failed: {e:?}, trying to create provider anyway");
            }
        }

        // Try to create provider even if props say unsupported (diagnostic)
        let supports_hand_removal = props_result
            .as_ref()
            .is_ok_and(|p| bool::from(p.supports_hand_removal));
        let depth_supported = props_result
            .map(|p| bool::from(p.supports_environment_depth))
            .unwrap_or(false);
        if !depth_supported {
            alvr_common::error!("[XR_DIAG] System reports depth unsupported, attempting provider creation anyway...");
        }

        // Load function pointers
        alvr_common::error!("[XR_DIAG] Depth init step 1: loading function pointers...");
        let create_provider: CreateEnvironmentDepthProviderMETA =
            get_instance_proc(&session, "xrCreateEnvironmentDepthProviderMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrCreateEnvironmentDepthProviderMETA: {e:?}"); e })?;
        let destroy_provider: DestroyEnvironmentDepthProviderMETA =
            get_instance_proc(&session, "xrDestroyEnvironmentDepthProviderMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrDestroyEnvironmentDepthProviderMETA: {e:?}"); e })?;
        let start_provider: StartEnvironmentDepthProviderMETA =
            get_instance_proc(&session, "xrStartEnvironmentDepthProviderMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrStartEnvironmentDepthProviderMETA: {e:?}"); e })?;
        let stop_provider: StopEnvironmentDepthProviderMETA =
            get_instance_proc(&session, "xrStopEnvironmentDepthProviderMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrStopEnvironmentDepthProviderMETA: {e:?}"); e })?;
        let create_swapchain: CreateEnvironmentDepthSwapchainMETA =
            get_instance_proc(&session, "xrCreateEnvironmentDepthSwapchainMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrCreateEnvironmentDepthSwapchainMETA: {e:?}"); e })?;
        let destroy_swapchain: DestroyEnvironmentDepthSwapchainMETA =
            get_instance_proc(&session, "xrDestroyEnvironmentDepthSwapchainMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrDestroyEnvironmentDepthSwapchainMETA: {e:?}"); e })?;
        let get_swapchain_state: GetEnvironmentDepthSwapchainStateMETA =
            get_instance_proc(&session, "xrGetEnvironmentDepthSwapchainStateMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrGetEnvironmentDepthSwapchainStateMETA: {e:?}"); e })?;
        let enumerate_swapchain_images: EnumerateEnvironmentDepthSwapchainImagesMETA =
            get_instance_proc(&session, "xrEnumerateEnvironmentDepthSwapchainImagesMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrEnumerateEnvironmentDepthSwapchainImagesMETA: {e:?}"); e })?;
        let acquire_image: AcquireEnvironmentDepthImageMETA =
            get_instance_proc(&session, "xrAcquireEnvironmentDepthImageMETA").map_err(|e| { alvr_common::error!("[XR_DIAG] Failed to load xrAcquireEnvironmentDepthImageMETA: {e:?}"); e })?;
        alvr_common::error!("[XR_DIAG] Depth init step 1: all function pointers loaded OK");

        // Create provider
        alvr_common::error!("[XR_DIAG] Depth init step 2: creating provider...");
        let create_info = XrEnvironmentDepthProviderCreateInfoMETA {
            ty: *TYPE_ENVIRONMENT_DEPTH_PROVIDER_CREATE_INFO_META,
            next: ptr::null(),
            create_flags: 0,
        };
        let mut provider: XrEnvironmentDepthProviderMETA = 0;
        unsafe {
            xr_res(create_provider(
                session.as_raw(),
                &create_info,
                &mut provider,
            )).map_err(|e| { alvr_common::error!("[XR_DIAG] Depth init step 2 FAILED: xrCreateEnvironmentDepthProviderMETA returned {e:?}"); e })?;
        }
        alvr_common::error!("[XR_DIAG] Depth init step 2: provider created OK (handle={provider})");

        // Hands in front of the headset would otherwise show up as room geometry
        if supports_hand_removal && DEPTH_HAND_REMOVAL {
            match get_instance_proc::<_, SetEnvironmentDepthHandRemovalMETA>(
                &session,
                "xrSetEnvironmentDepthHandRemovalMETA",
            ) {
                Ok(set_hand_removal) => {
                    let info = XrEnvironmentDepthHandRemovalSetInfoMETA {
                        ty: *TYPE_ENVIRONMENT_DEPTH_HAND_REMOVAL_SET_INFO_META,
                        next: ptr::null(),
                        enabled: sys::TRUE,
                    };
                    let result = unsafe { set_hand_removal(provider, &info) };
                    alvr_common::info!("[XR_DIAG] Depth hand removal enabled: {result:?}");
                }
                Err(e) => alvr_common::error!(
                    "[XR_DIAG] Failed to load xrSetEnvironmentDepthHandRemovalMETA: {e:?}"
                ),
            }
        } else if supports_hand_removal {
            alvr_common::info!("[XR_DIAG] Depth hand removal disabled at build time");
        } else {
            alvr_common::info!("[XR_DIAG] Depth hand removal not supported");
        }

        // Create swapchain
        alvr_common::error!("[XR_DIAG] Depth init step 3: creating swapchain...");
        let swapchain_create_info = XrEnvironmentDepthSwapchainCreateInfoMETA {
            ty: *TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_CREATE_INFO_META,
            next: ptr::null(),
            create_flags: 0,
        };
        let mut swapchain: XrEnvironmentDepthSwapchainMETA = 0;
        unsafe {
            xr_res(create_swapchain(
                provider,
                &swapchain_create_info,
                &mut swapchain,
            )).map_err(|e| { alvr_common::error!("[XR_DIAG] Depth init step 3 FAILED: xrCreateEnvironmentDepthSwapchainMETA returned {e:?}"); e })?;
        }
        alvr_common::error!("[XR_DIAG] Depth init step 3: swapchain created OK");

        // Get swapchain state (width, height)
        let mut state = XrEnvironmentDepthSwapchainStateMETA {
            ty: *TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_STATE_META,
            next: ptr::null_mut(),
            width: 0,
            height: 0,
        };
        unsafe {
            xr_res(get_swapchain_state(swapchain, &mut state))?;
        }

        // Enumerate swapchain images (OpenGL ES)
        let mut image_count = 0u32;
        unsafe {
            xr_res(enumerate_swapchain_images(
                swapchain,
                0,
                &mut image_count,
                ptr::null_mut(),
            ))?;
        }

        // OpenGL ES swapchain images
        #[repr(C)]
        struct XrSwapchainImageOpenGLESKHR {
            ty: xr::StructureType,
            next: *mut c_void,
            image: u32,
        }

        let mut images: Vec<XrSwapchainImageOpenGLESKHR> = (0..image_count)
            .map(|_| XrSwapchainImageOpenGLESKHR {
                ty: xr::StructureType::SWAPCHAIN_IMAGE_OPENGL_ES_KHR,
                next: ptr::null_mut(),
                image: 0,
            })
            .collect();

        unsafe {
            xr_res(enumerate_swapchain_images(
                swapchain,
                image_count,
                &mut image_count,
                images.as_mut_ptr() as *mut sys::SwapchainImageBaseHeader,
            ))?;
        }

        // Dump raw bytes for struct layout diagnosis
        alvr_common::error!("[XR_DIAG] Swapchain image struct size={}, count={}, ty_val={:?}",
            std::mem::size_of::<XrSwapchainImageOpenGLESKHR>(),
            image_count,
            xr::StructureType::SWAPCHAIN_IMAGE_OPENGL_ES_KHR);
        if !images.is_empty() {
            let raw_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    images.as_ptr() as *const u8,
                    std::mem::size_of::<XrSwapchainImageOpenGLESKHR>() * images.len().min(2),
                )
            };
            alvr_common::error!("[XR_DIAG] Raw swapchain image bytes: {:02X?}", raw_bytes);
        }
        let swapchain_images: Vec<u32> = images.iter().map(|img| img.image).collect();
        alvr_common::error!("[XR_DIAG] Depth swapchain images after create: {:?}", swapchain_images);
        alvr_common::error!(
            "[XR_DIAG] Depth resolution: {}x{} (fixed by hardware, not configurable)",
            state.width, state.height
        );

        Ok(Self {
            session: session.into_any_graphics(),
            provider,
            swapchain,
            swapchain_width: state.width,
            swapchain_height: state.height,
            swapchain_images,
            started: false,
            destroy_provider,
            start_provider,
            stop_provider,
            destroy_swapchain,
            get_swapchain_state,
            enumerate_swapchain_images,
            acquire_image,
        })
    }

    pub fn start(&mut self) -> xr::Result<()> {
        if !self.started {
            unsafe {
                xr_res((self.start_provider)(self.provider))?;
            }
            self.started = true;

            // Re-enumerate swapchain images after start — runtime may only
            // allocate GL textures once the provider is running
            if let Err(e) = self.re_enumerate_images() {
                alvr_common::error!("[XR_DIAG] Failed to re-enumerate depth images after start: {e:?}");
            }
        }
        Ok(())
    }

    /// Re-enumerate swapchain images (call after start if initial IDs were 0)
    fn re_enumerate_images(&mut self) -> xr::Result<()> {
        let mut image_count = 0u32;
        unsafe {
            xr_res((self.enumerate_swapchain_images)(
                self.swapchain, 0, &mut image_count, ptr::null_mut(),
            ))?;
        }

        #[repr(C)]
        struct XrSwapchainImageOpenGLESKHR {
            ty: xr::StructureType,
            next: *mut c_void,
            image: u32,
        }

        let mut images: Vec<XrSwapchainImageOpenGLESKHR> = (0..image_count)
            .map(|_| XrSwapchainImageOpenGLESKHR {
                ty: xr::StructureType::SWAPCHAIN_IMAGE_OPENGL_ES_KHR,
                next: ptr::null_mut(),
                image: 0,
            })
            .collect();

        unsafe {
            xr_res((self.enumerate_swapchain_images)(
                self.swapchain,
                image_count,
                &mut image_count,
                images.as_mut_ptr() as *mut sys::SwapchainImageBaseHeader,
            ))?;
        }

        self.swapchain_images = images.iter().map(|img| img.image).collect();
        alvr_common::error!("[XR_DIAG] Depth swapchain images after start: {:?}", self.swapchain_images);
        Ok(())
    }

    pub fn stop(&mut self) -> xr::Result<()> {
        if self.started {
            unsafe {
                xr_res((self.stop_provider)(self.provider))?;
            }
            self.started = false;
        }
        Ok(())
    }

    /// Acquire the latest depth image. Returns None if depth is not yet available.
    /// Must be called between xrBeginFrame and xrEndFrame.
    pub fn acquire_depth_image(
        &self,
        space: sys::Space,
        display_time: sys::Time,
    ) -> xr::Result<Option<XrEnvironmentDepthImageMETA>> {
        let acquire_info = XrEnvironmentDepthImageAcquireInfoMETA {
            ty: *TYPE_ENVIRONMENT_DEPTH_IMAGE_ACQUIRE_INFO_META,
            next: ptr::null(),
            space,
            display_time,
        };

        let default_view = XrEnvironmentDepthImageViewMETA {
            ty: *TYPE_ENVIRONMENT_DEPTH_IMAGE_VIEW_META,
            next: ptr::null(),
            fov: sys::Fovf {
                angle_left: 0.0,
                angle_right: 0.0,
                angle_up: 0.0,
                angle_down: 0.0,
            },
            pose: sys::Posef {
                orientation: sys::Quaternionf {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                position: sys::Vector3f {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
        };

        let mut image = XrEnvironmentDepthImageMETA {
            ty: *TYPE_ENVIRONMENT_DEPTH_IMAGE_META,
            next: ptr::null(),
            swapchain_index: 0,
            near_z: 0.0,
            far_z: 0.0,
            views: [default_view; 2],
        };

        let result = unsafe {
            (self.acquire_image)(self.provider, &acquire_info, &mut image)
        };

        if result.into_raw() == ENVIRONMENT_DEPTH_NOT_AVAILABLE_META {
            return Ok(None);
        }

        xr_res(result)?;
        Ok(Some(image))
    }

    pub fn is_started(&self) -> bool {
        self.started
    }
}

impl Drop for EnvironmentDepthMeta {
    fn drop(&mut self) {
        if self.started {
            unsafe {
                (self.stop_provider)(self.provider);
            }
        }
        unsafe {
            (self.destroy_swapchain)(self.swapchain);
            (self.destroy_provider)(self.provider);
        }
    }
}
