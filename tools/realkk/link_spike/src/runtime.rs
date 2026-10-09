//! Minimal OpenXR runtime loading without the Khronos loader: read the runtime manifest JSON,
//! load its library and negotiate xrGetInstanceProcAddr. API layers are not loaded.

use openxr::{self as xr, sys};
use serde_json::{Value, json};
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
};

/// Default install location of the Meta Quest Link (Oculus) PC runtime manifest.
pub const OCULUS_RUNTIME_JSON: &str =
    r"C:\Program Files\Oculus\Support\oculus-runtime\oculus_openxr_64.json";

pub struct RuntimeSelection {
    pub json_path: PathBuf,
    /// "--runtime-json", "XR_RUNTIME_JSON" or "registry ActiveRuntime"
    pub source: &'static str,
}

/// Same precedence as the Khronos loader: XR_RUNTIME_JSON, then the registry ActiveRuntime.
pub fn select_runtime(cli: Option<&Path>) -> Result<RuntimeSelection, String> {
    if let Some(p) = cli {
        return Ok(RuntimeSelection { json_path: p.to_path_buf(), source: "--runtime-json" });
    }
    if let Some(p) = std::env::var_os("XR_RUNTIME_JSON").filter(|v| !v.is_empty()) {
        return Ok(RuntimeSelection { json_path: PathBuf::from(p), source: "XR_RUNTIME_JSON" });
    }
    match active_runtime_from_registry() {
        Some(p) => Ok(RuntimeSelection { json_path: p, source: "registry ActiveRuntime" }),
        None => Err("no XR_RUNTIME_JSON and no HKLM\\SOFTWARE\\Khronos\\OpenXR\\1\\ActiveRuntime".into()),
    }
}

/// Where a Meta Quest Link install would put its runtime manifest, and whether it exists.
pub fn oculus_runtime_candidates() -> Value {
    let mut candidates = vec![PathBuf::from(OCULUS_RUNTIME_JSON)];
    if let Some(base) = std::env::var_os("OculusBase") {
        candidates.push(Path::new(&base).join(r"Support\oculus-runtime\oculus_openxr_64.json"));
    }
    let list: Vec<Value> = candidates
        .iter()
        .map(|p| json!({"path": p.display().to_string(), "exists": p.is_file()}))
        .collect();
    json!({
        "candidates": list,
        "found": candidates.iter().any(|p| p.is_file()),
        "OculusBase_env": std::env::var("OculusBase").ok(),
        "OVRService_dir_exists": Path::new(r"C:\Program Files\Oculus\Support").is_dir(),
    })
}

#[cfg(windows)]
fn active_runtime_from_registry() -> Option<PathBuf> {
    use windows::{
        Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW},
        core::w,
    };
    let mut buf = [0u16; 1024];
    let mut size = (buf.len() * 2) as u32;
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            w!("SOFTWARE\\Khronos\\OpenXR\\1"),
            w!("ActiveRuntime"),
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr() as *mut c_void),
            Some(&mut size),
        )
    };
    if status.is_err() {
        return None;
    }
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    Some(PathBuf::from(String::from_utf16_lossy(&buf[..len])))
}

#[cfg(not(windows))]
fn active_runtime_from_registry() -> Option<PathBuf> {
    None
}

pub struct LoadedRuntime {
    pub entry: xr::Entry,
    pub manifest: Value,
    pub library_path: PathBuf,
    pub negotiation: Value,
    /// Never unloaded: runtimes may keep threads alive after xrDestroyInstance.
    _lib: std::mem::ManuallyDrop<libloading::Library>,
}

/// Like the Khronos loader: plain LoadLibrary first, then again with the runtime's own
/// directory on the dependency search path (runtime DLLs often ship their dependencies there).
#[cfg(windows)]
fn load_library(path: &Path) -> Result<libloading::Library, libloading::Error> {
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library,
    };
    match unsafe { libloading::Library::new(path) } {
        Ok(lib) => Ok(lib),
        Err(first) => unsafe {
            Library::load_with_flags(path, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
                .map(Into::into)
                .map_err(|_| first)
        },
    }
}

#[cfg(not(windows))]
fn load_library(path: &Path) -> Result<libloading::Library, libloading::Error> {
    unsafe { libloading::Library::new(path) }
}

type NegotiateFn = unsafe extern "system" fn(
    *const sys::NegotiateLoaderInfo,
    *mut sys::NegotiateRuntimeRequest,
) -> sys::Result;

pub fn load_runtime(json_path: &Path) -> Result<LoadedRuntime, String> {
    let text = std::fs::read_to_string(json_path)
        .map_err(|e| format!("cannot read runtime manifest {}: {e}", json_path.display()))?;
    let manifest: Value =
        serde_json::from_str(&text).map_err(|e| format!("runtime manifest is not JSON: {e}"))?;
    let lib_rel = manifest["runtime"]["library_path"]
        .as_str()
        .ok_or("runtime manifest has no runtime.library_path")?;
    let library_path = if Path::new(lib_rel).is_absolute() {
        PathBuf::from(lib_rel)
    } else {
        json_path.parent().unwrap_or(Path::new(".")).join(lib_rel)
    };

    let lib = load_library(&library_path)
        .map_err(|e| format!("cannot load runtime library {}: {e}", library_path.display()))?;

    let (gipa, negotiation) = unsafe { negotiate(&lib)? };
    let entry = unsafe { xr::Entry::from_get_instance_proc_addr(gipa) }
        .map_err(|e| format!("Entry::from_get_instance_proc_addr failed: {e:?}"))?;

    Ok(LoadedRuntime {
        entry,
        manifest,
        library_path,
        negotiation,
        _lib: std::mem::ManuallyDrop::new(lib),
    })
}

unsafe fn negotiate(
    lib: &libloading::Library,
) -> Result<(sys::pfn::GetInstanceProcAddr, Value), String> {
    unsafe {
        if let Ok(negotiate) = lib.get::<NegotiateFn>(b"xrNegotiateLoaderRuntimeInterface\0") {
            let info = sys::NegotiateLoaderInfo {
                struct_type: sys::LoaderInterfaceStructs::from_raw(1), // LOADER_INFO
                struct_version: 1,
                struct_size: std::mem::size_of::<sys::NegotiateLoaderInfo>(),
                min_interface_version: 1,
                max_interface_version: 1,
                min_api_version: sys::Version::new(1, 0, 0),
                max_api_version: sys::Version::new(1, 0x3ff, 0xfff),
            };
            let mut request = sys::NegotiateRuntimeRequest {
                struct_type: sys::LoaderInterfaceStructs::from_raw(3), // RUNTIME_REQUEST
                struct_version: 1,
                struct_size: std::mem::size_of::<sys::NegotiateRuntimeRequest>(),
                runtime_interface_version: 0,
                runtime_api_version: sys::Version::from_raw(0),
                get_instance_proc_addr: None,
            };
            let r = negotiate(&info, &mut request);
            if r.into_raw() < 0 {
                return Err(format!("xrNegotiateLoaderRuntimeInterface failed: {r:?} ({})", r.into_raw()));
            }
            let gipa = request
                .get_instance_proc_addr
                .ok_or("negotiation returned no xrGetInstanceProcAddr")?;
            let v = request.runtime_api_version;
            let detail = serde_json::json!({
                "method": "xrNegotiateLoaderRuntimeInterface",
                "runtime_interface_version": request.runtime_interface_version,
                "runtime_api_version": format!("{}.{}.{}", v.major(), v.minor(), v.patch()),
            });
            return Ok((gipa, detail));
        }
        // Pre-negotiation runtimes export xrGetInstanceProcAddr directly.
        let gipa = lib
            .get::<sys::pfn::GetInstanceProcAddr>(b"xrGetInstanceProcAddr\0")
            .map_err(|e| format!("runtime exports neither negotiate nor xrGetInstanceProcAddr: {e}"))?;
        Ok((*gipa, serde_json::json!({"method": "exported xrGetInstanceProcAddr"})))
    }
}
