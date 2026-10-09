//! Quest Link developer-mode feasibility spike: can a native PC OpenXR app read
//! XR_META_environment_depth frames and XR_FB_scene data through the Meta Quest Link runtime?
//!
//! Every step is recorded in OUT/report.json with its XrResult; a failing step is logged and the
//! run continues with whatever does not depend on it. See README.md for the PM test procedure.

use link_spike::{
    d3d::D3d,
    depth, runtime, scene,
    tap::{DepthFrame, FORMAT_RAW_D16, MSG_DEPTH_FRAME_V2, TapWriter, encode_depth_frame_v2},
    xrctx::{XrCtx, unix_now_ns, xr_result_json},
};
use openxr::{self as xr, sys};
use serde_json::{Map, Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const VERSION: &str = concat!("link_spike ", env!("CARGO_PKG_VERSION"));
const MIN_FPS: f64 = 9.0;

/// Extensions the spike enables when the runtime offers them.
const WANTED: &[&str] = &[
    "XR_KHR_D3D11_enable",
    "XR_MND_headless",
    "XR_KHR_win32_convert_performance_counter_time",
    "XR_META_environment_depth",
    "XR_FB_scene",
    "XR_FB_spatial_entity",
    "XR_FB_spatial_entity_query",
    "XR_FB_spatial_entity_storage",
    "XR_FB_spatial_entity_container",
    "XR_META_spatial_entity_mesh",
    "XR_FB_passthrough",
];

struct Args {
    out: PathBuf,
    seconds: f64,
    runtime_json: Option<PathBuf>,
    depth: bool,
    scene: bool,
    list_only: bool,
    synth_tap: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        out: PathBuf::from("out").join(stamp()),
        seconds: 10.0,
        runtime_json: None,
        depth: true,
        scene: true,
        list_only: false,
        synth_tap: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--out" => a.out = PathBuf::from(value("--out")?),
            "--seconds" => a.seconds = value("--seconds")?.parse().map_err(|e| format!("--seconds: {e}"))?,
            "--runtime-json" => a.runtime_json = Some(PathBuf::from(value("--runtime-json")?)),
            "--no-depth" => a.depth = false,
            "--no-scene" => a.scene = false,
            "--list-only" => a.list_only = true,
            "--synth-tap" => a.synth_tap = Some(PathBuf::from(value("--synth-tap")?)),
            "-h" | "--help" => {
                println!(
                    "usage: link_spike [--out DIR] [--seconds 10] [--runtime-json PATH] \
                     [--no-depth] [--no-scene] [--list-only] [--synth-tap FILE.rktap]\n\
                     Runtime: --runtime-json, else XR_RUNTIME_JSON, else the registry ActiveRuntime.\n\
                     --list-only stops after listing extensions (no instance, no session)."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

struct Report {
    out: PathBuf,
    root: Map<String, Value>,
    steps: Vec<Value>,
}

impl Report {
    fn step(&mut self, name: &str, ok: bool, detail: Value) {
        println!("[{}] {name}: {}", if ok { " OK " } else { "FAIL" }, compact(&detail));
        self.steps.push(json!({"step": name, "ok": ok, "detail": detail}));
        self.save();
    }

    fn set(&mut self, key: &str, v: Value) {
        self.root.insert(key.into(), v);
        self.save();
    }

    fn save(&self) {
        let mut root = self.root.clone();
        root.insert("steps".into(), json!(self.steps));
        write_json(&self.out.join("report.json"), &Value::Object(root));
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Some(path) = &args.synth_tap {
        write_synth_tap(path);
        return;
    }
    std::fs::create_dir_all(&args.out).expect("cannot create output directory");
    println!("{VERSION}: writing to {}", args.out.display());

    let mut report = Report { out: args.out.clone(), root: Map::new(), steps: Vec::new() };
    report.set("tool", json!(VERSION));
    report.set("started_utc", json!(utc_iso(unix_now_ns())));
    report.set("env_XR_RUNTIME_JSON", json!(std::env::var("XR_RUNTIME_JSON").ok()));
    report.set("oculus_runtime", runtime::oculus_runtime_candidates());
    report.set("args", json!({
        "seconds": args.seconds, "depth": args.depth, "scene": args.scene, "list_only": args.list_only,
    }));

    run(&args, &mut report);

    report.set("verdict", verdict(&report));
    report.set("finished_utc", json!(utc_iso(unix_now_ns())));
    println!("report: {}", args.out.join("report.json").display());
}

fn run(args: &Args, report: &mut Report) {
    // 1. Pick and load the runtime
    let sel = match runtime::select_runtime(args.runtime_json.as_deref()) {
        Ok(s) => {
            report.step("select_runtime", true, json!({"json": s.json_path.display().to_string(), "source": s.source}));
            s
        }
        Err(e) => return report.step("select_runtime", false, json!(e)),
    };
    let rt = match runtime::load_runtime(&sel.json_path) {
        Ok(rt) => {
            report.step("load_runtime", true, json!({
                "library": rt.library_path.display().to_string(),
                "manifest_name": rt.manifest["runtime"]["name"],
                "negotiation": rt.negotiation,
            }));
            rt
        }
        Err(e) => return report.step("load_runtime", false, json!(e)),
    };

    // 2. Extensions
    let all = match enumerate_extensions(&rt.entry) {
        Ok(list) => list,
        Err(r) => return report.step("enumerate_extensions", false, xr_result_json(r)),
    };
    let present: Map<String, Value> = WANTED
        .iter()
        .map(|w| (w.to_string(), json!(all.iter().any(|(n, _)| n == w))))
        .collect();
    report.step("enumerate_extensions", true, json!({"count": all.len(), "wanted_present": present}));
    let mut ext_doc = json!({
        "runtime_json": sel.json_path.display().to_string(),
        "runtime_json_source": sel.source,
        "runtime_library": rt.library_path.display().to_string(),
        "manifest_name": rt.manifest["runtime"]["name"],
        "extensions": all.iter().map(|(n, v)| json!({"name": n, "version": v})).collect::<Vec<_>>(),
        "wanted_present": present,
    });
    let ext_path = args.out.join("extensions.json");
    write_json(&ext_path, &ext_doc);
    if args.list_only {
        return;
    }

    // 3. Instance
    let available = match rt.entry.enumerate_extensions() {
        Ok(a) => a,
        Err(r) => return report.step("enumerate_extensions(typed)", false, xr_result_json(r)),
    };
    let enabled = wanted_subset(&available);
    let app = xr::ApplicationInfo {
        application_name: "realkk link_spike",
        application_version: 1,
        engine_name: "none",
        engine_version: 0,
        api_version: xr::Version::new(1, 0, 0),
    };
    let instance = match rt.entry.create_instance(&app, &enabled, &[]) {
        Ok(i) => i,
        Err(r) => return report.step("create_instance", false, xr_result_json(r)),
    };
    match instance.properties() {
        Ok(p) => {
            let v = p.runtime_version;
            let props = json!({"runtime_name": p.runtime_name,
                               "runtime_version": format!("{}.{}.{}", v.major(), v.minor(), v.patch())});
            ext_doc["instance"] = props.clone();
            write_json(&ext_path, &ext_doc);
            report.step("create_instance", true, props);
        }
        Err(r) => report.step("instance_properties", false, xr_result_json(r)),
    }

    // 4. System
    let system = match instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY) {
        Ok(s) => s,
        Err(r) => return report.step("get_system(HMD)", false, xr_result_json(r)),
    };
    report.step("get_system(HMD)", true, system_props(&instance, system));

    // 5. Session (D3D11 when possible: depth pixels can only be read through a graphics API)
    let mut d3d = None;
    let session = if instance.exts().khr_d3d11_enable.is_some() {
        match create_d3d11_session(&instance, system) {
            Ok((dev, session, waiter, stream)) => {
                report.step("create_session(D3D11)", true, json!({"adapter": dev.adapter_name}));
                d3d = Some(dev);
                Some((session.into_any_graphics(), Some((waiter, stream))))
            }
            Err(e) => {
                report.step("create_session(D3D11)", false, e);
                None
            }
        }
    } else {
        None
    };
    let session = match session {
        Some(s) => Some(s),
        None if instance.exts().mnd_headless.is_some() => {
            match unsafe { instance.create_session::<xr::Headless>(system, &xr::headless::SessionCreateInfo {}) } {
                Ok((s, _, _)) => {
                    report.step("create_session(headless)", true, json!("scene only; no depth pixels"));
                    Some((s.into_any_graphics(), None))
                }
                Err(r) => {
                    report.step("create_session(headless)", false, xr_result_json(r));
                    None
                }
            }
        }
        None => None,
    };
    let Some((session, frames)) = session else {
        return report.step("create_session", false, json!("no usable graphics binding"));
    };
    let mut ctx = XrCtx::new(instance.clone(), session, frames);

    let running = ctx.wait_running(Duration::from_secs(15));
    report.step("session_running", running, json!({"states": ctx.state_log, "errors": ctx.errors}));
    if !running {
        return;
    }

    let space = ctx
        .session
        .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
        .map(|s| (s, "STAGE"))
        .or_else(|_| {
            ctx.session
                .create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)
                .map(|s| (s, "LOCAL"))
        });
    match space {
        Ok((s, name)) => {
            ctx.space = Some(s);
            report.step("reference_space", true, json!(name));
        }
        Err(r) => report.step("reference_space", false, xr_result_json(r)),
    }
    let clock_offset = ctx.clock_offset_ns();
    report.set("clock_offset_ns_unix_minus_xr", json!(clock_offset));

    // 6. Depth
    if args.depth {
        run_depth(args, report, &mut ctx, d3d.as_mut());
    }

    // 7. Scene
    if args.scene {
        let t0 = Instant::now();
        let scene = scene::run(&mut ctx);
        let ok = scene["ok"].as_bool().unwrap_or(false);
        write_json(&args.out.join("scene.json"), &scene);
        report.step("scene", ok, json!({
            "elapsed_s": t0.elapsed().as_secs_f64(),
            "rooms": scene.get("rooms").map(|r| r.as_array().map_or(0, Vec::len)),
            "anchor_count": scene.get("anchor_count"),
            "label_counts": scene.get("label_counts"),
            "global_mesh_triangles": scene.get("global_mesh_triangles"),
            "errors": [scene.get("error"), scene.get("rooms_error"), scene.get("anchors_error")],
        }));
    }

    ctx.shutdown();
    report.set("session", json!({"states": ctx.state_log, "errors": ctx.errors, "frames": ctx.frames_ticked}));
}

fn run_depth(args: &Args, report: &mut Report, ctx: &mut XrCtx, mut d3d: Option<&mut D3d>) {
    if ctx.instance.exts().meta_environment_depth.is_none() {
        return report.step("depth", false, json!("XR_META_environment_depth not available"));
    }
    let Some(space) = ctx.space.as_ref().map(|s| s.as_raw()) else {
        return report.step("depth", false, json!("no reference space"));
    };
    let with_images = d3d.is_some();
    let mut steps = Map::new();
    let probe = depth::create(&ctx.instance, ctx.session.as_raw(), with_images, &mut steps);
    let Some(mut probe) = probe else {
        return report.step("depth_create", false, Value::Object(steps));
    };
    let started = probe.start(&mut steps, with_images);
    report.step("depth_create", started, Value::Object(steps));
    if !started {
        return;
    }

    let tap_path = args.out.join("depth.rktap");
    let meta = json!({
        "recording_start_utc": utc_iso(unix_now_ns()),
        "listener_version": VERSION,
        "source": "link_spike (native OpenXR over Quest Link)",
        "feeds": {"depth": true, "camera": false},
        "marker_msg_type": link_spike::tap::MSG_MARKER,
    });
    match TapWriter::create(&tap_path, &meta) {
        Ok(w) => probe.set_writer(w),
        Err(e) => report.step("depth_tap_open", false, json!(e.to_string())),
    }

    let window = Duration::from_secs_f64(args.seconds);
    let t0 = Instant::now();
    while t0.elapsed() < window && !ctx.exit {
        let offset = ctx.clock_offset_ns();
        let d3d_now = d3d.as_deref_mut();
        ctx.tick(|time| probe.poll(space, time, d3d_now, offset));
    }
    let summary = probe.summary(MIN_FPS);
    drop(probe);
    write_json(&args.out.join("depth_summary.json"), &summary);
    let ok = summary["new_frames"].as_u64().unwrap_or(0) > 0;
    report.step("depth_capture", ok, json!({
        "seconds": args.seconds,
        "new_frames": summary["new_frames"],
        "fps": summary["fps"],
        "fps_ok": summary["fps_ok"],
        "resolution_per_view": summary["resolution_per_view"],
        "acquire": [summary["acquire_ok"], summary["acquire_not_available"], summary["acquire_errors"]],
        "readback_errors": summary["readback_errors"],
        "frames_written": summary["frames_written"],
        "pixels_ok": summary["pixels_ok"],
        "tap": tap_path.display().to_string(),
    }));
}

#[allow(clippy::type_complexity)]
fn create_d3d11_session(
    instance: &xr::Instance,
    system: xr::SystemId,
) -> Result<(D3d, xr::Session<xr::D3D11>, xr::FrameWaiter, xr::FrameStream<xr::D3D11>), Value> {
    let req = instance
        .graphics_requirements::<xr::D3D11>(system)
        .map_err(|r| json!({"step": "xrGetD3D11GraphicsRequirementsKHR", "error": xr_result_json(r)}))?;
    let dev = D3d::create(req.adapter_luid.LowPart, req.adapter_luid.HighPart, req.min_feature_level)
        .map_err(|e| json!({"step": "D3D11 device", "error": e}))?;
    let info = xr::d3d::SessionCreateInfoD3D11 { device: dev.device_ptr() };
    let (session, waiter, stream) = unsafe { instance.create_session::<xr::D3D11>(system, &info) }
        .map_err(|r| json!({"step": "xrCreateSession", "error": xr_result_json(r)}))?;
    Ok((dev, session, waiter, stream))
}

fn enumerate_extensions(entry: &xr::Entry) -> Result<Vec<(String, u32)>, sys::Result> {
    let f = entry.fp().enumerate_instance_extension_properties;
    unsafe {
        let mut count = 0u32;
        let r = f(std::ptr::null(), 0, &mut count, std::ptr::null_mut());
        if r.into_raw() < 0 {
            return Err(r);
        }
        let mut props: Vec<sys::ExtensionProperties> = (0..count)
            .map(|_| {
                let mut p: sys::ExtensionProperties = std::mem::zeroed();
                p.ty = sys::ExtensionProperties::TYPE;
                p
            })
            .collect();
        let r = f(std::ptr::null(), count, &mut count, props.as_mut_ptr());
        if r.into_raw() < 0 {
            return Err(r);
        }
        props.truncate(count as usize);
        Ok(props
            .iter()
            .map(|p| {
                let name = std::ffi::CStr::from_ptr(p.extension_name.as_ptr()).to_string_lossy().into_owned();
                (name, p.extension_version)
            })
            .collect())
    }
}

fn wanted_subset(a: &xr::ExtensionSet) -> xr::ExtensionSet {
    let mut e = xr::ExtensionSet::default();
    e.khr_d3d11_enable = a.khr_d3d11_enable;
    e.mnd_headless = a.mnd_headless;
    e.khr_win32_convert_performance_counter_time = a.khr_win32_convert_performance_counter_time;
    e.meta_environment_depth = a.meta_environment_depth;
    e.fb_scene = a.fb_scene;
    e.fb_spatial_entity = a.fb_spatial_entity;
    e.fb_spatial_entity_query = a.fb_spatial_entity_query;
    e.fb_spatial_entity_storage = a.fb_spatial_entity_storage;
    e.fb_spatial_entity_container = a.fb_spatial_entity_container;
    e.meta_spatial_entity_mesh = a.meta_spatial_entity_mesh;
    e
}

fn system_props(instance: &xr::Instance, system: xr::SystemId) -> Value {
    let mut out = Map::new();
    if let Ok(p) = instance.system_properties(system) {
        out.insert("system_name".into(), json!(p.system_name));
        out.insert("vendor_id".into(), json!(p.vendor_id));
    }
    if instance.exts().meta_environment_depth.is_some() {
        let mut depth: sys::SystemEnvironmentDepthPropertiesMETA = unsafe { std::mem::zeroed() };
        depth.ty = sys::SystemEnvironmentDepthPropertiesMETA::TYPE;
        let mut props: sys::SystemProperties = unsafe { std::mem::zeroed() };
        props.ty = sys::SystemProperties::TYPE;
        props.next = &mut depth as *mut _ as *mut _;
        let r = unsafe { (instance.fp().get_system_properties)(instance.as_raw(), system, &mut props) };
        out.insert("environment_depth_props".into(), if r.into_raw() >= 0 {
            json!({"supports_environment_depth": bool::from(depth.supports_environment_depth),
                   "supports_hand_removal": bool::from(depth.supports_hand_removal)})
        } else {
            xr_result_json(r)
        });
    }
    Value::Object(out)
}

/// Feasibility per README: depth >= 9 fps with pixels, and scene data present.
fn verdict(report: &Report) -> Value {
    let step = |name: &str| report.steps.iter().find(|s| s["step"] == name);
    let depth = step("depth_capture");
    let fps = depth.and_then(|d| d["detail"]["fps"].as_f64()).unwrap_or(0.0);
    let pixels = depth.is_some_and(|d| d["detail"]["pixels_ok"] == true);
    let scene_ok = step("scene").is_some_and(|s| s["ok"] == true);
    let last_failure = report.steps.iter().rev().find(|s| s["ok"] == false).map(|s| s["step"].clone());
    json!({
        "depth_fps": fps,
        "depth_pixels_ok": pixels,
        "depth_ok": depth.is_some_and(|d| d["ok"] == true) && pixels && fps >= MIN_FPS,
        "scene_ok": scene_ok,
        "feasible": depth.is_some_and(|d| d["ok"] == true) && pixels && fps >= MIN_FPS && scene_ok,
        "last_failed_step": last_failure,
    })
}

/// Synthetic two-view depth tap (a ramp), to check tap_inspect / fusion compatibility offline.
fn write_synth_tap(path: &Path) {
    let a = 0.8f32;
    let mut w = TapWriter::create(path, &json!({
        "recording_start_utc": utc_iso(unix_now_ns()),
        "listener_version": VERSION,
        "source": "link_spike --synth-tap",
        "feeds": {"depth": true, "camera": false},
    }))
    .expect("cannot create tap");
    let (width, h) = (320u32, 320u32);
    for i in 0..30u64 {
        let pixels: Vec<u8> = (0..width * h * 2)
            .flat_map(|p| (((p % width) * 200 + i as u32) as u16).to_le_bytes())
            .collect();
        let now = unix_now_ns() + i * 33_333_333;
        let frame = DepthFrame {
            client_timestamp_ns: i * 33_333_333,
            view_poses: [[0.0, 0.0, 0.0, 1.0, -0.03, 1.6, 0.0], [0.0, 0.0, 0.0, 1.0, 0.03, 1.6, 0.0]],
            width,
            height: h * 2,
            near_z: 0.1,
            far_z: f32::INFINITY,
            format: FORMAT_RAW_D16,
            fov_angles: [[-a, a, a, -a]; 2],
            server_timestamp_unix_ns: now,
            clock_offset_ns: 0,
            server_receive_unix_ns: now,
        };
        w.write(now, MSG_DEPTH_FRAME_V2, &encode_depth_frame_v2(&frame, &pixels)).expect("write");
    }
    println!("wrote {}", path.display());
}

fn write_json(path: &Path, v: &Value) {
    if let Err(e) = std::fs::write(path, serde_json::to_string_pretty(v).unwrap_or_default()) {
        eprintln!("cannot write {}: {e}", path.display());
    }
}

fn compact(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 300 { format!("{}...", &s[..s.floor_char_boundary(300)]) } else { s }
}

/// Unix ns -> "YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ" (no chrono dependency).
fn utc_iso(unix_ns: u64) -> String {
    let secs = unix_ns / 1_000_000_000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:09}Z",
        rem / 3600, rem % 3600 / 60, rem % 60, unix_ns % 1_000_000_000
    )
}

fn stamp() -> String {
    let s = utc_iso(unix_now_ns());
    format!("{}{}{}-{}{}{}", &s[0..4], &s[5..7], &s[8..10], &s[11..13], &s[14..16], &s[17..19])
}

// Howard Hinnant's days -> civil date.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}
