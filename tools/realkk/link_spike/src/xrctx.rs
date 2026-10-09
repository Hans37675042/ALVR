//! Session lifetime and the event / frame pump shared by the depth and scene probes.

use openxr::{self as xr, sys};
use serde_json::{Value, json};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub fn unix_now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

pub fn xr_result_json(r: sys::Result) -> Value {
    json!({"result": format!("{r:?}"), "code": r.into_raw()})
}

pub struct XrCtx {
    pub instance: xr::Instance,
    pub session: xr::Session<xr::AnyGraphics>,
    pub frames: Option<(xr::FrameWaiter, xr::FrameStream<xr::D3D11>)>,
    pub space: Option<xr::Space>,
    pub state: sys::SessionState,
    pub running: bool,
    pub exit: bool,
    /// Predicted display time of the last frame, or a QPC conversion when there are no frames.
    pub last_time: sys::Time,
    pub frames_ticked: u64,
    pub state_log: Vec<Value>,
    pub errors: Vec<Value>,
    started: Instant,
}

impl XrCtx {
    pub fn new(
        instance: xr::Instance,
        session: xr::Session<xr::AnyGraphics>,
        frames: Option<(xr::FrameWaiter, xr::FrameStream<xr::D3D11>)>,
    ) -> Self {
        Self {
            instance,
            session,
            frames,
            space: None,
            state: sys::SessionState::UNKNOWN,
            running: false,
            exit: false,
            last_time: sys::Time::from_nanos(0),
            frames_ticked: 0,
            state_log: Vec::new(),
            errors: Vec::new(),
            started: Instant::now(),
        }
    }

    /// XrTime "now" through XR_KHR_win32_convert_performance_counter_time, if enabled.
    pub fn xr_now(&self) -> Option<sys::Time> {
        let convert = self.instance.exts().khr_win32_convert_performance_counter_time.as_ref()?;
        let mut qpc = 0i64;
        unsafe {
            windows::Win32::System::Performance::QueryPerformanceCounter(&mut qpc).ok()?;
            let mut t = sys::Time::from_nanos(0);
            let r = (convert.convert_win32_performance_counter_to_time)(
                self.instance.as_raw(),
                &qpc,
                &mut t,
            );
            (r.into_raw() >= 0).then_some(t)
        }
    }

    /// Unix ns minus XrTime ns, when the runtime clock can be related to the wall clock.
    pub fn clock_offset_ns(&self) -> Option<i64> {
        let xr_now = self.xr_now()?;
        Some(unix_now_ns() as i64 - xr_now.as_nanos())
    }

    /// Wait until the session is running (or `timeout`); keeps pumping events.
    pub fn wait_running(&mut self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while !self.running && !self.exit && Instant::now() < deadline {
            self.tick(|_| {});
        }
        self.running
    }

    /// Pump events, then run one empty frame (calling `in_frame` between xrBeginFrame and
    /// xrEndFrame). Returns the events this pump does not handle itself.
    pub fn tick(&mut self, in_frame: impl FnOnce(sys::Time)) -> Vec<sys::EventDataBuffer> {
        let others = self.pump_events();
        if self.running
            && let Some((waiter, stream)) = self.frames.as_mut()
        {
            let step = (|| -> xr::Result<()> {
                let state = waiter.wait()?;
                stream.begin()?;
                self.last_time = state.predicted_display_time;
                in_frame(state.predicted_display_time);
                stream.end(state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[])
            })();
            self.frames_ticked += 1;
            if let Err(e) = step {
                self.note_error("frame loop", e);
                std::thread::sleep(Duration::from_millis(10));
            }
        } else {
            if let Some(t) = self.xr_now() {
                self.last_time = t;
                if self.running {
                    in_frame(t);
                }
            }
            std::thread::sleep(Duration::from_millis(11));
        }
        others
    }

    pub fn note_error(&mut self, what: &str, e: sys::Result) {
        if self.errors.len() < 50 {
            let mut v = xr_result_json(e);
            v["what"] = json!(what);
            v["t_s"] = json!(self.started.elapsed().as_secs_f64());
            self.errors.push(v);
        }
    }

    fn pump_events(&mut self) -> Vec<sys::EventDataBuffer> {
        let mut others = Vec::new();
        loop {
            let mut buf: sys::EventDataBuffer = unsafe { std::mem::zeroed() };
            buf.ty = sys::EventDataBuffer::TYPE;
            let r = unsafe { (self.instance.fp().poll_event)(self.instance.as_raw(), &mut buf) };
            if r != sys::Result::SUCCESS {
                if r.into_raw() < 0 {
                    self.note_error("xrPollEvent", r);
                }
                break;
            }
            match buf.ty {
                sys::StructureType::EVENT_DATA_SESSION_STATE_CHANGED => {
                    let ev = unsafe {
                        &*(&buf as *const _ as *const sys::EventDataSessionStateChanged)
                    };
                    self.on_state(ev.state);
                }
                sys::StructureType::EVENT_DATA_INSTANCE_LOSS_PENDING => {
                    self.state_log.push(json!({"event": "INSTANCE_LOSS_PENDING"}));
                    self.exit = true;
                }
                _ => others.push(buf),
            }
        }
        others
    }

    fn on_state(&mut self, state: sys::SessionState) {
        self.state = state;
        self.state_log.push(json!({
            "state": format!("{state:?}"),
            "t_s": self.started.elapsed().as_secs_f64(),
        }));
        match state {
            sys::SessionState::READY => {
                match self.session.begin(xr::ViewConfigurationType::PRIMARY_STEREO) {
                    Ok(_) => self.running = true,
                    Err(e) => {
                        self.note_error("xrBeginSession", e);
                        self.exit = true;
                    }
                }
            }
            sys::SessionState::STOPPING => {
                if let Err(e) = self.session.end() {
                    self.note_error("xrEndSession", e);
                }
                self.running = false;
                self.exit = true;
            }
            sys::SessionState::EXITING | sys::SessionState::LOSS_PENDING => self.exit = true,
            _ => {}
        }
    }

    /// Request exit and keep pumping until the runtime stops the session (bounded).
    pub fn shutdown(&mut self) {
        if self.running {
            if let Err(e) = self.session.request_exit() {
                self.note_error("xrRequestExitSession", e);
                return;
            }
            let deadline = Instant::now() + Duration::from_secs(3);
            while self.running && Instant::now() < deadline {
                self.tick(|_| {});
            }
        }
    }
}
