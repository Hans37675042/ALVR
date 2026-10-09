//! Pure bookkeeping for the client's environment depth uplink (timing stats, capture pacing,
//! readback slot ring, row flipping, send worker link). Nothing here touches GL or OpenXR, so it
//! can be unit tested on the host; `alvr_client_openxr` drives it from the render thread.

use std::{
    sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError},
    time::{Duration, Instant},
};

/// Percentile summary of one stage's samples since the last report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageSummary {
    pub count: usize,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
}

/// Samples of one pipeline stage (milliseconds, frames, ...), summarized and cleared together.
#[derive(Default)]
pub struct StageStats {
    samples: Vec<f32>,
}

impl StageStats {
    pub fn record(&mut self, value: f32) {
        self.samples.push(value);
    }

    /// Nearest-rank p50/p95 and max of the samples recorded since the last call.
    pub fn take_summary(&mut self) -> Option<StageSummary> {
        if self.samples.is_empty() {
            return None;
        }
        self.samples.sort_by(f32::total_cmp);
        let n = self.samples.len();
        let rank = |p: f32| self.samples[((p * n as f32).ceil() as usize).clamp(1, n) - 1];
        let summary = StageSummary {
            count: n,
            p50: rank(0.50),
            p95: rank(0.95),
            max: self.samples[n - 1],
        };
        self.samples.clear();
        Some(summary)
    }
}

/// Named stages reported on one `[XR_PERF]` log line, in the order they were first recorded.
#[derive(Default)]
pub struct StageSet {
    stages: Vec<(&'static str, StageStats)>,
}

impl StageSet {
    pub fn record(&mut self, name: &'static str, value: f32) {
        if let Some((_, stats)) = self.stages.iter_mut().find(|(n, _)| *n == name) {
            stats.record(value);
        } else {
            let mut stats = StageStats::default();
            stats.record(value);
            self.stages.push((name, stats));
        }
    }

    /// One line with every stage that got samples since the last call; clears the samples.
    pub fn format_and_reset(&mut self) -> String {
        self.stages
            .iter_mut()
            .filter_map(|(name, stats)| {
                let s = stats.take_summary()?;
                Some(format!(
                    "{name} p50 {:.2} p95 {:.2} max {:.2} (n={})",
                    s.p50, s.p95, s.max, s.count
                ))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Decides on which rendered frames a depth capture is due.
///
/// Captures follow a fixed schedule (`next_due += interval`) instead of "one interval after the
/// last capture", so frame quantization does not stretch the period (at 72 Hz the latter turns
/// 100 ms into 111 ms, i.e. 9 fps). After a stall the schedule restarts from the capture instead
/// of catching up with a burst. Frames where nothing was captured do not touch the schedule, so
/// the next frame retries.
#[derive(Default)]
pub struct CapturePacer {
    next_due: Option<Instant>,
}

impl CapturePacer {
    pub fn is_due(&self, now: Instant) -> bool {
        self.next_due.is_none_or(|due| now >= due)
    }

    /// A capture was taken (or attempted with GPU work) on this frame.
    pub fn on_captured(&mut self, now: Instant, interval: Duration) {
        let next = self.next_due.map_or(now + interval, |due| due + interval);
        self.next_due = Some(if next <= now { now + interval } else { next });
    }

    /// Wait a full interval from now, e.g. after an error that should not be retried every frame.
    pub fn defer(&mut self, now: Instant, interval: Duration) {
        self.next_due = Some(now + interval);
    }
}

enum Slot<F, M> {
    Free,
    Pending {
        fence: F,
        meta: M,
        enqueued_frame: u64,
    },
}

/// A finished readback slot, handed out oldest first. The slot is free again once returned, so
/// the caller must consume its buffer before submitting new GPU work into it.
#[derive(Debug)]
pub enum SlotEvent<F, M> {
    /// The fence signaled: the slot's buffer holds the image described by `meta`.
    Ready {
        slot: usize,
        fence: F,
        meta: M,
        latency_frames: u64,
    },
    /// The fence did not signal within the timeout; the buffer content must not be used.
    Expired { slot: usize, fence: F, meta: M },
}

/// Ring of asynchronous GPU readback slots (one pixel pack buffer each). `F` is the GPU fence,
/// `M` whatever describes the image in flight (pose, FOV, size), so late frames keep the header
/// of the moment they were captured.
pub struct SlotRing<F, M> {
    slots: Vec<Slot<F, M>>,
    next: usize,
}

impl<F, M> SlotRing<F, M> {
    pub fn new(count: usize) -> Self {
        Self {
            slots: (0..count).map(|_| Slot::Free).collect(),
            next: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The next free slot in round-robin order; None if every slot is still in flight.
    pub fn free_slot(&self) -> Option<usize> {
        let n = self.slots.len();
        (0..n)
            .map(|i| (self.next + i) % n)
            .find(|&i| matches!(self.slots[i], Slot::Free))
    }

    pub fn submit(&mut self, slot: usize, fence: F, meta: M, frame: u64) {
        assert!(matches!(self.slots[slot], Slot::Free), "slot {slot} is in flight");
        self.slots[slot] = Slot::Pending {
            fence,
            meta,
            enqueued_frame: frame,
        };
        self.next = (slot + 1) % self.slots.len();
    }

    pub fn pending(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, Slot::Pending { .. }))
            .count()
    }

    /// Checks the oldest in-flight slot without waiting. Returns it when its fence signaled or
    /// when it has been pending for more than `timeout_frames`; call again until None.
    pub fn poll_one(
        &mut self,
        frame: u64,
        timeout_frames: u64,
        mut is_signaled: impl FnMut(&F) -> bool,
    ) -> Option<SlotEvent<F, M>> {
        let oldest = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Pending { enqueued_frame, .. } => Some((i, *enqueued_frame)),
                Slot::Free => None,
            })
            .min_by_key(|&(_, f)| f)?;
        let (slot, enqueued_frame) = oldest;
        let Slot::Pending { fence, .. } = &self.slots[slot] else {
            unreachable!()
        };
        let latency_frames = frame.saturating_sub(enqueued_frame);
        let signaled = is_signaled(fence);
        if !signaled && latency_frames <= timeout_frames {
            return None;
        }
        let Slot::Pending { fence, meta, .. } = std::mem::replace(&mut self.slots[slot], Slot::Free)
        else {
            unreachable!()
        };
        Some(if signaled {
            SlotEvent::Ready {
                slot,
                fence,
                meta,
                latency_frames,
            }
        } else {
            SlotEvent::Expired { slot, fence, meta }
        })
    }

    /// Frees every slot and returns the fences still in flight, e.g. to delete them on teardown.
    pub fn drain(&mut self) -> Vec<F> {
        self.slots
            .iter_mut()
            .filter_map(|s| match std::mem::replace(s, Slot::Free) {
                Slot::Pending { fence, .. } => Some(fence),
                Slot::Free => None,
            })
            .collect()
    }
}

pub enum SubmitOutcome {
    Queued,
    /// The worker is still busy with the previous frame; this one was dropped.
    DroppedFull,
    /// The worker is gone; this one was dropped.
    Disconnected,
}

/// Render thread side of the depth send worker: hands frames over without blocking and reuses
/// the buffers the worker returns, so steady state allocates nothing.
pub struct SendLink<M> {
    jobs: SyncSender<(Vec<u8>, M)>,
    recycled: Receiver<Vec<u8>>,
    spare: Vec<Vec<u8>>,
    allocations: u64,
    dropped: u64,
}

/// Worker thread side: receives frames and returns each buffer once processed.
pub struct SendWorker<M> {
    jobs: Receiver<(Vec<u8>, M)>,
    recycle: Sender<Vec<u8>>,
}

/// `capacity` frames may wait for the worker; more are dropped instead of queued.
pub fn send_link<M>(capacity: usize) -> (SendLink<M>, SendWorker<M>) {
    let (jobs_tx, jobs_rx) = mpsc::sync_channel(capacity);
    let (recycle_tx, recycle_rx) = mpsc::channel();
    (
        SendLink {
            jobs: jobs_tx,
            recycled: recycle_rx,
            spare: vec![],
            allocations: 0,
            dropped: 0,
        },
        SendWorker {
            jobs: jobs_rx,
            recycle: recycle_tx,
        },
    )
}

impl<M> SendLink<M> {
    /// A buffer of `len` bytes, recycled when possible. Its content is unspecified.
    pub fn take_buffer(&mut self, len: usize) -> Vec<u8> {
        self.spare.extend(self.recycled.try_iter());
        let mut buf = self.spare.pop().unwrap_or_else(|| {
            self.allocations += 1;
            Vec::with_capacity(len)
        });
        buf.resize(len, 0);
        buf
    }

    pub fn submit(&mut self, buf: Vec<u8>, meta: M) -> SubmitOutcome {
        match self.jobs.try_send((buf, meta)) {
            Ok(()) => SubmitOutcome::Queued,
            Err(TrySendError::Full((buf, _))) => {
                self.dropped += 1;
                self.spare.push(buf);
                SubmitOutcome::DroppedFull
            }
            Err(TrySendError::Disconnected((buf, _))) => {
                self.dropped += 1;
                self.spare.push(buf);
                SubmitOutcome::Disconnected
            }
        }
    }

    /// Buffers allocated so far (bounded by the frames in flight)
    pub fn allocations(&self) -> u64 {
        self.allocations
    }

    /// Frames dropped because the worker was busy or gone
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

impl<M> SendWorker<M> {
    /// Waits for one frame and processes it. Returns false once the link was dropped.
    pub fn run_one(&self, process: impl FnOnce(&[u8], M)) -> bool {
        let Ok((buf, meta)) = self.jobs.recv() else {
            return false;
        };
        process(&buf, meta);
        // The link may already be gone; the buffer is then simply freed
        self.recycle.send(buf).ok();
        true
    }

    /// Processes frames until the link is dropped.
    pub fn run(self, mut process: impl FnMut(&[u8], M)) {
        while self.run_one(&mut process) {}
    }
}

/// Whether the capture with this index runs the per-step GL diagnostics (framebuffer status and
/// glGetError after every step). Other captures only read the sticky GL error flag once at the
/// end, which still catches a failed step but cannot say which one.
pub fn detailed_gl_check_due(capture_index: u64) -> bool {
    capture_index < 20 || capture_index % 100 == 0
}

/// Copies `views` stacked images of `rows_per_view` rows each from `src` (bottom-up rows, as
/// glReadPixels returns them, `src_row_stride` bytes apart) into `dst` with top-down rows of
/// `row_bytes` bytes, flipping each view vertically.
pub fn copy_flipped_rows(
    src: &[u8],
    src_row_stride: usize,
    dst: &mut [u8],
    row_bytes: usize,
    rows_per_view: usize,
    views: usize,
) {
    assert!(src_row_stride >= row_bytes);
    assert!(dst.len() >= row_bytes * rows_per_view * views);
    for view in 0..views {
        for row in 0..rows_per_view {
            let src_start = (view * rows_per_view + rows_per_view - 1 - row) * src_row_stride;
            let dst_start = (view * rows_per_view + row) * row_bytes;
            dst[dst_start..dst_start + row_bytes]
                .copy_from_slice(&src[src_start..src_start + row_bytes]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_stats_report_nearest_rank_percentiles_and_max() {
        let mut stats = StageStats::default();
        // 1..=100 shuffled: p50 = 50, p95 = 95, max = 100
        for v in (1..=100).rev() {
            stats.record(v as f32);
        }
        let summary = stats.take_summary().unwrap();
        assert_eq!(summary.count, 100);
        assert_eq!(summary.p50, 50.0);
        assert_eq!(summary.p95, 95.0);
        assert_eq!(summary.max, 100.0);
    }

    #[test]
    fn stage_stats_reset_after_summary() {
        let mut stats = StageStats::default();
        assert!(stats.take_summary().is_none());
        stats.record(3.0);
        let summary = stats.take_summary().unwrap();
        assert_eq!((summary.count, summary.p50, summary.p95, summary.max), (1, 3.0, 3.0, 3.0));
        assert!(stats.take_summary().is_none());
    }

    #[test]
    fn stage_set_formats_stages_in_first_recorded_order() {
        let mut set = StageSet::default();
        set.record("enqueue_ms", 0.5);
        set.record("map_ms", 1.0);
        set.record("enqueue_ms", 1.5);
        assert_eq!(
            set.format_and_reset(),
            "enqueue_ms p50 0.50 p95 1.50 max 1.50 (n=2), map_ms p50 1.00 p95 1.00 max 1.00 (n=1)"
        );
        // Stages stay registered but report nothing until they get new samples
        set.record("map_ms", 2.0);
        assert_eq!(set.format_and_reset(), "map_ms p50 2.00 p95 2.00 max 2.00 (n=1)");
    }


    /// Runs the pacer against a fixed display clock; returns the capture times (seconds).
    fn simulate_pacer(
        refresh_hz: f64,
        seconds: f64,
        interval: Duration,
        stall: Option<(f64, f64)>,
    ) -> Vec<f64> {
        let t0 = Instant::now();
        let mut pacer = CapturePacer::default();
        let mut captures = vec![];
        let frames = (refresh_hz * seconds) as u64;
        for k in 0..frames {
            let t = k as f64 / refresh_hz;
            if let Some((at, length)) = stall
                && t >= at
                && t < at + length
            {
                continue; // no frames rendered during the stall
            }
            let now = t0 + Duration::from_secs_f64(t);
            if pacer.is_due(now) {
                pacer.on_captured(now, interval);
                captures.push(t);
            }
        }
        captures
    }

    #[test]
    fn pacer_holds_target_rate_at_common_refresh_rates() {
        for hz in [72.0, 90.0, 120.0] {
            let captures = simulate_pacer(hz, 10.0, Duration::from_millis(100), None);
            let fps = captures.len() as f64 / 10.0;
            assert!((fps - 10.0).abs() <= 0.2, "{hz} Hz -> {fps} fps");
        }
    }

    #[test]
    fn pacer_does_not_burst_after_a_stall() {
        let captures = simulate_pacer(72.0, 3.0, Duration::from_millis(100), Some((1.0, 0.2)));
        let min_gap = captures
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::MAX, f64::min);
        // Frame-quantized gaps at 72 Hz never drop below ~86 ms unless captures bunch up
        assert!(min_gap > 0.08, "captures bunched up after the stall: gap {min_gap}");
    }

    #[test]
    fn pacer_retries_on_the_next_frame_when_nothing_was_captured() {
        let t0 = Instant::now();
        let mut pacer = CapturePacer::default();
        assert!(pacer.is_due(t0));
        // e.g. the runtime had no depth image yet: do not wait a full interval
        let next_frame = t0 + Duration::from_millis(14);
        assert!(pacer.is_due(next_frame));
        pacer.on_captured(next_frame, Duration::from_millis(100));
        assert!(!pacer.is_due(next_frame + Duration::from_millis(50)));
        assert!(pacer.is_due(next_frame + Duration::from_millis(100)));
    }

    #[test]
    fn pacer_defer_waits_one_interval_from_now() {
        let t0 = Instant::now();
        let mut pacer = CapturePacer::default();
        pacer.defer(t0, Duration::from_millis(100));
        assert!(!pacer.is_due(t0 + Duration::from_millis(99)));
        assert!(pacer.is_due(t0 + Duration::from_millis(100)));
    }

    // Mock fence: an id; the test decides which ids the "GPU" has signaled
    type Ring = SlotRing<u32, &'static str>;

    fn poll(ring: &mut Ring, frame: u64, signaled: &[u32]) -> Option<SlotEvent<u32, &'static str>> {
        ring.poll_one(frame, 30, |fence| signaled.contains(fence))
    }

    #[test]
    fn ring_hands_out_free_slots_round_robin() {
        let mut ring = Ring::new(3);
        let a = ring.free_slot().unwrap();
        ring.submit(a, 1, "a", 0);
        let b = ring.free_slot().unwrap();
        ring.submit(b, 2, "b", 1);
        assert_ne!(a, b);
        assert_eq!(ring.pending(), 2);
    }

    #[test]
    fn ring_keeps_unsignaled_slots_pending() {
        let mut ring = Ring::new(3);
        ring.submit(ring.free_slot().unwrap(), 1, "a", 0);
        assert!(poll(&mut ring, 1, &[]).is_none());
        assert!(poll(&mut ring, 2, &[]).is_none());
        assert_eq!(ring.pending(), 1);
    }

    #[test]
    fn ring_returns_signaled_slot_with_its_metadata_and_frees_it() {
        let mut ring = Ring::new(3);
        let slot = ring.free_slot().unwrap();
        ring.submit(slot, 7, "header of frame 10", 10);
        match poll(&mut ring, 12, &[7]) {
            Some(SlotEvent::Ready {
                slot: s,
                fence,
                meta,
                latency_frames,
            }) => {
                assert_eq!((s, fence, meta, latency_frames), (slot, 7, "header of frame 10", 2));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(ring.pending(), 0);
        assert!(poll(&mut ring, 13, &[7]).is_none());
    }

    #[test]
    fn ring_completes_oldest_first_and_metadata_follows_its_slot() {
        let mut ring = Ring::new(3);
        // Submit in an order that does not match slot indices once slots get reused
        let s0 = ring.free_slot().unwrap();
        ring.submit(s0, 1, "first", 0);
        let s1 = ring.free_slot().unwrap();
        ring.submit(s1, 2, "second", 1);
        assert!(matches!(poll(&mut ring, 2, &[1]), Some(SlotEvent::Ready { meta: "first", .. })));
        let s2 = ring.free_slot().unwrap();
        ring.submit(s2, 3, "third", 2);
        let s3 = ring.free_slot().unwrap();
        ring.submit(s3, 4, "fourth", 3);
        assert_eq!(s3, s0, "the freed slot is reused");
        let signaled = [2, 3, 4];
        let order: Vec<_> = std::iter::from_fn(|| poll(&mut ring, 4, &signaled))
            .map(|e| match e {
                SlotEvent::Ready { meta, .. } => meta,
                SlotEvent::Expired { meta, .. } => panic!("expired {meta}"),
            })
            .collect();
        assert_eq!(order, ["second", "third", "fourth"]);
    }

    #[test]
    fn full_ring_has_no_free_slot() {
        let mut ring = Ring::new(3);
        for i in 0..3 {
            ring.submit(ring.free_slot().unwrap(), i, "x", i as u64);
        }
        assert!(ring.free_slot().is_none());
    }

    #[test]
    fn ring_expires_slots_stuck_past_the_timeout() {
        let mut ring = Ring::new(3);
        ring.submit(ring.free_slot().unwrap(), 9, "stuck", 100);
        assert!(poll(&mut ring, 130, &[]).is_none());
        match poll(&mut ring, 131, &[]) {
            Some(SlotEvent::Expired { fence, meta, .. }) => assert_eq!((fence, meta), (9, "stuck")),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(ring.pending(), 0);
        assert!(ring.free_slot().is_some());
    }

    #[test]
    fn ring_drain_returns_every_pending_fence() {
        let mut ring = Ring::new(3);
        ring.submit(ring.free_slot().unwrap(), 1, "a", 0);
        ring.submit(ring.free_slot().unwrap(), 2, "b", 1);
        let mut fences = ring.drain();
        fences.sort();
        assert_eq!(fences, [1, 2]);
        assert_eq!(ring.pending(), 0);
    }

    /// The flip the depth uplink used before the PBO path (in-place row swap), kept as oracle.
    fn legacy_flip(depth_bytes: &mut [u8], width: usize, height: usize) {
        let u16_byte_count = width * height * 2;
        let row_bytes = width * 2;
        let h = height;
        for eye in 0..2usize {
            let eye_offset = eye * u16_byte_count;
            for row in 0..h / 2 {
                let top = eye_offset + row * row_bytes;
                let bot = eye_offset + (h - 1 - row) * row_bytes;
                for col in 0..row_bytes {
                    depth_bytes.swap(top + col, bot + col);
                }
            }
        }
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 % 251) as u8).collect()
    }

    #[test]
    fn flipped_copy_matches_the_legacy_in_place_flip() {
        for (w, h) in [(320, 320), (5, 3), (4, 4), (1, 1)] {
            let src = pattern(w * h * 2 * 2);
            let mut expected = src.clone();
            legacy_flip(&mut expected, w, h);
            let mut dst = vec![0u8; src.len()];
            copy_flipped_rows(&src, w * 2, &mut dst, w * 2, h, 2);
            assert_eq!(dst, expected, "{w}x{h}");
        }
    }

    #[test]
    fn flipped_copy_skips_source_row_padding() {
        // 3 px wide rows (6 bytes) stored with an 8-byte stride, 2 rows, 2 views
        let (w, h, stride) = (3, 2, 8);
        let mut src = vec![0xEEu8; stride * h * 2];
        let mut packed = vec![];
        for view_row in 0..h * 2 {
            let row: Vec<u8> = (0..w * 2).map(|c| (view_row * 16 + c) as u8).collect();
            src[view_row * stride..view_row * stride + w * 2].copy_from_slice(&row);
            packed.extend(row);
        }
        let mut expected = packed.clone();
        legacy_flip(&mut expected, w, h);
        let mut dst = vec![0u8; packed.len()];
        copy_flipped_rows(&src, stride, &mut dst, w * 2, h, 2);
        assert_eq!(dst, expected);
    }

    #[test]
    fn worker_receives_the_frame_and_returns_its_buffer_for_reuse() {
        let (mut link, worker) = send_link::<u32>(1);
        let mut buf = link.take_buffer(4);
        buf.copy_from_slice(&[1, 2, 3, 4]);
        assert!(matches!(link.submit(buf, 42), SubmitOutcome::Queued));

        let mut seen = vec![];
        assert!(worker.run_one(|bytes, meta| seen.push((bytes.to_vec(), meta))));
        assert_eq!(seen, [(vec![1, 2, 3, 4], 42)]);

        // The same allocation comes back for the next frame
        let again = link.take_buffer(4);
        assert_eq!(again.len(), 4);
        assert_eq!(link.allocations(), 1);
    }

    #[test]
    fn full_queue_drops_the_frame_and_keeps_its_buffer() {
        let (mut link, _worker) = send_link::<u32>(1);
        let first = link.take_buffer(8);
        assert!(matches!(link.submit(first, 1), SubmitOutcome::Queued));
        let second = link.take_buffer(8);
        assert!(matches!(link.submit(second, 2), SubmitOutcome::DroppedFull));
        assert_eq!(link.dropped(), 1);
        let _third = link.take_buffer(8);
        assert_eq!(link.allocations(), 2, "the dropped frame's buffer is reused");
    }

    #[test]
    fn buffer_pool_stays_bounded_with_a_slow_worker() {
        let (mut link, worker) = send_link::<u64>(1);
        for i in 0..1000u64 {
            let buf = link.take_buffer(409_600);
            link.submit(buf, i);
            if i % 3 == 0 {
                worker.run_one(|_, _| ());
            }
        }
        // One being filled, one queued, one returned by the worker
        assert!(link.allocations() <= 3, "{} allocations", link.allocations());
        assert!(link.dropped() > 0);
    }

    #[test]
    fn take_buffer_resizes_recycled_buffers() {
        let (mut link, worker) = send_link::<()>(1);
        let buf = link.take_buffer(16);
        link.submit(buf, ());
        worker.run_one(|_, _| ());
        assert_eq!(link.take_buffer(32).len(), 32);
    }

    #[test]
    fn submit_reports_a_stopped_worker() {
        let (mut link, worker) = send_link::<()>(1);
        drop(worker);
        let buf = link.take_buffer(4);
        assert!(matches!(link.submit(buf, ()), SubmitOutcome::Disconnected));
    }

    #[test]
    fn worker_exits_when_the_link_is_dropped() {
        let (mut link, worker) = send_link::<u8>(1);
        let handle = std::thread::spawn(move || {
            let mut count = 0;
            worker.run(|_, _| count += 1);
            count
        });
        let buf = link.take_buffer(1);
        link.submit(buf, 0);
        drop(link);
        assert_eq!(handle.join().unwrap(), 1);
    }

    #[test]
    fn detailed_gl_checks_run_early_then_every_hundredth_capture() {
        let checked: Vec<u64> = (0..450).filter(|&i| detailed_gl_check_due(i)).collect();
        let mut expected: Vec<u64> = (0..20).collect();
        expected.extend([100, 200, 300, 400]);
        assert_eq!(checked, expected);
    }
}
