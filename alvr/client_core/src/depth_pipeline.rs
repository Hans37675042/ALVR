//! Pure bookkeeping for the client's environment depth uplink (timing stats, capture pacing,
//! readback slot ring, row flipping, send worker link). Nothing here touches GL or OpenXR, so it
//! can be unit tested on the host; `alvr_client_openxr` drives it from the render thread.

use std::time::{Duration, Instant};

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

    use std::time::{Duration, Instant};

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
}
