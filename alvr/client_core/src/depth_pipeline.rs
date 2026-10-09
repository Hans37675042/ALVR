//! Pure bookkeeping for the client's environment depth uplink (timing stats, capture pacing,
//! readback slot ring, row flipping, send worker link). Nothing here touches GL or OpenXR, so it
//! can be unit tested on the host; `alvr_client_openxr` drives it from the render thread.

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
}
