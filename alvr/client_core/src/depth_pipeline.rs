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
}
