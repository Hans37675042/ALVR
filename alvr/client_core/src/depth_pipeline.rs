//! Pure bookkeeping for the client's environment depth uplink (timing stats, capture pacing,
//! readback slot ring, row flipping, send worker link). Nothing here touches GL or OpenXR, so it
//! can be unit tested on the host; `alvr_client_openxr` drives it from the render thread.

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
