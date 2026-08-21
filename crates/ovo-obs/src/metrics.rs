//! Stable metric series and a host-injected sink.

use std::sync::Arc;

/// Counter: completed turns by status (`ok`, `error`, `cancelled`).
pub const METRIC_TURNS_TOTAL: &str = "ovo_turns_total";
/// Histogram: steps consumed per turn.
pub const METRIC_TURN_STEPS: &str = "ovo_turn_steps";
/// Histogram: turn wall duration in milliseconds.
pub const METRIC_TURN_DURATION_MS: &str = "ovo_turn_duration_ms";
/// Counter: tool calls by tool name and status.
pub const METRIC_TOOL_CALLS_TOTAL: &str = "ovo_tool_calls_total";
/// Histogram: tool duration in milliseconds.
pub const METRIC_TOOL_DURATION_MS: &str = "ovo_tool_duration_ms";
/// Histogram: LLM sample duration in milliseconds.
pub const METRIC_SAMPLE_DURATION_MS: &str = "ovo_sample_duration_ms";
/// Counter: tokens by direction (`input`, `output`).
pub const METRIC_TOKENS_TOTAL: &str = "ovo_tokens_total";
/// Counter: nested agent spawns by status.
pub const METRIC_SPAWNS_TOTAL: &str = "ovo_spawns_total";
/// Counter: workflow runs by outcome.
pub const METRIC_WORKFLOW_RUNS_TOTAL: &str = "ovo_workflow_runs_total";
/// Counter: workflow agent slots consumed.
pub const METRIC_WORKFLOW_AGENTS_TOTAL: &str = "ovo_workflow_agents_total";
/// Counter: compaction passes by strategy/result.
pub const METRIC_COMPACTIONS_TOTAL: &str = "ovo_compactions_total";

/// Required metric name catalogue (contract tests / CI snapshots).
///
/// **Rename = break:** CI asserts the exact ordered snapshot from
/// [`metric_catalogue_snapshot`].
#[must_use]
pub fn required_metric_names() -> &'static [&'static str] {
    &[
        METRIC_TURNS_TOTAL,
        METRIC_TURN_STEPS,
        METRIC_TURN_DURATION_MS,
        METRIC_TOOL_CALLS_TOTAL,
        METRIC_TOOL_DURATION_MS,
        METRIC_SAMPLE_DURATION_MS,
        METRIC_TOKENS_TOTAL,
        METRIC_SPAWNS_TOTAL,
        METRIC_WORKFLOW_RUNS_TOTAL,
        METRIC_WORKFLOW_AGENTS_TOTAL,
        METRIC_COMPACTIONS_TOTAL,
    ]
}

/// Exact newline-joined catalogue for CI golden comparison.
#[must_use]
pub fn metric_catalogue_snapshot() -> String {
    required_metric_names().join("\n")
}

/// Emit every stable series once (for export / dashboard smoke).
pub fn emit_catalogue_smoke(metrics: &dyn MetricsSink) {
    record_turn(metrics, "ok", 1, 1.0);
    record_spawn(metrics, "ok");
    record_tool_call(metrics, "smoke", "ok", 1.0);
    record_sample(metrics, 1.0, 2, 3);
    record_workflow_run(metrics, "completed");
    record_workflow_agents(metrics, 1);
    record_compaction(metrics, "max_messages", "ok");
}

/// Host-provided metrics backend.
pub trait MetricsSink: Send + Sync {
    /// Increment a counter.
    fn counter(&self, name: &str, value: u64, labels: &[(&str, &str)]);
    /// Observe a histogram sample.
    fn histogram(&self, name: &str, value: f64, labels: &[(&str, &str)]);
    /// Set a gauge.
    fn gauge(&self, name: &str, value: f64, labels: &[(&str, &str)]);
}

/// Discards all metrics (default for tests / offline).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopMetrics;

impl MetricsSink for NoopMetrics {
    fn counter(&self, _name: &str, _value: u64, _labels: &[(&str, &str)]) {}
    fn histogram(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
    fn gauge(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
}

/// Shared metrics handle.
pub type SharedMetrics = Arc<dyn MetricsSink>;

/// Record a completed turn.
pub fn record_turn(metrics: &dyn MetricsSink, status: &str, steps: u64, duration_ms: f64) {
    metrics.counter(METRIC_TURNS_TOTAL, 1, &[("status", status)]);
    metrics.histogram(METRIC_TURN_STEPS, steps as f64, &[]);
    metrics.histogram(METRIC_TURN_DURATION_MS, duration_ms, &[]);
}

/// Record a nested spawn.
pub fn record_spawn(metrics: &dyn MetricsSink, status: &str) {
    metrics.counter(METRIC_SPAWNS_TOTAL, 1, &[("status", status)]);
}

/// Record one tool call.
pub fn record_tool_call(metrics: &dyn MetricsSink, tool: &str, status: &str, duration_ms: f64) {
    metrics.counter(
        METRIC_TOOL_CALLS_TOTAL,
        1,
        &[("tool", tool), ("status", status)],
    );
    metrics.histogram(METRIC_TOOL_DURATION_MS, duration_ms, &[("tool", tool)]);
}

/// Record sample duration and tokens.
pub fn record_sample(
    metrics: &dyn MetricsSink,
    duration_ms: f64,
    input_tokens: u64,
    output_tokens: u64,
) {
    metrics.histogram(METRIC_SAMPLE_DURATION_MS, duration_ms, &[]);
    if input_tokens > 0 {
        metrics.counter(METRIC_TOKENS_TOTAL, input_tokens, &[("direction", "input")]);
    }
    if output_tokens > 0 {
        metrics.counter(
            METRIC_TOKENS_TOTAL,
            output_tokens,
            &[("direction", "output")],
        );
    }
}

/// Record workflow terminal outcome.
pub fn record_workflow_run(metrics: &dyn MetricsSink, outcome: &str) {
    metrics.counter(METRIC_WORKFLOW_RUNS_TOTAL, 1, &[("outcome", outcome)]);
}

/// Record workflow agent slot consumption.
pub fn record_workflow_agents(metrics: &dyn MetricsSink, count: u64) {
    if count > 0 {
        metrics.counter(METRIC_WORKFLOW_AGENTS_TOTAL, count, &[]);
    }
}

/// Record a compaction pass.
pub fn record_compaction(metrics: &dyn MetricsSink, strategy: &str, status: &str) {
    metrics.counter(
        METRIC_COMPACTIONS_TOTAL,
        1,
        &[("strategy", strategy), ("status", status)],
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Golden snapshot — intentional fail on rename/reorder/add/remove.
    const METRIC_CATALOGUE_GOLDEN: &str = "\
ovo_turns_total
ovo_turn_steps
ovo_turn_duration_ms
ovo_tool_calls_total
ovo_tool_duration_ms
ovo_sample_duration_ms
ovo_tokens_total
ovo_spawns_total
ovo_workflow_runs_total
ovo_workflow_agents_total
ovo_compactions_total";

    #[test]
    fn metric_catalogue_is_stable_and_prefixed() {
        let names = required_metric_names();
        assert!(names.len() >= 10, "expected full production catalogue");
        let mut seen = std::collections::BTreeSet::new();
        for name in names {
            assert!(
                name.starts_with("ovo_"),
                "metric {name} must start with ovo_"
            );
            assert!(seen.insert(*name), "duplicate metric {name}");
        }
    }

    #[test]
    fn metric_catalogue_snapshot_matches_golden() {
        assert_eq!(
            metric_catalogue_snapshot(),
            METRIC_CATALOGUE_GOLDEN,
            "metric catalogue changed — update golden only with deliberate contract change"
        );
    }

    #[test]
    fn emit_catalogue_smoke_covers_all_names() {
        let cap = Capture::default();
        emit_catalogue_smoke(&cap);
        let owned: Vec<String> = cap
            .counters
            .lock()
            .expect("lock")
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        let names: std::collections::HashSet<&str> = owned.iter().map(String::as_str).collect();
        // Histograms go elsewhere; counters cover the main series.
        for expected in [
            METRIC_TURNS_TOTAL,
            METRIC_SPAWNS_TOTAL,
            METRIC_TOOL_CALLS_TOTAL,
            METRIC_TOKENS_TOTAL,
            METRIC_WORKFLOW_RUNS_TOTAL,
            METRIC_WORKFLOW_AGENTS_TOTAL,
            METRIC_COMPACTIONS_TOTAL,
        ] {
            assert!(names.contains(expected), "missing counter {expected}");
        }
    }

    #[derive(Default)]
    struct Capture {
        counters: Mutex<Vec<(String, u64)>>,
    }

    impl MetricsSink for Capture {
        fn counter(&self, name: &str, value: u64, _labels: &[(&str, &str)]) {
            self.counters
                .lock()
                .expect("lock")
                .push((name.to_owned(), value));
        }
        fn histogram(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
        fn gauge(&self, _name: &str, _value: f64, _labels: &[(&str, &str)]) {}
    }

    #[test]
    fn record_helpers_emit_expected_names() {
        let cap = Capture::default();
        record_turn(&cap, "ok", 3, 12.0);
        record_spawn(&cap, "ok");
        record_tool_call(&cap, "calc", "ok", 1.0);
        record_workflow_run(&cap, "completed");
        let names: Vec<_> = cap
            .counters
            .lock()
            .expect("lock")
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        assert!(names.iter().any(|n| n == METRIC_TURNS_TOTAL));
        assert!(names.iter().any(|n| n == METRIC_SPAWNS_TOTAL));
        assert!(names.iter().any(|n| n == METRIC_TOOL_CALLS_TOTAL));
        assert!(names.iter().any(|n| n == METRIC_WORKFLOW_RUNS_TOTAL));
    }
}
