//! Integration: turn emits metrics into `RecordingMetrics` / Prometheus.
#![allow(
    unused_crate_dependencies,
    clippy::expect_used,
    clippy::tests_outside_test_module,
    reason = "integration test"
)]

use std::sync::Arc;

use ovo_agent::AgentBuilder;
use ovo_llm::MockSampler;
use ovo_obs::{
    METRIC_SAMPLE_DURATION_MS, METRIC_TURNS_TOTAL, PrometheusRecorder, RecordingMetrics,
    SharedMetrics, emit_catalogue_smoke, metric_catalogue_snapshot, record_turn,
    required_metric_names, span_catalogue_snapshot,
};
use ovo_runtime::{Session, TurnInput, TurnOptions, VecConversationState};

#[tokio::test]
async fn session_records_turn_metrics() {
    let rec = Arc::new(RecordingMetrics::new());
    let metrics: SharedMetrics = rec.clone();
    let sampler = Arc::new(MockSampler::new());
    sampler.push_text("ok");
    let agent = AgentBuilder::named("a")
        .model("mock")
        .build()
        .expect("agent");
    let mut state = VecConversationState::new();
    let mut session = Session::new();
    session
        .run_turn_with_metrics(
            &agent,
            sampler.as_ref(),
            &mut state,
            TurnInput::Text("hi".into()),
            TurnOptions::default().with_metrics(metrics),
            rec.as_ref(),
        )
        .await
        .expect("turn");
    assert!(rec.saw(METRIC_TURNS_TOTAL));
    assert!(rec.saw(METRIC_SAMPLE_DURATION_MS) || rec.counter_sum(METRIC_TURNS_TOTAL) >= 1);
}

#[test]
fn prometheus_text_nonempty() {
    let p = PrometheusRecorder::new();
    record_turn(&p, "ok", 3, 1.5);
    let text = p.render();
    assert!(text.contains(METRIC_TURNS_TOTAL), "{text}");
}

#[test]
fn catalogue_snapshots_are_pinned() {
    assert!(metric_catalogue_snapshot().contains(METRIC_TURNS_TOTAL));
    assert_eq!(required_metric_names().len(), 11);
    assert!(span_catalogue_snapshot().contains("ovo.turn"));
}

#[test]
fn prometheus_catalogue_smoke_export() {
    let p = PrometheusRecorder::new();
    emit_catalogue_smoke(&p);
    let text = p.render();
    for name in required_metric_names() {
        assert!(
            p.series_names().contains(*name),
            "missing {name} in export:\n{text}"
        );
    }
    assert!(text.contains("# HELP"));
    assert!(text.contains("# TYPE"));
}
