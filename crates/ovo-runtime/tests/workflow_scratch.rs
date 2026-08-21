//! Integration: workflow scratch + template via `WorkflowSideEffects`.
#![allow(
    unused_crate_dependencies,
    clippy::expect_used,
    clippy::tests_outside_test_module,
    reason = "integration test binary"
)]

use std::sync::Arc;

use ovo_llm::MockSampler;
use ovo_runtime::{InProcessHost, SessionHost, WorkflowSideEffects, run_workflow_configured};
use ovo_workflow::{Journal, WorkflowOutcome, WorkflowRunParams};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scratch_and_template_round_trip() {
    let sampler = Arc::new(MockSampler::new());
    sampler.map_user_text("summarize", "summary-ok");
    let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, vec![]));
    let effects = WorkflowSideEffects::shared();
    effects.register_template("report", "Report: {{body}}");

    let script = r#"
        let meta = #{ name: "scratch", description: "side effects" };
        let path = write_scratch_file("note.txt", "hello-side-effect");
        let body = read_scratch_file("note.txt");
        let rendered = render_template("report", #{ body: body });
        let a = agent("summarize", #{ label: "s" });
        complete(#{ path: path, body: body, rendered: rendered, agent: a.output });
    "#;

    let (tx, _rx) = mpsc::unbounded_channel();
    let outcome = run_workflow_configured(
        host,
        WorkflowRunParams {
            script: script.into(),
            args: serde_json::json!({}),
            journal: Journal::new(None),
            host_tx: tx,
            cancel: CancellationToken::new(),
            max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
        },
        Some(8),
        Arc::new(ovo_obs::NoopMetrics),
        Arc::clone(&effects),
    )
    .await
    .expect("run");

    let result = match outcome {
        WorkflowOutcome::Completed { result } => result,
        other => {
            unreachable!("expected completed outcome, got {other:?}");
        }
    };
    assert_eq!(
        result.get("body").and_then(|v| v.as_str()),
        Some("hello-side-effect")
    );
    assert_eq!(
        result.get("rendered").and_then(|v| v.as_str()),
        Some("Report: hello-side-effect")
    );
    assert_eq!(effects.scratch_len(), 1);
}
