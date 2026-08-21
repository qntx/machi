//! Short business workflow: plan → parallel → scratch report (offline mock).
//!
//! ```bash
//! cargo run -p ovo --example workflow_plan
//! ```
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    unused_crate_dependencies,
    reason = "demo binary"
)]

use std::sync::Arc;

use ovo::{
    InProcessHost, Journal, MockSampler, PrometheusRecorder, SessionHost, SharedMetrics,
    WorkflowOutcome, WorkflowRunParams, WorkflowSideEffects, run_workflow_configured,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    let sampler = Arc::new(MockSampler::new());
    sampler.push_text("Plan: cover async runtimes and trait objects.");
    sampler.map_user_text(
        "research async runtimes",
        "Tokio uses a work-stealing scheduler.",
    );
    sampler.map_user_text(
        "research trait objects",
        "dyn Trait enables runtime polymorphism.",
    );

    let metrics = Arc::new(PrometheusRecorder::new());
    let metrics_dyn: SharedMetrics = metrics.clone();
    let host: Arc<dyn SessionHost> = Arc::new(
        InProcessHost::new(sampler, Vec::new())
            .with_agent_budget(16)
            .with_metrics(Arc::clone(&metrics_dyn)),
    );
    let effects = WorkflowSideEffects::shared();

    let script = r##"
        let meta = #{
            name: "plan-fanout",
            description: "plan then parallel research then scratch report",
        };
        phase("Plan");
        let plan = agent(
            "Plan: cover async runtimes and trait objects.",
            #{ label: "planner" }
        );
        phase("Research");
        let shards = parallel([
            #{ prompt: "research async runtimes", label: "w0" },
            #{ prompt: "research trait objects", label: "w1" },
        ]);
        let report = "# Report" + "\n\n"
            + "Plan: " + json_encode(plan.output) + "\n\n"
            + "Findings:\n"
            + "- " + json_encode(shards[0].output) + "\n"
            + "- " + json_encode(shards[1].output) + "\n";
        let path = write_scratch_file("report.md", report);
        let b = budget();
        complete(#{
            path: path,
            plan: plan.output,
            shards: shards,
            agents_spent_hint: b.spent,
        });
    "##;

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
        Some(16),
        metrics_dyn,
        Arc::clone(&effects),
    )
    .await
    .expect("workflow");

    match outcome {
        WorkflowOutcome::Completed { result } => {
            println!("workflow_outcome=completed");
            println!(
                "scratch_path={}",
                result.get("path").and_then(|v| v.as_str()).unwrap_or("")
            );
            let body = effects.read_scratch("report.md").expect("scratch");
            println!("report=\n{body}");
            assert!(
                body.contains("Tokio") || body.contains("trait") || body.contains("Plan"),
                "scratch report should include research content: {body}"
            );
            let prom = metrics.render();
            assert!(
                prom.contains("ovo_workflow") || prom.contains("ovo_spawn"),
                "expected workflow/spawn metrics: {prom}"
            );
            println!("workflow_plan_ok=true");
        }
        other => panic!("unexpected: {other:?}"),
    }
}
