//! Journaled workflow fan-out via Rhai + `SessionHost` adapter (offline mock).
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    unused_crate_dependencies,
    reason = "offline demo binary uses stdout and expect for setup"
)]

use std::sync::Arc;

use ovo::{
    InProcessHost, Journal, MockSampler, SessionHost, WorkflowOutcome, WorkflowRunParams,
    run_workflow_on_host,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<(), String> {
    let sampler = Arc::new(MockSampler::new());
    sampler.push_text("plan: two shards");
    // Concurrent parallel() workers: key by prompt to avoid FIFO races.
    sampler.map_user_text("research shard 0", "shard-0 evidence");
    sampler.map_user_text("research shard 1", "shard-1 evidence");

    let host: Arc<dyn SessionHost> = Arc::new(InProcessHost::new(sampler, Vec::new()));
    let script = r#"
        let meta = #{
            name: "fanout-demo",
            description: "plan then parallel research",
        };
        phase("plan");
        let plan = agent("make a plan", #{ label: "planner" });
        phase("research");
        let shards = parallel([
            #{ prompt: "research shard 0", label: "shard-0" },
            #{ prompt: "research shard 1", label: "shard-1" },
        ]);
        complete(#{ plan: plan, shards: shards });
    "#;

    let (tx, _rx) = mpsc::unbounded_channel();
    let outcome = run_workflow_on_host(
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
    )
    .await
    .map_err(|e| e.to_string())?;

    match outcome {
        WorkflowOutcome::Completed { result } => {
            println!("workflow_outcome=completed");
            println!(
                "plan_output={}",
                result
                    .get("plan")
                    .and_then(|p| p.get("output"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
            );
            let shards = result
                .get("shards")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            println!("shard_count={}", shards.len());
            for (i, s) in shards.iter().enumerate() {
                println!(
                    "shard_{i}_output={}",
                    s.get("output").and_then(|v| v.as_str()).unwrap_or("")
                );
            }
            println!("final={result}");
            Ok(())
        }
        other => Err(format!("unexpected outcome: {other:?}")),
    }
}
