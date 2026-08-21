//! Live Ollama workflow e2e (optional; needs `ollama serve` + model).
//!
//! ```bash
//! cargo run -p ovo --example workflow_ollama --features ollama
//! ```
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    clippy::unwrap_used,
    unused_crate_dependencies,
    reason = "live e2e demo"
)]

use std::sync::Arc;
use std::time::Instant;

use ovo::{
    InProcessHost, Journal, OllamaConfig, OllamaSampler, SessionHost, WorkflowOutcome,
    WorkflowRunParams, run_workflow_on_host,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<(), String> {
    let base = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
    let model = std::env::var("OVO_OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:latest".into());
    println!("ollama_target base={base} model={model}");

    let sampler =
        Arc::new(OllamaSampler::new(OllamaConfig::new(&base)).map_err(|e| e.to_string())?);
    let host: Arc<dyn SessionHost> = Arc::new(
        InProcessHost::new(sampler, Vec::new())
            .with_agent_budget(16)
            .with_instructions("Answer in one short sentence. No preamble."),
    );

    let model_lit = model.replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!(
        r#"
        let meta = #{{
            name: "ollama-e2e",
            description: "live ollama plan + parallel",
        }};
        phase("plan");
        let plan = agent(
            "In one short sentence, name two research topics about rust async.",
            #{{ label: "planner", model: "{model_lit}" }}
        );
        phase("research");
        let shards = parallel([
            #{{
                prompt: "In one short sentence, state one fact about tokio runtime.",
                label: "shard-0",
                model: "{model_lit}",
            }},
            #{{
                prompt: "In one short sentence, state one fact about async traits in rust.",
                label: "shard-1",
                model: "{model_lit}",
            }},
        ]);
        complete(#{{ plan: plan, shards: shards }});
        "#
    );

    let (tx, _rx) = mpsc::unbounded_channel();
    let started = Instant::now();
    let outcome = run_workflow_on_host(
        host,
        WorkflowRunParams {
            script,
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
    let elapsed_ms = started.elapsed().as_millis();

    let WorkflowOutcome::Completed { result } = outcome else {
        return Err(format!("unexpected outcome: {outcome:?}"));
    };
    let plan = result
        .pointer("/plan/output")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let shards = result
        .get("shards")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    println!("workflow_outcome=completed");
    println!("elapsed_ms={elapsed_ms}");
    println!("plan_output={plan}");
    println!("shard_count={}", shards.len());
    if plan.trim().is_empty() || shards.len() != 2 {
        return Err("incomplete ollama workflow result".into());
    }
    println!("ollama_workflow_e2e_ok=true");
    Ok(())
}
