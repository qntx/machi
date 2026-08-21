//! Dynamic multi-agent delegation via [`InProcessHost::spawn_agents`] (offline mock).
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::exit,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    unused_crate_dependencies,
    reason = "offline demo binary uses stdout, expect, and exit codes"
)]

use std::sync::Arc;

use ovo::{InProcessHost, MockSampler, SessionHost, SpawnOpts};

#[tokio::main]
async fn main() {
    let sampler = Arc::new(MockSampler::new());
    // Keyed replies keep concurrent spawn order-correct under races.
    sampler.map_user_text("Investigate topic alpha", "research: alpha findings");
    sampler.map_user_text("Investigate topic beta", "research: beta findings");

    let host = InProcessHost::new(sampler, Vec::new()).with_agent_budget(8);
    let results = host
        .spawn_agents(vec![
            SpawnOpts::new("Investigate topic alpha").with_label("worker-alpha"),
            SpawnOpts::new("Investigate topic beta").with_label("worker-beta"),
        ])
        .await
        .expect("delegate workers");

    println!("delegated_workers={}", results.len());
    for r in &results {
        let label = r.label.as_deref().unwrap_or("?");
        let out = r
            .output
            .as_str()
            .map_or_else(|| r.output.to_string(), str::to_owned);
        println!("worker label={label} success={} output={out}", r.success);
    }

    let summary: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "{}:{}",
                r.label.as_deref().unwrap_or("?"),
                r.output.as_str().unwrap_or("")
            )
        })
        .collect();
    println!("aggregate={}", summary.join(" | "));
}
