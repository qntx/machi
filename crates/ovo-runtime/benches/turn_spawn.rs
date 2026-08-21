//! Smoke benches: single-agent turn + concurrent host spawn (offline mock).
//!
//! ```bash
//! cargo bench -p ovo-runtime --bench turn_spawn
//! ```

#![allow(
    missing_docs,
    unused_crate_dependencies,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "bench harness: transitive deps + timing output"
)]

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use ovo_agent::AgentBuilder;
use ovo_llm::MockSampler;
use ovo_runtime::{
    InProcessHost, SessionHost, SpawnOpts, TurnInput, TurnOptions, TurnRuntime,
    VecConversationState,
};

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");

    rt.block_on(async {
        bench_turn(2_000).await;
        bench_spawn(500).await;
    });
}

async fn bench_turn(iters: u32) {
    let sampler = Arc::new(MockSampler::new());
    for i in 0..iters {
        sampler.map_user_text(format!("u{i}"), "ok");
    }
    let agent = AgentBuilder::named("bench")
        .model("mock")
        .build()
        .expect("agent");
    let runtime = TurnRuntime::new();
    let mut state = VecConversationState::new();

    let start = Instant::now();
    for i in 0..iters {
        let out = runtime
            .run(
                &agent,
                sampler.as_ref(),
                &mut state,
                TurnInput::Text(format!("u{i}")),
                TurnOptions::default(),
            )
            .await
            .expect("turn");
        black_box(out.output_text);
        state = VecConversationState::new();
    }
    let elapsed = start.elapsed();
    let per = elapsed / iters;
    println!(
        "turn_basic: {iters} iters in {elapsed:?} ({per:?}/iter, {:.0} turns/s)",
        f64::from(iters) / elapsed.as_secs_f64()
    );
}

async fn bench_spawn(iters: u32) {
    let sampler = Arc::new(MockSampler::new());
    for i in 0..iters {
        sampler.map_user_text(format!("s{i}"), "child-ok");
    }
    let host = InProcessHost::new(sampler, vec![])
        .with_agent_budget(u64::from(iters) + 8)
        .with_max_concurrent_children(Some(32));

    let start = Instant::now();
    for i in 0..iters {
        let run = host
            .spawn_agent(SpawnOpts::new(format!("s{i}")))
            .await
            .expect("spawn");
        black_box(run.output);
    }
    let elapsed = start.elapsed();
    let per = elapsed / iters;
    println!(
        "host_spawn: {iters} iters in {elapsed:?} ({per:?}/iter, {:.0} spawns/s)",
        f64::from(iters) / elapsed.as_secs_f64()
    );
}
