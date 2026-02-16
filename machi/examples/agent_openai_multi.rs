//! Multi-agent collaboration example using `OpenAI`.
//!
//! Demonstrates managed agents: a parent "coordinator" delegates
//! tasks to specialized sub-agents, each with its own instructions.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_multi
//! ```

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use machi::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing to observe parallel execution via span timestamps.
    // Set RUST_LOG=info (or debug) to see agent/tool spans with timing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .init();

    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    // Sub-agent: writes poetry.
    let poet = Agent::new("poet")
        .description("A poet who writes short poems on any topic.")
        .instructions("You are a poet. Write a short, vivid poem about the given topic.")
        .model("gpt-4o-mini")
        .provider(Arc::clone(&provider));

    // Sub-agent: provides factual summaries.
    let researcher = Agent::new("researcher")
        .description("A researcher who provides concise factual summaries.")
        .instructions(
            "You are a researcher. Provide a brief factual summary about the given topic.",
        )
        .model("gpt-4o-mini")
        .provider(Arc::clone(&provider));

    // Parent agent: coordinates the sub-agents.
    let coordinator = Agent::new("coordinator")
        .instructions(
            "You coordinate tasks. Use the 'poet' agent for creative writing \
             and the 'researcher' agent for factual questions. \
             Combine their outputs into a final response.",
        )
        .model("gpt-4o-mini")
        .provider(provider)
        .managed_agent(poet)
        .managed_agent(researcher);

    let result = coordinator
        .run(
            "Tell me about the Northern Lights — both facts and a poem.",
            RunConfig::default(),
        )
        .await?;

    println!("{}", result.output);
    println!("\nCompleted in {} step(s), {}", result.steps, result.usage);

    Ok(())
}
