//! Agent with tracing / observability example using OpenAI.
//!
//! Demonstrates how to wire up `tracing-subscriber` so that the
//! built-in agent, LLM, and tool spans are printed to stderr.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_tracing
//! ```

#![allow(clippy::print_stdout)]

use machi::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize a tracing subscriber that prints spans + events to stderr.
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter("info")
        .init();

    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("assistant")
        .instructions("You are a helpful assistant. Keep answers concise.")
        .model("gpt-4o-mini")
        .provider(provider);

    let result = agent
        .run("What is the tallest mountain on Earth?", RunConfig::default())
        .await?;

    println!("{}", result.output);

    Ok(())
}
