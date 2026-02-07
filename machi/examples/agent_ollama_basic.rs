//! Basic agent example using Ollama.
//!
//! Demonstrates the simplest possible agent: a single agent with
//! instructions, a model, and a provider.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_ollama_basic
//! ```

#![allow(clippy::print_stdout)]

use machi::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(Ollama::with_defaults()?);

    let agent = Agent::new("assistant")
        .instructions("You are a helpful assistant. Keep answers concise.")
        .model("qwen3")
        .provider(provider);

    let result = agent.run("What is the capital of France?", RunConfig::default()).await?;
    println!("{}", result.output);

    Ok(())
}
