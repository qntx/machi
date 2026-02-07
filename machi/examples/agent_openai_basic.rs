//! Basic agent example using OpenAI.
//!
//! Demonstrates the simplest possible agent: a single agent with
//! instructions, a model, and a provider.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_basic
//! ```

#![allow(clippy::print_stdout)]

use machi::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("assistant")
        .instructions("You are a helpful assistant. Keep answers concise.")
        .model("gpt-4o-mini")
        .provider(provider);

    let result = agent
        .run("What is the capital of France?", RunConfig::default())
        .await?;
    println!("{}", result.output);

    Ok(())
}
