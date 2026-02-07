//! Agent with streaming output example using OpenAI.
//!
//! Demonstrates `Agent::run_streamed()` which yields `RunEvent`s
//! in real-time, enabling progressive display of tokens.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_streaming
//! ```

#![allow(clippy::print_stdout)]

use futures::StreamExt;
use machi::prelude::*;
use std::io::{Write, stdout};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("poet")
        .instructions("You are a creative poet. Write vivid, expressive poetry.")
        .model("gpt-4o-mini")
        .provider(provider);

    let mut stream = agent.run_streamed("Write a haiku about Rust.", RunConfig::default());

    while let Some(event) = stream.next().await {
        match event? {
            RunEvent::TextDelta(text) => {
                print!("{text}");
                stdout().flush()?;
            }
            RunEvent::RunCompleted { result } => {
                println!("\n\nCompleted in {} step(s), {}", result.steps, result.usage);
            }
            _ => {}
        }
    }

    Ok(())
}
