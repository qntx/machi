//! Agent with streaming output example using Ollama.
//!
//! Demonstrates `Agent::run_streamed()` which yields `RunEvent`s
//! in real-time, enabling progressive display of tokens.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_ollama_streaming
//! ```

#![allow(clippy::print_stdout)]

use futures::StreamExt;
use machi::prelude::*;
use std::io::{Write, stdout};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(Ollama::with_defaults()?);

    let agent = Agent::new("poet")
        .instructions("You are a creative poet. Write vivid, expressive poetry.")
        .model("qwen3")
        .provider(provider);

    let mut stream = agent.run_streamed("Write a haiku about Rust.", RunConfig::default());

    while let Some(event) = stream.next().await {
        match event? {
            RunEvent::TextDelta(text) => {
                print!("{text}");
                stdout().flush()?;
            }
            RunEvent::ReasoningDelta(text) => {
                print!("\x1b[2m{text}\x1b[0m");
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
