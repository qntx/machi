//! Minimal streaming example.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_simple
//! ```

#![allow(clippy::print_stdout)]

use futures::StreamExt;
use machi::prelude::*;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() {
    let model = OllamaClient::new().completion_model("qwen3");
    let mut agent = Agent::builder().model(model).build();

    let mut stream = std::pin::pin!(agent.stream("What is 2+2?"));

    while let Some(Ok(event)) = stream.next().await {
        match event {
            StreamEvent::TextDelta(t) => { print!("{t}"); let _ = stdout().flush(); }
            StreamEvent::FinalAnswer { answer } => println!("\n=> {answer}"),
            _ => {}
        }
    }
}
