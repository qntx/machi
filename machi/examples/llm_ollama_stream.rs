//! Streaming chat example using Ollama.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example llm_ollama_stream
//! ```

#![allow(clippy::print_stdout)]

use futures::StreamExt;
use machi::prelude::*;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Ollama::with_defaults()?;

    let request = ChatRequest::new("qwen3").user("Write a haiku about Rust.");

    let mut stream = client.chat_stream(&request).await?;

    while let Some(chunk) = stream.next().await {
        if let StreamChunk::Text(text) = chunk? {
            print!("{text}");
            stdout().flush()?;
        }
    }
    println!();

    Ok(())
}
