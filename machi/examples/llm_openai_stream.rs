//! Streaming chat example using OpenAI.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example llm_openai_stream
//! ```

#![allow(clippy::print_stdout)]

use futures::StreamExt;
use machi::prelude::*;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAI::from_env()?;

    let request = ChatRequest::new("gpt-4o-mini").user("Write a haiku about Rust.");

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
