//! Basic chat example using Ollama.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example llm_ollama_chat
//! ```

#![allow(clippy::print_stdout)]

use machi::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Ollama::with_defaults()?;

    let request = ChatRequest::new("qwen3")
        .system("You are a helpful assistant.")
        .user("What is the capital of France?");

    let response = client.chat(&request).await?;
    println!("{}", response.text().unwrap_or_default());

    Ok(())
}
