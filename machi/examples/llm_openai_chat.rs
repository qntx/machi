//! Basic chat example using OpenAI.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example llm_openai_chat
//! ```

#![allow(clippy::print_stdout)]

use machi::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAI::from_env()?;

    let request = ChatRequest::new("gpt-4o-mini")
        .system("You are a helpful assistant.")
        .user("What is the capital of France?");

    let response = client.chat(&request).await?;
    println!("{}", response.text().unwrap_or_default());

    Ok(())
}
