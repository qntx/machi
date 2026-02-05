//! Structured output example using Ollama with JSON schema.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example llm_ollama_structured
//! ```

#![allow(clippy::print_stdout)]

use machi::chat::ResponseFormat;
use machi::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Ollama::with_defaults()?;

    let schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "capital": { "type": "string" },
            "population": { "type": "integer" }
        },
        "required": ["name", "capital", "population"]
    });

    let request = ChatRequest::new("qwen3")
        .user("Tell me about France.")
        .response_format(ResponseFormat::json_schema("country", schema));

    let response = client.chat(&request).await?;
    println!("{}", response.text().unwrap_or_default());

    Ok(())
}
