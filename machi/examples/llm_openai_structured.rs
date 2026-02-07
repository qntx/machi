//! Structured output example using OpenAI with JSON schema.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example llm_openai_structured
//! ```

#![allow(clippy::print_stdout)]

use machi::chat::ResponseFormat;
use machi::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAI::from_env()?;

    let schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "capital": { "type": "string" },
            "population": { "type": "integer" }
        },
        "required": ["name", "capital", "population"]
    });

    let request = ChatRequest::new("gpt-4o-mini")
        .user("Tell me about France.")
        .response_format(ResponseFormat::json_schema("country", schema));

    let response = client.chat(&request).await?;
    println!("{}", response.text().unwrap_or_default());

    Ok(())
}
