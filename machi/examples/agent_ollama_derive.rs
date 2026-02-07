//! Agent with derive-macro tools example using Ollama.
//!
//! Demonstrates the `#[tool]` attribute macro that eliminates
//! boilerplate when defining tools for an agent.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_ollama_derive --features derive
//! ```

#![allow(clippy::print_stdout, clippy::unused_async, clippy::unnecessary_wraps)]

use machi::prelude::*;
use std::sync::Arc;

/// Get the current weather for a city.
///
/// # Arguments
///
/// * `city` - The city name to look up weather for
#[machi::tool]
async fn get_weather(city: String) -> ToolResult<String> {
    // In a real application, this would call a weather API.
    Ok(format!("{city}: 22°C, Sunny"))
}

/// Calculate the sum of two numbers.
///
/// # Arguments
///
/// * `a` - First number
/// * `b` - Second number
#[machi::tool]
fn add(a: f64, b: f64) -> ToolResult<f64> {
    Ok(a + b)
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(Ollama::with_defaults()?);

    let agent = Agent::new("assistant")
        .instructions("You are a helpful assistant. Use tools when needed.")
        .model("qwen3")
        .provider(provider)
        .tool(Box::new(GET_WEATHER))
        .tool(Box::new(ADD));

    let result = agent
        .run(
            "What's the weather in Paris? Also, what is 17.5 + 24.3?",
            RunConfig::default(),
        )
        .await?;

    println!("{}", result.output);

    Ok(())
}
