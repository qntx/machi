//! Agent with lifecycle callbacks example using Ollama.
//!
//! Demonstrates custom [`Hooks`] to observe the agent's
//! reasoning loop — start, LLM calls, tool invocations, and end.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_ollama_callback
//! ```

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use async_trait::async_trait;
use machi::prelude::*;
use serde_json::Value;

/// Custom hooks that print lifecycle events to stdout.
struct PrintHooks;

#[async_trait]
impl Hooks for PrintHooks {
    async fn on_agent_start(&self, context: &RunContext, agent_name: &str) {
        println!("[hook] {agent_name} started (step {})", context.step());
    }

    async fn on_llm_start(
        &self,
        _context: &RunContext,
        _agent_name: &str,
        _system: Option<&str>,
        _messages: &[Message],
    ) {
        println!("[hook] LLM call starting...");
    }

    async fn on_llm_end(&self, _context: &RunContext, _agent_name: &str, response: &ChatResponse) {
        println!(
            "[hook] LLM call done — {} tokens",
            response.usage.map_or(0, |u| u.total_tokens),
        );
    }

    async fn on_agent_end(&self, _context: &RunContext, _agent_name: &str, output: &Value) {
        let text = output.as_str().unwrap_or("<structured>");
        println!(
            "[hook] Agent finished — output length: {} chars",
            text.len()
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(Ollama::with_defaults()?);

    let agent = Agent::new("assistant")
        .instructions("You are a helpful assistant. Keep answers concise.")
        .model("qwen3")
        .provider(provider);

    let config = RunConfig::new().hooks(Arc::new(PrintHooks));

    let result = agent.run("What is the speed of light?", config).await?;

    println!("\n{}", result.output);

    Ok(())
}
