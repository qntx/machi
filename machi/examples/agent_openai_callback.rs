//! Agent with lifecycle callbacks example using OpenAI.
//!
//! Demonstrates custom `AgentHooks` to observe the agent's
//! reasoning loop — start, LLM calls, tool invocations, and end.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_callback
//! ```

#![allow(clippy::print_stdout)]

use async_trait::async_trait;
use machi::prelude::*;
use serde_json::Value;
use std::sync::Arc;

/// Custom hooks that print lifecycle events to stdout.
struct PrintHooks;

#[async_trait]
impl AgentHooks for PrintHooks {
    async fn on_start(&self, context: &RunContext) {
        println!("[hook] Agent started (step {})", context.step());
    }

    async fn on_llm_start(&self, _context: &RunContext, _system: Option<&str>, _messages: &[Message]) {
        println!("[hook] LLM call starting...");
    }

    async fn on_llm_end(&self, _context: &RunContext, response: &ChatResponse) {
        println!(
            "[hook] LLM call done — {} tokens",
            response.usage.map_or(0, |u| u.total_tokens),
        );
    }

    async fn on_end(&self, _context: &RunContext, output: &Value) {
        let text = output.as_str().unwrap_or("<structured>");
        println!("[hook] Agent finished — output length: {} chars", text.len());
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("assistant")
        .instructions("You are a helpful assistant. Keep answers concise.")
        .model("gpt-4o-mini")
        .provider(provider)
        .hooks(Arc::new(PrintHooks));

    let result = agent
        .run("What is the speed of light?", RunConfig::default())
        .await?;

    println!("\n{}", result.output);

    Ok(())
}
