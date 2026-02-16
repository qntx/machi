//! Agent with input and output guardrails using Ollama.
//!
//! Demonstrates `InputGuardrail` and `OutputGuardrail` to validate
//! user input (topic filter) and agent output (placeholder detector).
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_ollama_guardrail
//! ```

#![allow(clippy::print_stdout, clippy::unused_async)]

use std::sync::Arc;

use async_trait::async_trait;
use machi::prelude::*;

/// Blocks requests that are not about programming or technology.
struct TopicFilter;

#[async_trait]
impl InputGuardrailCheck for TopicFilter {
    async fn check(
        &self,
        _context: &RunContext,
        _agent_name: &str,
        input: &[Message],
    ) -> Result<GuardrailOutput> {
        let text: String = input
            .iter()
            .filter(|m| m.role == Role::User)
            .filter_map(Message::text)
            .collect();

        let blocked = ["recipe", "cooking", "food", "weather forecast"];
        if blocked.iter().any(|kw| text.to_lowercase().contains(kw)) {
            return Ok(GuardrailOutput::tripwire("Off-topic request"));
        }
        Ok(GuardrailOutput::pass())
    }
}

/// Rejects responses containing placeholder text.
struct PlaceholderDetector;

#[async_trait]
impl OutputGuardrailCheck for PlaceholderDetector {
    async fn check(
        &self,
        _context: &RunContext,
        _agent_name: &str,
        output: &serde_json::Value,
    ) -> Result<GuardrailOutput> {
        let text = output.as_str().unwrap_or_default();
        if text.contains("TODO") || text.contains("FIXME") {
            return Ok(GuardrailOutput::tripwire("Placeholder text in response"));
        }
        Ok(GuardrailOutput::pass())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(Ollama::with_defaults()?);

    let agent = Agent::new("tech-assistant")
        .instructions("You are a programming assistant. Answer concisely.")
        .model("qwen3")
        .provider(provider)
        .input_guardrail(InputGuardrail::new("topic-filter", TopicFilter))
        .output_guardrail(OutputGuardrail::new(
            "placeholder-detector",
            PlaceholderDetector,
        ));

    // Allowed: programming question.
    let result = agent
        .run("What is a trait in Rust?", RunConfig::default())
        .await?;
    println!("{}", result.output);

    // Blocked: off-topic request triggers the input guardrail.
    match agent
        .run("Give me a cookie recipe", RunConfig::default())
        .await
    {
        Err(Error::Agent(AgentError::InputGuardrailTriggered { name, info })) => {
            println!("\nBlocked by '{name}': {info}");
        }
        other => println!("\nUnexpected: {other:?}"),
    }

    Ok(())
}
