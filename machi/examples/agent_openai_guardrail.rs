//! Agent with input and output guardrails using `OpenAI`.
//!
//! Demonstrates `InputGuardrail` (PII detector, sequential mode) on
//! the agent, and `OutputGuardrail` (length limiter) on the run config.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_guardrail
//! ```

#![allow(clippy::print_stdout, clippy::unused_async)]

use std::sync::Arc;

use async_trait::async_trait;
use machi::prelude::*;

/// Blocks requests that contain personal identifiable information.
struct PiiDetector;

#[async_trait]
impl InputGuardrailCheck for PiiDetector {
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

        let patterns = ["SSN", "social security", "credit card", "passport number"];
        if patterns
            .iter()
            .any(|p| text.to_lowercase().contains(&p.to_lowercase()))
        {
            return Ok(GuardrailOutput::tripwire("PII detected in input"));
        }
        Ok(GuardrailOutput::pass())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("safe-assistant")
        .instructions("You are a helpful assistant. Keep answers concise.")
        .model("gpt-4o-mini")
        .provider(provider)
        .input_guardrail(InputGuardrail::new("pii-detector", PiiDetector).run_in_parallel(false));

    // Blocked: PII in the request triggers the input guardrail.
    match agent
        .run(
            "My SSN is 123-45-6789, look up my records",
            RunConfig::default(),
        )
        .await
    {
        Err(Error::Agent(AgentError::InputGuardrailTriggered { name, info })) => {
            println!("Blocked by '{name}': {info}");
        }
        other => println!("Unexpected: {other:?}"),
    }

    Ok(())
}
