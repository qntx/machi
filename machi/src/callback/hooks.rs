//! Unified hook trait for agent lifecycle callbacks.
//!
//! [`Hooks`] provides a single observation layer for all agent lifecycle events.
//! Every method receives the agent name, enabling multi-agent observability
//! with a single implementation.
//!
//! # Lifecycle Events
//!
//! 1. **`on_agent_start`** — agent begins execution
//! 2. **Step loop** (repeats until done):
//!    - `on_llm_start` → *LLM call* → `on_llm_end`
//!    - `on_tool_start` → *tool execution* → `on_tool_end`
//! 3. **`on_agent_end`** — agent produces final output, or **`on_error`** on failure

use async_trait::async_trait;
use serde_json::Value;

use super::context::RunContext;
use crate::chat::ChatResponse;
use crate::error::Error;
use crate::message::Message;

/// A shared, thread-safe [`Hooks`] trait object.
pub type SharedHooks = std::sync::Arc<dyn Hooks>;

/// Lifecycle hooks for observing agent execution.
///
/// Every method receives `agent_name` so a single implementation can
/// distinguish between agents in multi-agent scenarios. All methods
/// have default no-op implementations — override only the events you need.
///
/// # Object Safety
///
/// This trait is object-safe and can be used as `Arc<dyn Hooks>`.
#[async_trait]
pub trait Hooks: Send + Sync {
    /// Called before the agent begins execution.
    async fn on_agent_start(&self, _ctx: &RunContext, _agent_name: &str) {}

    /// Called after the agent produces a final output.
    async fn on_agent_end(&self, _ctx: &RunContext, _agent_name: &str, _output: &Value) {}

    /// Called just before invoking the LLM.
    async fn on_llm_start(
        &self,
        _ctx: &RunContext,
        _agent_name: &str,
        _system_prompt: Option<&str>,
        _messages: &[Message],
    ) {
    }

    /// Called immediately after the LLM returns a response.
    async fn on_llm_end(&self, _ctx: &RunContext, _agent_name: &str, _response: &ChatResponse) {}

    /// Called immediately before a tool is invoked.
    async fn on_tool_start(&self, _ctx: &RunContext, _agent_name: &str, _tool_name: &str) {}

    /// Called immediately after a tool completes.
    async fn on_tool_end(
        &self,
        _ctx: &RunContext,
        _agent_name: &str,
        _tool_name: &str,
        _result: &str,
    ) {
    }

    /// Called when an error occurs during the agent run.
    async fn on_error(&self, _ctx: &RunContext, _agent_name: &str, _error: &Error) {}
}
