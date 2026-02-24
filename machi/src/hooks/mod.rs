//! Lifecycle hooks for observing agent execution.
//!
//! [`Hooks`] is a single trait that observes all agents in a run. Every method
//! receives the agent name, enabling multi-agent observability with one impl.
//!
//! # Provided Implementations
//!
//! | Type | Description |
//! |------|-------------|
//! | [`NoopHooks`] | Zero-overhead no-op (default) |
//! | [`LoggingHooks`] | Structured logging via `tracing` |
//!
//! # Quick Start
//!
//! ```rust
//! use async_trait::async_trait;
//! use serde_json::Value;
//! use machi::hooks::{Hooks, RunContext};
//!
//! struct MyHooks;
//!
//! #[async_trait]
//! impl Hooks for MyHooks {
//!     async fn on_agent_start(&self, ctx: &RunContext, agent_name: &str) {
//!         println!("[step {}] {agent_name} started", ctx.step());
//!     }
//! }
//! ```

mod context;
mod logging;

use async_trait::async_trait;
pub use context::RunContext;
pub use logging::{LogLevel, LoggingHooks};
use serde_json::Value;

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

/// A zero-sized no-op implementation of [`Hooks`].
///
/// All methods are inherited from the trait defaults (empty bodies).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopHooks;

#[async_trait]
impl Hooks for NoopHooks {}
