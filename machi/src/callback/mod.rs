//! Callback hooks for agent lifecycle events.
//!
//! This module implements a dual-layer async hooks system inspired by the
//! [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) pattern:
//!
//! - **[`RunHooks`]** — Global run-level hooks that observe all agents in a run.
//! - **[`AgentHooks`]** — Per-agent hooks bound to a specific agent instance.
//!
//! Both layers are triggered in parallel (via `tokio::join!`) at each lifecycle
//! point, following the same design as `asyncio.gather` in the Python SDK.
//!
//! # Architecture
//!
//! **`RunHooks`** (global, observes ALL agents):
//!
//! - `on_agent_start` / `on_agent_end`
//! - `on_llm_start` / `on_llm_end`
//! - `on_tool_start` / `on_tool_end`
//! - `on_handoff`
//! - `on_error`
//!
//! **`AgentHooks`** (per-agent, one instance per agent):
//!
//! - `on_start` / `on_end`
//! - `on_llm_start` / `on_llm_end`
//! - `on_tool_start` / `on_tool_end`
//! - `on_handoff`
//! - `on_error`
//!
//! At each lifecycle point, both layers fire in parallel via `tokio::join!`:
//!
//! > `RunHooks::on_<event>` + `AgentHooks::on_<event>` → concurrent execution
//!
//! # Provided Implementations
//!
//! | Type | Description |
//! |------|-------------|
//! | [`NoopRunHooks`] / [`NoopAgentHooks`] | Zero-overhead no-op (default) |
//! | [`LoggingRunHooks`] / [`LoggingAgentHooks`] | Structured logging via `tracing` |
//!
//! # Quick Start
//!
//! ```rust
//! use async_trait::async_trait;
//! use serde_json::Value;
//! use machi::callback::{RunHooks, AgentHooks, RunContext};
//!
//! // 1. Implement RunHooks for global observation
//! struct MyRunHooks;
//!
//! #[async_trait]
//! impl RunHooks for MyRunHooks {
//!     async fn on_agent_start(&self, ctx: &RunContext, agent_name: &str) {
//!         println!("[step {}] {agent_name} started", ctx.step());
//!     }
//! }
//!
//! // 2. Implement AgentHooks for per-agent observation
//! struct MyAgentHooks;
//!
//! #[async_trait]
//! impl AgentHooks for MyAgentHooks {
//!     async fn on_tool_end(&self, _ctx: &RunContext, tool_name: &str, result: &str) {
//!         println!("Tool {tool_name} → {result}");
//!     }
//! }
//! ```

mod context;
mod hooks;
mod logging;
mod noop;

pub use context::RunContext;
pub use hooks::{
    AgentHooks, BoxedAgentHooks, BoxedRunHooks, RunHooks, SharedAgentHooks, SharedRunHooks,
};
pub use logging::{LogLevel, LoggingAgentHooks, LoggingRunHooks};
pub use noop::{NoopAgentHooks, NoopRunHooks};
