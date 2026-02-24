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
//! use machi::callback::{Hooks, RunContext};
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
mod hooks;
mod logging;
mod noop;

pub use context::RunContext;
pub use hooks::{Hooks, SharedHooks};
pub use logging::{LogLevel, LoggingHooks};
pub use noop::NoopHooks;
