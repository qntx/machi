//! Tracing-based logging implementation of the [`Hooks`] trait.
//!
//! # Example
//!
//! ```rust
//! use machi::hooks::{LoggingHooks, LogLevel};
//!
//! let hooks = LoggingHooks::new();
//! let debug_hooks = LoggingHooks::with_level(LogLevel::Debug);
//! ```

use async_trait::async_trait;
use serde_json::Value;

use super::Hooks;
use super::context::RunContext;
use crate::chat::ChatResponse;
use crate::error::Error;
use crate::message::Message;

/// Log verbosity level for hook events.
///
/// Maps directly to `tracing` levels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LogLevel {
    /// Trace-level logging (most verbose).
    Trace,
    /// Debug-level logging.
    Debug,
    /// Info-level logging (default).
    #[default]
    Info,
    /// Warn-level logging.
    Warn,
}

/// Emit a log event at the specified level using `tracing` macros.
macro_rules! log_at_level {
    ($level:expr, $($arg:tt)*) => {
        match $level {
            LogLevel::Trace => tracing::trace!($($arg)*),
            LogLevel::Debug => tracing::debug!($($arg)*),
            LogLevel::Info  => tracing::info!($($arg)*),
            LogLevel::Warn  => tracing::warn!($($arg)*),
        }
    };
}

/// A [`Hooks`] implementation that logs lifecycle events via `tracing`.
///
/// Emits structured log events for every hook call, including agent name,
/// step number, tool name, and usage statistics.
///
/// # Example
///
/// ```rust
/// use machi::hooks::LoggingHooks;
///
/// let hooks = LoggingHooks::new();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LoggingHooks {
    level: LogLevel,
}

impl LoggingHooks {
    /// Create logging hooks with the default log level (INFO).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create logging hooks with a custom log level.
    #[must_use]
    pub const fn with_level(level: LogLevel) -> Self {
        Self { level }
    }
}

#[async_trait]
impl Hooks for LoggingHooks {
    async fn on_agent_start(&self, ctx: &RunContext, agent_name: &str) {
        log_at_level!(
            self.level,
            agent = agent_name,
            step = ctx.step(),
            "Agent started"
        );
    }

    async fn on_agent_end(&self, ctx: &RunContext, agent_name: &str, output: &Value) {
        let usage = ctx.usage();
        log_at_level!(self.level,
            agent = agent_name,
            step = ctx.step(),
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            total_tokens = usage.total_tokens,
            output = %output,
            "Agent completed"
        );
    }

    async fn on_llm_start(
        &self,
        ctx: &RunContext,
        agent_name: &str,
        _system_prompt: Option<&str>,
        messages: &[Message],
    ) {
        log_at_level!(
            self.level,
            agent = agent_name,
            step = ctx.step(),
            message_count = messages.len(),
            "LLM request started"
        );
    }

    async fn on_llm_end(&self, ctx: &RunContext, agent_name: &str, response: &ChatResponse) {
        let model = response.model.as_deref().unwrap_or("unknown");
        let usage_str = response
            .usage
            .map_or_else(|| "none".to_owned(), |u| u.to_string());
        log_at_level!(self.level,
            agent = agent_name,
            step = ctx.step(),
            model = model,
            usage = %usage_str,
            stop_reason = ?response.stop_reason,
            "LLM request completed"
        );
    }

    async fn on_tool_start(&self, ctx: &RunContext, agent_name: &str, tool_name: &str) {
        log_at_level!(
            self.level,
            agent = agent_name,
            step = ctx.step(),
            tool = tool_name,
            "Tool execution started"
        );
    }

    async fn on_tool_end(&self, ctx: &RunContext, agent_name: &str, tool_name: &str, result: &str) {
        log_at_level!(
            self.level,
            agent = agent_name,
            step = ctx.step(),
            tool = tool_name,
            result_len = result.len(),
            "Tool execution completed"
        );
    }

    async fn on_error(&self, ctx: &RunContext, agent_name: &str, error: &Error) {
        tracing::warn!(
            agent = agent_name,
            step = ctx.step(),
            error = %error,
            "Agent error"
        );
    }
}
