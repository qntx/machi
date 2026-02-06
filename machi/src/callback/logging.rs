//! Tracing-based logging implementations of hook traits.
//!
//! Provides [`LoggingRunHooks`] and [`LoggingAgentHooks`] that emit structured
//! log events via the `tracing` crate at configurable verbosity levels.
//!
//! # Example
//!
//! ```rust
//! use machi::callback::{LoggingRunHooks, LoggingAgentHooks, LogLevel};
//!
//! // Default: logs at INFO level
//! let run_hooks = LoggingRunHooks::new();
//!
//! // Custom: logs at DEBUG level
//! let agent_hooks = LoggingAgentHooks::with_level(LogLevel::Debug);
//! ```

use async_trait::async_trait;
use serde_json::Value;

use crate::chat::ChatResponse;
use crate::error::Error;
use crate::message::Message;

use super::context::RunContext;
use super::hooks::{AgentHooks, RunHooks};

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

/// A [`RunHooks`] implementation that logs lifecycle events via `tracing`.
///
/// Emits structured log events for every hook call, including agent name,
/// step number, tool name, and usage statistics.
///
/// # Example
///
/// ```rust
/// use machi::callback::LoggingRunHooks;
///
/// let hooks = LoggingRunHooks::new();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LoggingRunHooks {
    level: LogLevel,
}

impl LoggingRunHooks {
    /// Create a new logging run hooks with the default log level (INFO).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create logging run hooks with a custom log level.
    #[must_use]
    pub const fn with_level(level: LogLevel) -> Self {
        Self { level }
    }
}

#[async_trait]
impl RunHooks for LoggingRunHooks {
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

    async fn on_handoff(&self, ctx: &RunContext, from_agent: &str, to_agent: &str) {
        log_at_level!(
            self.level,
            from = from_agent,
            to = to_agent,
            step = ctx.step(),
            "Agent handoff"
        );
    }

    async fn on_error(&self, ctx: &RunContext, agent_name: &str, error: &Error) {
        // Errors always log at WARN or above regardless of configured level.
        tracing::warn!(
            agent = agent_name,
            step = ctx.step(),
            error = %error,
            "Agent error"
        );
    }
}

/// A [`AgentHooks`] implementation that logs lifecycle events via `tracing`.
///
/// Similar to [`LoggingRunHooks`] but designed for per-agent attachment.
/// Does not include agent name in log output (it is implicit from the
/// agent the hooks are bound to).
///
/// # Example
///
/// ```rust
/// use machi::callback::LoggingAgentHooks;
///
/// let hooks = LoggingAgentHooks::new();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LoggingAgentHooks {
    level: LogLevel,
}

impl LoggingAgentHooks {
    /// Create a new logging agent hooks with the default log level (INFO).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create logging agent hooks with a custom log level.
    #[must_use]
    pub const fn with_level(level: LogLevel) -> Self {
        Self { level }
    }
}

#[async_trait]
impl AgentHooks for LoggingAgentHooks {
    async fn on_start(&self, ctx: &RunContext) {
        log_at_level!(self.level, step = ctx.step(), "Agent started");
    }

    async fn on_end(&self, ctx: &RunContext, output: &Value) {
        let usage = ctx.usage();
        log_at_level!(self.level,
            step = ctx.step(),
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            output = %output,
            "Agent completed"
        );
    }

    async fn on_llm_start(
        &self,
        ctx: &RunContext,
        _system_prompt: Option<&str>,
        messages: &[Message],
    ) {
        log_at_level!(
            self.level,
            step = ctx.step(),
            message_count = messages.len(),
            "LLM request started"
        );
    }

    async fn on_llm_end(&self, ctx: &RunContext, response: &ChatResponse) {
        let model = response.model.as_deref().unwrap_or("unknown");
        log_at_level!(self.level,
            step = ctx.step(),
            model = model,
            stop_reason = ?response.stop_reason,
            "LLM request completed"
        );
    }

    async fn on_tool_start(&self, ctx: &RunContext, tool_name: &str) {
        log_at_level!(
            self.level,
            step = ctx.step(),
            tool = tool_name,
            "Tool execution started"
        );
    }

    async fn on_tool_end(&self, ctx: &RunContext, tool_name: &str, result: &str) {
        log_at_level!(
            self.level,
            step = ctx.step(),
            tool = tool_name,
            result_len = result.len(),
            "Tool execution completed"
        );
    }

    async fn on_handoff(&self, ctx: &RunContext, to_agent: &str) {
        log_at_level!(
            self.level,
            to = to_agent,
            step = ctx.step(),
            "Agent handoff"
        );
    }

    async fn on_error(&self, ctx: &RunContext, error: &Error) {
        tracing::warn!(
            step = ctx.step(),
            error = %error,
            "Agent error"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use crate::callback::context::RunContext;
    use crate::callback::hooks::{AgentHooks, BoxedAgentHooks, BoxedRunHooks, RunHooks};
    use crate::chat::ChatResponse;
    use crate::error::Error;
    use crate::message::Message;
    use crate::usage::Usage;

    /// Helper to create a minimal [`ChatResponse`] for testing.
    fn test_response() -> ChatResponse {
        let mut resp = ChatResponse::new(Message::assistant("test"));
        resp.model = Some("gpt-4o".to_owned());
        resp.usage = Some(Usage::new(10, 20));
        resp
    }

    mod log_level {
        use super::*;

        #[test]
        fn default_is_info() {
            assert_eq!(LogLevel::default(), LogLevel::Info);
        }

        #[test]
        fn eq_works() {
            assert_eq!(LogLevel::Trace, LogLevel::Trace);
            assert_eq!(LogLevel::Debug, LogLevel::Debug);
            assert_ne!(LogLevel::Info, LogLevel::Warn);
        }

        #[test]
        fn clone_and_copy() {
            let level = LogLevel::Debug;
            let cloned = level;
            let copied = level;
            assert_eq!(cloned, LogLevel::Debug);
            assert_eq!(copied, LogLevel::Debug);
        }

        #[test]
        fn debug_impl() {
            let debug_str = format!("{:?}", LogLevel::Trace);
            assert!(debug_str.contains("Trace"));
        }
    }

    mod logging_run_hooks {
        use super::*;

        #[test]
        fn new_uses_default_level() {
            let hooks = LoggingRunHooks::new();
            assert_eq!(hooks.level, LogLevel::Info);
        }

        #[test]
        fn with_level_sets_custom() {
            let hooks = LoggingRunHooks::with_level(LogLevel::Debug);
            assert_eq!(hooks.level, LogLevel::Debug);
        }

        #[test]
        fn default_creates_info() {
            let hooks = LoggingRunHooks::default();
            assert_eq!(hooks.level, LogLevel::Info);
        }

        #[test]
        fn debug_impl() {
            let hooks = LoggingRunHooks::new();
            let debug_str = format!("{hooks:?}");
            assert!(debug_str.contains("LoggingRunHooks"));
        }

        #[test]
        fn clone_and_copy() {
            let hooks = LoggingRunHooks::with_level(LogLevel::Warn);
            let cloned = hooks;
            let copied = hooks;
            assert_eq!(cloned.level, LogLevel::Warn);
            assert_eq!(copied.level, LogLevel::Warn);
        }

        #[test]
        fn into_boxed() {
            let _: BoxedRunHooks = Box::new(LoggingRunHooks::new());
        }

        #[tokio::test]
        async fn all_hooks_execute_without_panic() {
            // Use Trace level so all log_at_level! branches are exercised.
            let hooks = LoggingRunHooks::with_level(LogLevel::Trace);
            let ctx = RunContext::new()
                .with_agent_name("test")
                .with_step(1)
                .with_usage(Usage::new(50, 30));

            let output = serde_json::json!({"answer": 42});
            let response = test_response();
            let messages = vec![Message::system("sys"), Message::user("hello")];
            let error = Error::agent("something went wrong");

            hooks.on_agent_start(&ctx, "test").await;
            hooks.on_agent_end(&ctx, "test", &output).await;
            hooks
                .on_llm_start(&ctx, "test", Some("system prompt"), &messages)
                .await;
            hooks.on_llm_end(&ctx, "test", &response).await;
            hooks.on_tool_start(&ctx, "test", "calculator").await;
            hooks
                .on_tool_end(&ctx, "test", "calculator", "result: 42")
                .await;
            hooks.on_handoff(&ctx, "agent_a", "agent_b").await;
            hooks.on_error(&ctx, "test", &error).await;
        }

        #[tokio::test]
        async fn hooks_at_each_log_level() {
            let ctx = RunContext::new().with_agent_name("lvl");

            for level in [
                LogLevel::Trace,
                LogLevel::Debug,
                LogLevel::Info,
                LogLevel::Warn,
            ] {
                let hooks = LoggingRunHooks::with_level(level);
                hooks.on_agent_start(&ctx, "lvl").await;
                hooks.on_tool_end(&ctx, "lvl", "tool", "ok").await;
            }
        }

        #[tokio::test]
        async fn llm_end_with_no_model_or_usage() {
            let hooks = LoggingRunHooks::new();
            let ctx = RunContext::new();
            // Response with no model and no usage.
            let response = ChatResponse::new(Message::assistant("bare"));
            hooks.on_llm_end(&ctx, "agent", &response).await;
        }
    }

    mod logging_agent_hooks {
        use super::*;

        #[test]
        fn new_uses_default_level() {
            let hooks = LoggingAgentHooks::new();
            assert_eq!(hooks.level, LogLevel::Info);
        }

        #[test]
        fn with_level_sets_custom() {
            let hooks = LoggingAgentHooks::with_level(LogLevel::Trace);
            assert_eq!(hooks.level, LogLevel::Trace);
        }

        #[test]
        fn default_creates_info() {
            let hooks = LoggingAgentHooks::default();
            assert_eq!(hooks.level, LogLevel::Info);
        }

        #[test]
        fn debug_impl() {
            let hooks = LoggingAgentHooks::new();
            let debug_str = format!("{hooks:?}");
            assert!(debug_str.contains("LoggingAgentHooks"));
        }

        #[test]
        fn clone_and_copy() {
            let hooks = LoggingAgentHooks::with_level(LogLevel::Debug);
            let cloned = hooks;
            let copied = hooks;
            assert_eq!(cloned.level, LogLevel::Debug);
            assert_eq!(copied.level, LogLevel::Debug);
        }

        #[test]
        fn into_boxed() {
            let _: BoxedAgentHooks = Box::new(LoggingAgentHooks::new());
        }

        #[tokio::test]
        async fn all_hooks_execute_without_panic() {
            let hooks = LoggingAgentHooks::with_level(LogLevel::Trace);
            let ctx = RunContext::new()
                .with_step(2)
                .with_usage(Usage::new(100, 60));

            let output = serde_json::json!("done");
            let response = test_response();
            let messages = vec![Message::user("question")];
            let error = Error::agent("agent failure");

            hooks.on_start(&ctx).await;
            hooks.on_end(&ctx, &output).await;
            hooks
                .on_llm_start(&ctx, Some("system prompt"), &messages)
                .await;
            hooks.on_llm_end(&ctx, &response).await;
            hooks.on_tool_start(&ctx, "search").await;
            hooks.on_tool_end(&ctx, "search", "found 3 results").await;
            hooks.on_handoff(&ctx, "other_agent").await;
            hooks.on_error(&ctx, &error).await;
        }

        #[tokio::test]
        async fn hooks_at_each_log_level() {
            let ctx = RunContext::new();

            for level in [
                LogLevel::Trace,
                LogLevel::Debug,
                LogLevel::Info,
                LogLevel::Warn,
            ] {
                let hooks = LoggingAgentHooks::with_level(level);
                hooks.on_start(&ctx).await;
                hooks.on_tool_end(&ctx, "tool", "ok").await;
            }
        }

        #[tokio::test]
        async fn llm_end_with_no_model() {
            let hooks = LoggingAgentHooks::new();
            let ctx = RunContext::new();
            let response = ChatResponse::new(Message::assistant("bare"));
            hooks.on_llm_end(&ctx, &response).await;
        }

        #[tokio::test]
        async fn llm_start_with_none_system_prompt() {
            let hooks = LoggingAgentHooks::new();
            let ctx = RunContext::new();
            let messages = vec![Message::user("test")];
            hooks.on_llm_start(&ctx, None, &messages).await;
        }
    }
}
