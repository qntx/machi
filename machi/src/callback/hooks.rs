//! Core hook traits for agent lifecycle callbacks.
//!
//! This module defines two async trait hierarchies following the OpenAI Agents SDK
//! dual-layer hooks pattern:
//!
//! - [`RunHooks`]: Global run-level hooks that observe **all** agents in a run.
//! - [`AgentHooks`]: Per-agent hooks bound to a specific agent instance.
//!
//! Both traits use `async_trait` for object safety (`dyn RunHooks`, `dyn AgentHooks`)
//! and provide default no-op implementations for every method, so users only need
//! to override the events they care about.
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

use crate::chat::ChatResponse;
use crate::error::Error;
use crate::message::Message;

use super::context::RunContext;

/// A boxed, thread-safe [`RunHooks`] trait object.
pub type BoxedRunHooks = Box<dyn RunHooks>;

/// A shared, thread-safe [`RunHooks`] trait object.
pub type SharedRunHooks = std::sync::Arc<dyn RunHooks>;

/// A boxed, thread-safe [`AgentHooks`] trait object.
pub type BoxedAgentHooks = Box<dyn AgentHooks>;

/// A shared, thread-safe [`AgentHooks`] trait object.
pub type SharedAgentHooks = std::sync::Arc<dyn AgentHooks>;

/// Global run-level lifecycle hooks.
///
/// Implementations of this trait observe **all** agents within a single run.
/// Every method receives the agent name so listeners can distinguish between
/// agents in multi-agent scenarios.
///
/// All methods have default no-op implementations, allowing users to override
/// only the events they need.
///
/// # Object Safety
///
/// This trait is object-safe and can be used as `Box<dyn RunHooks>` or
/// `Arc<dyn RunHooks>`.
#[async_trait]
pub trait RunHooks: Send + Sync {
    /// Called before the agent begins execution.
    async fn on_agent_start(&self, _ctx: &RunContext, _agent_name: &str) {}

    /// Called after the agent produces a final output.
    async fn on_agent_end(&self, _ctx: &RunContext, _agent_name: &str, _output: &Value) {}

    /// Called just before invoking the LLM.
    ///
    /// `system_prompt` is the current system prompt (if any), and `messages`
    /// contains the full conversation context sent to the model.
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
    ///
    /// `result` contains the tool output as a string (success or error message).
    async fn on_tool_end(
        &self,
        _ctx: &RunContext,
        _agent_name: &str,
        _tool_name: &str,
        _result: &str,
    ) {
    }

    /// Called when control is handed off from one agent to another.
    ///
    /// This is a forward-looking hook for multi-agent orchestration.
    async fn on_handoff(&self, _ctx: &RunContext, _from_agent: &str, _to_agent: &str) {}

    /// Called when an error occurs during the agent run.
    async fn on_error(&self, _ctx: &RunContext, _agent_name: &str, _error: &Error) {}
}

/// Per-agent lifecycle hooks.
///
/// Implementations of this trait are bound to a specific agent instance and
/// only receive events for that agent. Unlike [`RunHooks`], the agent name
/// is implicit (the bound agent) and not passed as a parameter.
///
/// All methods have default no-op implementations.
///
/// # Object Safety
///
/// This trait is object-safe and can be used as `Box<dyn AgentHooks>` or
/// `Arc<dyn AgentHooks>`.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Called before this agent begins execution.
    async fn on_start(&self, _ctx: &RunContext) {}

    /// Called after this agent produces a final output.
    async fn on_end(&self, _ctx: &RunContext, _output: &Value) {}

    /// Called just before invoking the LLM for this agent.
    async fn on_llm_start(
        &self,
        _ctx: &RunContext,
        _system_prompt: Option<&str>,
        _messages: &[Message],
    ) {
    }

    /// Called immediately after the LLM returns a response for this agent.
    async fn on_llm_end(&self, _ctx: &RunContext, _response: &ChatResponse) {}

    /// Called immediately before a tool is invoked by this agent.
    async fn on_tool_start(&self, _ctx: &RunContext, _tool_name: &str) {}

    /// Called immediately after a tool completes for this agent.
    async fn on_tool_end(&self, _ctx: &RunContext, _tool_name: &str, _result: &str) {}

    /// Called when this agent hands off control to another agent.
    async fn on_handoff(&self, _ctx: &RunContext, _to_agent: &str) {}

    /// Called when an error occurs during this agent's execution.
    async fn on_error(&self, _ctx: &RunContext, _error: &Error) {}
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::chat::ChatResponse;
    use crate::message::Message;

    /// Shared counter for tracking how many times each hook is called.
    #[derive(Debug, Default, Clone)]
    struct CallCounter(Arc<AtomicUsize>);

    impl CallCounter {
        fn new() -> Self {
            Self(Arc::new(AtomicUsize::new(0)))
        }

        fn increment(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn count(&self) -> usize {
            self.0.load(Ordering::SeqCst)
        }
    }

    /// A [`RunHooks`] implementation that counts invocations.
    struct CountingRunHooks {
        agent_start: CallCounter,
        agent_end: CallCounter,
        llm_start: CallCounter,
        llm_end: CallCounter,
        tool_start: CallCounter,
        tool_end: CallCounter,
        handoff: CallCounter,
        error: CallCounter,
    }

    impl CountingRunHooks {
        fn new() -> Self {
            Self {
                agent_start: CallCounter::new(),
                agent_end: CallCounter::new(),
                llm_start: CallCounter::new(),
                llm_end: CallCounter::new(),
                tool_start: CallCounter::new(),
                tool_end: CallCounter::new(),
                handoff: CallCounter::new(),
                error: CallCounter::new(),
            }
        }
    }

    #[async_trait]
    impl RunHooks for CountingRunHooks {
        async fn on_agent_start(&self, _ctx: &RunContext, _agent_name: &str) {
            self.agent_start.increment();
        }
        async fn on_agent_end(&self, _ctx: &RunContext, _agent_name: &str, _output: &Value) {
            self.agent_end.increment();
        }
        async fn on_llm_start(
            &self,
            _ctx: &RunContext,
            _agent_name: &str,
            _system_prompt: Option<&str>,
            _messages: &[Message],
        ) {
            self.llm_start.increment();
        }
        async fn on_llm_end(&self, _ctx: &RunContext, _agent_name: &str, _response: &ChatResponse) {
            self.llm_end.increment();
        }
        async fn on_tool_start(&self, _ctx: &RunContext, _agent_name: &str, _tool_name: &str) {
            self.tool_start.increment();
        }
        async fn on_tool_end(
            &self,
            _ctx: &RunContext,
            _agent_name: &str,
            _tool_name: &str,
            _result: &str,
        ) {
            self.tool_end.increment();
        }
        async fn on_handoff(&self, _ctx: &RunContext, _from: &str, _to: &str) {
            self.handoff.increment();
        }
        async fn on_error(&self, _ctx: &RunContext, _agent_name: &str, _error: &Error) {
            self.error.increment();
        }
    }

    /// A [`AgentHooks`] implementation that counts invocations.
    struct CountingAgentHooks {
        start: CallCounter,
        end: CallCounter,
        llm_start: CallCounter,
        llm_end: CallCounter,
        tool_start: CallCounter,
        tool_end: CallCounter,
        handoff: CallCounter,
        error: CallCounter,
    }

    impl CountingAgentHooks {
        fn new() -> Self {
            Self {
                start: CallCounter::new(),
                end: CallCounter::new(),
                llm_start: CallCounter::new(),
                llm_end: CallCounter::new(),
                tool_start: CallCounter::new(),
                tool_end: CallCounter::new(),
                handoff: CallCounter::new(),
                error: CallCounter::new(),
            }
        }
    }

    #[async_trait]
    impl AgentHooks for CountingAgentHooks {
        async fn on_start(&self, _ctx: &RunContext) {
            self.start.increment();
        }
        async fn on_end(&self, _ctx: &RunContext, _output: &Value) {
            self.end.increment();
        }
        async fn on_llm_start(
            &self,
            _ctx: &RunContext,
            _system_prompt: Option<&str>,
            _messages: &[Message],
        ) {
            self.llm_start.increment();
        }
        async fn on_llm_end(&self, _ctx: &RunContext, _response: &ChatResponse) {
            self.llm_end.increment();
        }
        async fn on_tool_start(&self, _ctx: &RunContext, _tool_name: &str) {
            self.tool_start.increment();
        }
        async fn on_tool_end(&self, _ctx: &RunContext, _tool_name: &str, _result: &str) {
            self.tool_end.increment();
        }
        async fn on_handoff(&self, _ctx: &RunContext, _to_agent: &str) {
            self.handoff.increment();
        }
        async fn on_error(&self, _ctx: &RunContext, _error: &Error) {
            self.error.increment();
        }
    }

    /// Helper to create a minimal [`ChatResponse`] for testing.
    fn test_response() -> ChatResponse {
        ChatResponse::new(Message::assistant("test response"))
    }

    mod run_hooks {
        use super::*;

        #[tokio::test]
        async fn all_hooks_called_once() {
            let hooks = CountingRunHooks::new();
            let ctx = RunContext::new().with_agent_name("test");
            let output = serde_json::json!("result");
            let response = test_response();
            let messages = vec![Message::user("hello")];
            let error = Error::agent("test error");

            hooks.on_agent_start(&ctx, "test").await;
            hooks.on_agent_end(&ctx, "test", &output).await;
            hooks
                .on_llm_start(&ctx, "test", Some("system"), &messages)
                .await;
            hooks.on_llm_end(&ctx, "test", &response).await;
            hooks.on_tool_start(&ctx, "test", "my_tool").await;
            hooks.on_tool_end(&ctx, "test", "my_tool", "ok").await;
            hooks.on_handoff(&ctx, "a", "b").await;
            hooks.on_error(&ctx, "test", &error).await;

            assert_eq!(hooks.agent_start.count(), 1);
            assert_eq!(hooks.agent_end.count(), 1);
            assert_eq!(hooks.llm_start.count(), 1);
            assert_eq!(hooks.llm_end.count(), 1);
            assert_eq!(hooks.tool_start.count(), 1);
            assert_eq!(hooks.tool_end.count(), 1);
            assert_eq!(hooks.handoff.count(), 1);
            assert_eq!(hooks.error.count(), 1);
        }

        #[tokio::test]
        async fn multiple_invocations_accumulate() {
            let hooks = CountingRunHooks::new();
            let ctx = RunContext::new();

            for _ in 0..5 {
                hooks.on_tool_start(&ctx, "agent", "tool").await;
                hooks.on_tool_end(&ctx, "agent", "tool", "ok").await;
            }

            assert_eq!(hooks.tool_start.count(), 5);
            assert_eq!(hooks.tool_end.count(), 5);
        }

        #[tokio::test]
        async fn object_safety_boxed() {
            let hooks: BoxedRunHooks = Box::new(CountingRunHooks::new());
            let ctx = RunContext::new();
            hooks.on_agent_start(&ctx, "test").await;
            // Compiles and runs — proves object safety.
        }

        #[tokio::test]
        async fn object_safety_arc() {
            let hooks: SharedRunHooks = Arc::new(CountingRunHooks::new());
            let ctx = RunContext::new();
            hooks.on_agent_start(&ctx, "test").await;
            // Compiles and runs — proves object safety with Arc.
        }

        #[tokio::test]
        async fn parallel_invocation_with_tokio_join() {
            let run_hooks = Arc::new(CountingRunHooks::new());
            let agent_hooks = Arc::new(CountingAgentHooks::new());
            let ctx = RunContext::new().with_agent_name("test");

            // Simulate parallel invocation like OpenAI Agents SDK's asyncio.gather
            let (rh, ah) = (Arc::clone(&run_hooks), Arc::clone(&agent_hooks));
            tokio::join!(rh.on_agent_start(&ctx, "test"), ah.on_start(&ctx),);

            assert_eq!(run_hooks.agent_start.count(), 1);
            assert_eq!(agent_hooks.start.count(), 1);
        }
    }

    mod agent_hooks {
        use super::*;

        #[tokio::test]
        async fn all_hooks_called_once() {
            let hooks = CountingAgentHooks::new();
            let ctx = RunContext::new();
            let output = serde_json::json!("done");
            let response = test_response();
            let messages = vec![Message::user("hello")];
            let error = Error::agent("fail");

            hooks.on_start(&ctx).await;
            hooks.on_end(&ctx, &output).await;
            hooks.on_llm_start(&ctx, Some("system"), &messages).await;
            hooks.on_llm_end(&ctx, &response).await;
            hooks.on_tool_start(&ctx, "tool").await;
            hooks.on_tool_end(&ctx, "tool", "ok").await;
            hooks.on_handoff(&ctx, "other").await;
            hooks.on_error(&ctx, &error).await;

            assert_eq!(hooks.start.count(), 1);
            assert_eq!(hooks.end.count(), 1);
            assert_eq!(hooks.llm_start.count(), 1);
            assert_eq!(hooks.llm_end.count(), 1);
            assert_eq!(hooks.tool_start.count(), 1);
            assert_eq!(hooks.tool_end.count(), 1);
            assert_eq!(hooks.handoff.count(), 1);
            assert_eq!(hooks.error.count(), 1);
        }

        #[tokio::test]
        async fn object_safety_boxed() {
            let hooks: BoxedAgentHooks = Box::new(CountingAgentHooks::new());
            let ctx = RunContext::new();
            hooks.on_start(&ctx).await;
        }

        #[tokio::test]
        async fn object_safety_arc() {
            let hooks: SharedAgentHooks = Arc::new(CountingAgentHooks::new());
            let ctx = RunContext::new();
            hooks.on_start(&ctx).await;
        }

        #[tokio::test]
        async fn llm_start_with_none_system_prompt() {
            let hooks = CountingAgentHooks::new();
            let ctx = RunContext::new();
            let messages = vec![Message::user("test")];

            hooks.on_llm_start(&ctx, None, &messages).await;
            assert_eq!(hooks.llm_start.count(), 1);
        }
    }

    mod type_aliases {
        use super::*;
        use crate::callback::noop::{NoopAgentHooks, NoopRunHooks};

        #[test]
        fn boxed_run_hooks_from_noop() {
            let _: BoxedRunHooks = Box::new(NoopRunHooks);
        }

        #[test]
        fn shared_run_hooks_from_noop() {
            let _: SharedRunHooks = Arc::new(NoopRunHooks);
        }

        #[test]
        fn boxed_agent_hooks_from_noop() {
            let _: BoxedAgentHooks = Box::new(NoopAgentHooks);
        }

        #[test]
        fn shared_agent_hooks_from_noop() {
            let _: SharedAgentHooks = Arc::new(NoopAgentHooks);
        }
    }
}
