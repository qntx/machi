//! No-op (empty) implementations of hook traits.
//!
//! These types provide zero-overhead default implementations for both
//! [`RunHooks`] and [`AgentHooks`]. They are used as the default when
//! no custom hooks are configured.
//!
//! # Design
//!
//! Since both traits already have default no-op implementations for every
//! method, `NoopRunHooks` and `NoopAgentHooks` simply rely on those defaults.
//! The structs exist as concrete types so they can be used as default values
//! in builder patterns and `Option<BoxedRunHooks>` scenarios.

use async_trait::async_trait;

use super::hooks::{AgentHooks, RunHooks};

/// A no-op implementation of [`RunHooks`] that does nothing.
///
/// This is the default hook used when no custom run hooks are configured.
/// All methods are inherited from the trait defaults (empty bodies).
///
/// # Example
///
/// ```rust
/// use machi::callback::NoopRunHooks;
///
/// let hooks = NoopRunHooks;
/// // All hook methods do nothing — zero overhead.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopRunHooks;

#[async_trait]
impl RunHooks for NoopRunHooks {}

/// A no-op implementation of [`AgentHooks`] that does nothing.
///
/// This is the default hook used when no custom agent hooks are configured.
/// All methods are inherited from the trait defaults (empty bodies).
///
/// # Example
///
/// ```rust
/// use machi::callback::NoopAgentHooks;
///
/// let hooks = NoopAgentHooks;
/// // All hook methods do nothing — zero overhead.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopAgentHooks;

#[async_trait]
impl AgentHooks for NoopAgentHooks {}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::callback::context::RunContext;
    use crate::callback::hooks::{
        BoxedAgentHooks, BoxedRunHooks, SharedAgentHooks, SharedRunHooks,
    };
    use crate::chat::ChatResponse;
    use crate::error::Error;
    use crate::message::Message;

    /// Helper to create a minimal [`ChatResponse`] for testing.
    fn test_response() -> ChatResponse {
        ChatResponse::new(Message::assistant("noop"))
    }

    mod noop_run_hooks {
        use std::mem::size_of_val;

        use super::*;

        #[test]
        fn debug_impl() {
            let hooks = NoopRunHooks;
            let debug_str = format!("{hooks:?}");
            assert!(debug_str.contains("NoopRunHooks"));
        }

        #[test]
        fn clone_and_copy() {
            let hooks = NoopRunHooks;
            let cloned = hooks;
            let copied = hooks;
            // All are identical zero-sized types.
            assert_eq!(size_of_val(&hooks), 0);
            assert_eq!(size_of_val(&cloned), 0);
            assert_eq!(size_of_val(&copied), 0);
        }

        #[test]
        fn default_creates_instance() {
            let hooks = NoopRunHooks;
            assert_eq!(size_of_val(&hooks), 0);
        }

        #[tokio::test]
        async fn all_hooks_are_noop() {
            let hooks = NoopRunHooks;
            let ctx = RunContext::new();
            let output = serde_json::json!("test");
            let response = test_response();
            let messages = vec![Message::user("hello")];
            let error = Error::agent("err");

            // All these should complete without panicking or side effects.
            hooks.on_agent_start(&ctx, "agent").await;
            hooks.on_agent_end(&ctx, "agent", &output).await;
            hooks
                .on_llm_start(&ctx, "agent", Some("sys"), &messages)
                .await;
            hooks.on_llm_end(&ctx, "agent", &response).await;
            hooks.on_tool_start(&ctx, "agent", "tool").await;
            hooks.on_tool_end(&ctx, "agent", "tool", "ok").await;
            hooks.on_handoff(&ctx, "a", "b").await;
            hooks.on_error(&ctx, "agent", &error).await;
        }

        #[test]
        fn into_boxed() {
            let _: BoxedRunHooks = Box::new(NoopRunHooks);
        }

        #[test]
        fn into_shared() {
            let _: SharedRunHooks = Arc::new(NoopRunHooks);
        }
    }

    mod noop_agent_hooks {
        use std::mem::size_of_val;

        use super::*;

        #[test]
        fn debug_impl() {
            let hooks = NoopAgentHooks;
            let debug_str = format!("{hooks:?}");
            assert!(debug_str.contains("NoopAgentHooks"));
        }

        #[test]
        fn clone_and_copy() {
            let hooks = NoopAgentHooks;
            let cloned = hooks;
            let copied = hooks;
            assert_eq!(size_of_val(&cloned), 0);
            assert_eq!(size_of_val(&copied), 0);
            assert_eq!(size_of_val(&hooks), 0);
        }

        #[test]
        fn default_creates_instance() {
            let hooks = NoopAgentHooks;
            assert_eq!(size_of_val(&hooks), 0);
        }

        #[tokio::test]
        async fn all_hooks_are_noop() {
            let hooks = NoopAgentHooks;
            let ctx = RunContext::new();
            let output = serde_json::json!("test");
            let response = test_response();
            let messages = vec![Message::user("hello")];
            let error = Error::agent("err");

            hooks.on_start(&ctx).await;
            hooks.on_end(&ctx, &output).await;
            hooks.on_llm_start(&ctx, Some("sys"), &messages).await;
            hooks.on_llm_end(&ctx, &response).await;
            hooks.on_tool_start(&ctx, "tool").await;
            hooks.on_tool_end(&ctx, "tool", "ok").await;
            hooks.on_handoff(&ctx, "other").await;
            hooks.on_error(&ctx, &error).await;
        }

        #[test]
        fn into_boxed() {
            let _: BoxedAgentHooks = Box::new(NoopAgentHooks);
        }

        #[test]
        fn into_shared() {
            let _: SharedAgentHooks = Arc::new(NoopAgentHooks);
        }
    }
}
