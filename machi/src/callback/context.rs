//! Hook context types for callback lifecycle events.
//!
//! Provides [`RunContext`] which carries shared state across all hook invocations
//! during an agent run, including cumulative token usage, step tracking, and
//! user-defined state.

use std::collections::HashMap;

use serde_json::Value;

use crate::usage::Usage;

/// Context passed to all hook methods during an agent run.
///
/// This struct carries shared state that is available to every hook invocation,
/// following the OpenAI Agents SDK pattern of providing a context wrapper
/// that flows through the entire agent lifecycle.
///
/// # Design
///
/// - **Immutable by default**: Hooks receive `&RunContext` — they observe but
///   do not modify the execution flow (separation of concerns with guardrails).
/// - **Cumulative usage**: Tracks token consumption across all LLM calls in the run.
/// - **User state**: Arbitrary key-value pairs for user-defined data sharing.
///
/// # Example
///
/// ```rust
/// use machi::callback::RunContext;
///
/// let ctx = RunContext::new()
///     .with_agent_name("my_agent")
///     .with_step(3);
///
/// assert_eq!(ctx.agent_name(), Some("my_agent"));
/// assert_eq!(ctx.step(), 3);
/// ```
#[derive(Debug, Clone, Default)]
pub struct RunContext {
    /// Cumulative token usage across all LLM calls in this run.
    usage: Usage,
    /// Current step number (1-indexed during execution, 0 before start).
    step: usize,
    /// Name of the currently active agent.
    agent_name: Option<String>,
    /// User-defined state for sharing data across hooks.
    state: HashMap<String, Value>,
}

impl RunContext {
    /// Create a new empty run context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the agent name.
    #[must_use]
    pub fn with_agent_name(mut self, name: impl Into<String>) -> Self {
        self.agent_name = Some(name.into());
        self
    }

    /// Set the current step number.
    #[must_use]
    pub const fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Set the cumulative token usage.
    #[must_use]
    pub const fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    /// Get the cumulative token usage.
    #[must_use]
    pub const fn usage(&self) -> &Usage {
        &self.usage
    }

    /// Get the current step number.
    #[must_use]
    pub const fn step(&self) -> usize {
        self.step
    }

    /// Get the agent name, if set.
    #[must_use]
    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }

    /// Get a reference to the user-defined state map.
    #[must_use]
    pub const fn state(&self) -> &HashMap<String, Value> {
        &self.state
    }

    /// Get a value from the user-defined state.
    #[must_use]
    pub fn get_state(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }

    /// Insert a value into the user-defined state.
    pub fn set_state(&mut self, key: impl Into<String>, value: Value) {
        self.state.insert(key.into(), value);
    }

    /// Remove a value from the user-defined state.
    pub fn remove_state(&mut self, key: &str) -> Option<Value> {
        self.state.remove(key)
    }

    /// Update the cumulative token usage by adding new usage.
    pub fn add_usage(&mut self, usage: Usage) {
        self.usage += usage;
    }

    /// Advance to the next step.
    pub const fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Update the agent name.
    pub fn set_agent_name(&mut self, name: impl Into<String>) {
        self.agent_name = Some(name.into());
    }

    /// Reset the context for a new run.
    pub fn reset(&mut self) {
        self.usage = Usage::zero();
        self.step = 0;
        self.state.clear();
        // Preserve agent_name — it is typically set once at construction.
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    mod construction {
        use super::*;

        #[test]
        fn new_creates_empty_context() {
            let ctx = RunContext::new();
            assert_eq!(ctx.step(), 0);
            assert!(ctx.agent_name().is_none());
            assert!(ctx.usage().is_empty());
            assert!(ctx.state().is_empty());
        }

        #[test]
        fn default_is_same_as_new() {
            let ctx = RunContext::default();
            assert_eq!(ctx.step(), 0);
            assert!(ctx.agent_name().is_none());
        }

        #[test]
        fn with_agent_name_sets_name() {
            let ctx = RunContext::new().with_agent_name("test_agent");
            assert_eq!(ctx.agent_name(), Some("test_agent"));
        }

        #[test]
        fn with_step_sets_step() {
            let ctx = RunContext::new().with_step(5);
            assert_eq!(ctx.step(), 5);
        }

        #[test]
        fn with_usage_sets_usage() {
            let usage = Usage::new(100, 50);
            let ctx = RunContext::new().with_usage(usage);
            assert_eq!(ctx.usage().input_tokens, 100);
            assert_eq!(ctx.usage().output_tokens, 50);
        }

        #[test]
        fn builder_chain() {
            let ctx = RunContext::new()
                .with_agent_name("agent_a")
                .with_step(3)
                .with_usage(Usage::new(200, 100));

            assert_eq!(ctx.agent_name(), Some("agent_a"));
            assert_eq!(ctx.step(), 3);
            assert_eq!(ctx.usage().total_tokens, 300);
        }
    }

    mod state_management {
        use super::*;

        #[test]
        fn set_and_get_state() {
            let mut ctx = RunContext::new();
            ctx.set_state("key1", serde_json::json!("value1"));
            assert_eq!(ctx.get_state("key1"), Some(&serde_json::json!("value1")));
        }

        #[test]
        fn get_state_returns_none_for_missing_key() {
            let ctx = RunContext::new();
            assert!(ctx.get_state("nonexistent").is_none());
        }

        #[test]
        fn set_state_overwrites_existing() {
            let mut ctx = RunContext::new();
            ctx.set_state("key", serde_json::json!(1));
            ctx.set_state("key", serde_json::json!(2));
            assert_eq!(ctx.get_state("key"), Some(&serde_json::json!(2)));
        }

        #[test]
        fn remove_state_returns_value() {
            let mut ctx = RunContext::new();
            ctx.set_state("key", serde_json::json!("hello"));
            let removed = ctx.remove_state("key");
            assert_eq!(removed, Some(serde_json::json!("hello")));
            assert!(ctx.get_state("key").is_none());
        }

        #[test]
        fn remove_state_returns_none_for_missing() {
            let mut ctx = RunContext::new();
            assert!(ctx.remove_state("missing").is_none());
        }

        #[test]
        fn state_returns_full_map() {
            let mut ctx = RunContext::new();
            ctx.set_state("a", serde_json::json!(1));
            ctx.set_state("b", serde_json::json!(2));
            assert_eq!(ctx.state().len(), 2);
        }
    }

    mod mutation {
        use super::*;

        #[test]
        fn add_usage_accumulates() {
            let mut ctx = RunContext::new();
            ctx.add_usage(Usage::new(100, 50));
            ctx.add_usage(Usage::new(200, 100));
            assert_eq!(ctx.usage().input_tokens, 300);
            assert_eq!(ctx.usage().output_tokens, 150);
            assert_eq!(ctx.usage().total_tokens, 450);
        }

        #[test]
        fn advance_step_increments() {
            let mut ctx = RunContext::new();
            assert_eq!(ctx.step(), 0);
            ctx.advance_step();
            assert_eq!(ctx.step(), 1);
            ctx.advance_step();
            assert_eq!(ctx.step(), 2);
        }

        #[test]
        fn set_agent_name_updates() {
            let mut ctx = RunContext::new().with_agent_name("old");
            ctx.set_agent_name("new");
            assert_eq!(ctx.agent_name(), Some("new"));
        }
    }

    mod reset {
        use super::*;

        #[test]
        fn reset_clears_usage_step_and_state() {
            let mut ctx = RunContext::new()
                .with_agent_name("agent")
                .with_step(5)
                .with_usage(Usage::new(100, 50));
            ctx.set_state("key", serde_json::json!("val"));

            ctx.reset();

            assert_eq!(ctx.step(), 0);
            assert!(ctx.usage().is_empty());
            assert!(ctx.state().is_empty());
        }

        #[test]
        fn reset_preserves_agent_name() {
            let mut ctx = RunContext::new().with_agent_name("preserved");
            ctx.reset();
            assert_eq!(ctx.agent_name(), Some("preserved"));
        }
    }

    mod clone {
        use super::*;

        #[test]
        fn clone_creates_independent_copy() {
            let mut ctx = RunContext::new().with_agent_name("original").with_step(3);
            ctx.set_state("k", serde_json::json!(42));

            let mut cloned = ctx.clone();
            cloned.advance_step();
            cloned.set_state("k", serde_json::json!(99));

            // Original unchanged
            assert_eq!(ctx.step(), 3);
            assert_eq!(ctx.get_state("k"), Some(&serde_json::json!(42)));

            // Clone modified
            assert_eq!(cloned.step(), 4);
            assert_eq!(cloned.get_state("k"), Some(&serde_json::json!(99)));
        }
    }
}
