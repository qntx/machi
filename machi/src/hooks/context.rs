//! Hook context types for lifecycle events.
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
/// following the `OpenAI` Agents SDK pattern of providing a context wrapper
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
/// use machi::hooks::RunContext;
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
