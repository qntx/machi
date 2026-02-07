//! Guardrail module — safety checks for agent inputs and outputs.
//!
//! Guardrails are validation checks that run alongside agent execution to
//! ensure inputs and outputs meet safety, quality, and policy criteria.
//!
//! - **[`InputGuardrail`]** — validates user input before or alongside the
//!   first LLM call (e.g., off-topic detection, content filtering).
//! - **[`OutputGuardrail`]** — validates the agent's final output after
//!   generation (e.g., PII detection, format checking, policy compliance).
//!
//! # Tripwire Mechanism
//!
//! Each guardrail returns a [`GuardrailOutput`] containing a `tripwire_triggered`
//! flag. When any guardrail triggers its tripwire, the agent run is immediately
//! halted and an [`Error::InputGuardrailTriggered`](crate::Error) or
//! [`Error::OutputGuardrailTriggered`](crate::Error) is returned.
//!
//! # Execution Modes
//!
//! Input guardrails support two execution modes via [`InputGuardrail::run_in_parallel`]:
//!
//! - **Sequential** (`false`): Runs before the first LLM call. If the tripwire
//!   triggers, the LLM call is never made — saving cost and latency.
//! - **Parallel** (`true`, default): Runs concurrently with the first LLM call
//!   via `tokio::join!`. If the tripwire triggers, the LLM result is discarded.
//!
//! Output guardrails always run after the agent produces a final output,
//! and are executed concurrently with each other.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! struct ContentFilter;
//!
//! #[async_trait::async_trait]
//! impl InputGuardrailCheck for ContentFilter {
//!     async fn check(
//!         &self,
//!         _context: &RunContext,
//!         _agent_name: &str,
//!         input: &[Message],
//!     ) -> Result<GuardrailOutput> {
//!         let text = input.iter()
//!             .filter_map(|m| m.text())
//!             .collect::<String>();
//!         if text.contains("forbidden") {
//!             Ok(GuardrailOutput::tripwire("Forbidden content detected"))
//!         } else {
//!             Ok(GuardrailOutput::pass())
//!         }
//!     }
//! }
//!
//! let agent = Agent::new("safe-agent")
//!     .instructions("You are a helpful assistant.")
//!     .model("gpt-4o")
//!     .provider(provider.clone())
//!     .input_guardrail(InputGuardrail::new("content-filter", ContentFilter));
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::callback::RunContext;
use crate::error::Result;
use crate::message::Message;

/// The output of a guardrail check function.
///
/// Contains a boolean tripwire flag and optional structured information
/// about the check that was performed. When `tripwire_triggered` is `true`,
/// the agent run is halted immediately.
#[derive(Debug, Clone)]
pub struct GuardrailOutput {
    /// Whether the tripwire was triggered.
    ///
    /// If `true`, the agent's execution will be immediately halted and
    /// an error will be returned to the caller.
    pub tripwire_triggered: bool,

    /// Optional structured information about the guardrail's output.
    ///
    /// Can contain details about the checks performed, confidence scores,
    /// detected issues, or any other metadata useful for debugging and
    /// observability.
    pub output_info: Value,
}

impl GuardrailOutput {
    /// Create a passing guardrail output (tripwire not triggered).
    #[must_use]
    pub const fn pass() -> Self {
        Self {
            tripwire_triggered: false,
            output_info: Value::Null,
        }
    }

    /// Create a failing guardrail output (tripwire triggered).
    ///
    /// The `info` parameter should describe why the tripwire was triggered,
    /// and will be included in the resulting error for observability.
    #[must_use]
    pub fn tripwire(info: impl Into<Value>) -> Self {
        Self {
            tripwire_triggered: true,
            output_info: info.into(),
        }
    }

    /// Create a passing output with additional diagnostic information.
    ///
    /// Useful when the guardrail passes but you want to record metadata
    /// (e.g., confidence scores, partial matches) for observability.
    #[must_use]
    pub fn pass_with_info(info: impl Into<Value>) -> Self {
        Self {
            tripwire_triggered: false,
            output_info: info.into(),
        }
    }

    /// Returns `true` if the tripwire was triggered.
    #[must_use]
    pub const fn is_triggered(&self) -> bool {
        self.tripwire_triggered
    }
}

/// Trait for implementing input guardrail check logic.
///
/// Implement this trait on your own struct to define custom input validation.
/// The [`check`](InputGuardrailCheck::check) method receives the run context,
/// agent name, and the full message list (system prompt + history + user input),
/// and must return a [`GuardrailOutput`] indicating whether the input passes.
#[async_trait]
pub trait InputGuardrailCheck: Send + Sync {
    /// Check the input messages and return a guardrail output.
    ///
    /// # Arguments
    ///
    /// * `context` — the current run context (usage, step, state)
    /// * `agent_name` — name of the agent being executed
    /// * `input` — the full message list being sent to the LLM
    async fn check(
        &self,
        context: &RunContext,
        agent_name: &str,
        input: &[Message],
    ) -> Result<GuardrailOutput>;
}

/// An input guardrail that validates user input before or alongside the LLM.
///
/// Input guardrails are configured on an [`Agent`](crate::agent::Agent) or
/// [`RunConfig`](crate::agent::RunConfig) and are automatically executed by
/// the [`Runner`](crate::agent::Runner) during the first step of a run.
///
/// # Execution Modes
///
/// - **Sequential** (`run_in_parallel: false`): Runs before the LLM call.
///   If triggered, the LLM call is never made.
/// - **Parallel** (`run_in_parallel: true`, default): Runs concurrently with
///   the first LLM call. If triggered, the LLM result is discarded.
#[derive(Clone)]
pub struct InputGuardrail {
    /// Name of this guardrail (used in tracing and error messages).
    name: String,

    /// Whether to run concurrently with the first LLM call.
    run_in_parallel: bool,

    /// The guardrail check implementation.
    check: Arc<dyn InputGuardrailCheck>,
}

impl InputGuardrail {
    /// Create a new input guardrail with the given name and check logic.
    ///
    /// By default, the guardrail runs in parallel with the first LLM call.
    #[must_use]
    pub fn new(name: impl Into<String>, check: impl InputGuardrailCheck + 'static) -> Self {
        Self {
            name: name.into(),
            run_in_parallel: true,
            check: Arc::new(check),
        }
    }

    /// Set whether this guardrail runs in parallel with the LLM call.
    ///
    /// - `true` (default): Runs concurrently — lower latency but the LLM
    ///   call is still made even if the guardrail triggers.
    /// - `false`: Runs before the LLM call — higher latency but avoids
    ///   unnecessary LLM costs when the guardrail triggers.
    #[must_use]
    pub const fn run_in_parallel(mut self, parallel: bool) -> Self {
        self.run_in_parallel = parallel;
        self
    }

    /// Returns the name of this guardrail.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns whether this guardrail runs in parallel with the LLM.
    #[must_use]
    pub const fn is_parallel(&self) -> bool {
        self.run_in_parallel
    }

    /// Execute this guardrail check.
    ///
    /// Returns an [`InputGuardrailResult`] containing the guardrail reference
    /// and the check output.
    pub async fn run(
        &self,
        context: &RunContext,
        agent_name: &str,
        input: &[Message],
    ) -> Result<InputGuardrailResult> {
        let output = self.check.check(context, agent_name, input).await?;
        Ok(InputGuardrailResult {
            guardrail_name: self.name.clone(),
            output,
        })
    }
}

impl std::fmt::Debug for InputGuardrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputGuardrail")
            .field("name", &self.name)
            .field("run_in_parallel", &self.run_in_parallel)
            .finish_non_exhaustive()
    }
}

/// The result of running an input guardrail.
#[derive(Debug, Clone)]
pub struct InputGuardrailResult {
    /// Name of the guardrail that produced this result.
    pub guardrail_name: String,

    /// The guardrail check output.
    pub output: GuardrailOutput,
}

impl InputGuardrailResult {
    /// Returns `true` if the tripwire was triggered.
    #[must_use]
    pub const fn is_triggered(&self) -> bool {
        self.output.tripwire_triggered
    }
}

/// Trait for implementing output guardrail check logic.
///
/// Implement this trait on your own struct to define custom output validation.
/// The [`check`](OutputGuardrailCheck::check) method receives the run context,
/// agent name, and the final output value, and must return a [`GuardrailOutput`]
/// indicating whether the output passes.
#[async_trait]
pub trait OutputGuardrailCheck: Send + Sync {
    /// Check the agent's final output and return a guardrail output.
    ///
    /// # Arguments
    ///
    /// * `context` — the current run context (usage, step, state)
    /// * `agent_name` — name of the agent that produced the output
    /// * `output` — the final output value from the agent
    async fn check(
        &self,
        context: &RunContext,
        agent_name: &str,
        output: &Value,
    ) -> Result<GuardrailOutput>;
}

/// An output guardrail that validates the agent's final response.
///
/// Output guardrails are configured on an [`Agent`](crate::agent::Agent) or
/// [`RunConfig`](crate::agent::RunConfig) and are automatically executed by
/// the [`Runner`](crate::agent::Runner) after the agent produces a final output.
///
/// All output guardrails run concurrently. If any guardrail's tripwire is
/// triggered, the run returns an error and the output is not delivered.
#[derive(Clone)]
pub struct OutputGuardrail {
    /// Name of this guardrail (used in tracing and error messages).
    name: String,

    /// The guardrail check implementation.
    check: Arc<dyn OutputGuardrailCheck>,
}

impl OutputGuardrail {
    /// Create a new output guardrail with the given name and check logic.
    #[must_use]
    pub fn new(name: impl Into<String>, check: impl OutputGuardrailCheck + 'static) -> Self {
        Self {
            name: name.into(),
            check: Arc::new(check),
        }
    }

    /// Returns the name of this guardrail.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Execute this guardrail check.
    ///
    /// Returns an [`OutputGuardrailResult`] containing the guardrail reference
    /// and the check output.
    pub async fn run(
        &self,
        context: &RunContext,
        agent_name: &str,
        output: &Value,
    ) -> Result<OutputGuardrailResult> {
        let guardrail_output = self.check.check(context, agent_name, output).await?;
        Ok(OutputGuardrailResult {
            guardrail_name: self.name.clone(),
            output: guardrail_output,
        })
    }
}

impl std::fmt::Debug for OutputGuardrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputGuardrail")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// The result of running an output guardrail.
#[derive(Debug, Clone)]
pub struct OutputGuardrailResult {
    /// Name of the guardrail that produced this result.
    pub guardrail_name: String,

    /// The guardrail check output.
    pub output: GuardrailOutput,
}

impl OutputGuardrailResult {
    /// Returns `true` if the tripwire was triggered.
    #[must_use]
    pub const fn is_triggered(&self) -> bool {
        self.output.tripwire_triggered
    }
}
