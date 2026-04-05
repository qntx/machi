//! Error types for agent runtime operations.
//!
//! [`AgentError`] covers all failure modes during agent execution — missing
//! configuration, step limits, guardrail triggers, and interruptions.
//! It integrates into the global [`Error`](crate::Error) hierarchy via `Error::Agent`.

/// Error type for agent runtime operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
#[allow(clippy::module_name_repetitions)]
pub enum AgentError {
    /// General agent runtime error with a descriptive message.
    #[error("{0}")]
    Runtime(String),

    /// Maximum steps reached during agent execution.
    #[error("Maximum steps ({max_steps}) reached without final answer")]
    MaxSteps {
        /// The maximum number of steps configured.
        max_steps: usize,
    },

    /// Input guardrail tripwire was triggered.
    #[error("Input guardrail '{name}' tripwire triggered")]
    InputGuardrailTriggered {
        /// Name of the guardrail that triggered.
        name: String,
        /// Diagnostic information from the guardrail.
        info: serde_json::Value,
    },

    /// Output guardrail tripwire was triggered.
    #[error("Output guardrail '{name}' tripwire triggered")]
    OutputGuardrailTriggered {
        /// Name of the guardrail that triggered.
        name: String,
        /// Diagnostic information from the guardrail.
        info: serde_json::Value,
    },
}

impl AgentError {
    /// Create a runtime error with a message.
    #[must_use]
    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::Runtime(msg.into())
    }

    /// Create a max steps error.
    #[must_use]
    pub const fn max_steps(max_steps: usize) -> Self {
        Self::MaxSteps { max_steps }
    }

    /// Create an input guardrail triggered error.
    #[must_use]
    pub fn input_guardrail_triggered(name: impl Into<String>, info: serde_json::Value) -> Self {
        Self::InputGuardrailTriggered {
            name: name.into(),
            info,
        }
    }

    /// Create an output guardrail triggered error.
    #[must_use]
    pub fn output_guardrail_triggered(name: impl Into<String>, info: serde_json::Value) -> Self {
        Self::OutputGuardrailTriggered {
            name: name.into(),
            info,
        }
    }
}
