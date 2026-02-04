//! Agent run result types.

use std::fmt::Write as _;

use serde_json::Value;

use crate::{memory::Timing, providers::common::TokenUsage};

use super::RunState;

/// Extended result of an agent run, containing detailed execution information.
///
/// Use `agent.run_with_result()` to get this instead of just the final answer.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// The final output of the agent run.
    pub output: Option<Value>,
    /// The state of the run (success, max_steps_reached, etc.).
    pub state: RunState,
    /// Total token usage during the run.
    pub token_usage: TokenUsage,
    /// Number of steps executed.
    pub steps_taken: usize,
    /// Timing information.
    pub timing: Timing,
    /// Error message if the run failed.
    pub error: Option<String>,
}

impl RunResult {
    /// Check if the run was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self.state, RunState::Success)
    }

    /// Get the output value, if available.
    #[must_use]
    pub const fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    /// Convert the run result to a `Result<Value>`.
    ///
    /// Returns `Ok(value)` if successful, otherwise returns an appropriate error.
    pub fn into_result(self, max_steps: usize) -> crate::Result<Value> {
        use crate::error::AgentError;
        match self.state {
            RunState::Success => self.output.ok_or_else(|| {
                AgentError::configuration("Run succeeded but no output was produced")
            }),
            RunState::MaxStepsReached => Err(AgentError::max_steps(self.steps_taken, max_steps)),
            RunState::Interrupted => Err(AgentError::Interrupted),
            RunState::Failed => Err(AgentError::configuration(
                self.error.unwrap_or_else(|| "Unknown error".to_string()),
            )),
        }
    }

    /// Generate a summary of the run.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut summary = String::with_capacity(256);
        let _ = writeln!(summary, "Run State: {}", self.state);
        let _ = writeln!(summary, "Steps Taken: {}", self.steps_taken);
        let _ = writeln!(
            summary,
            "Duration: {:.2}s",
            self.timing.duration_secs().unwrap_or_default()
        );
        let _ = writeln!(
            summary,
            "Tokens: {} (in: {}, out: {})",
            self.token_usage.total(),
            self.token_usage.input_tokens,
            self.token_usage.output_tokens
        );
        if let Some(output) = &self.output {
            let _ = writeln!(summary, "Output: {output}");
        }
        if let Some(error) = &self.error {
            let _ = writeln!(summary, "Error: {error}");
        }
        summary
    }
}
