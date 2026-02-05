//! Agent run result types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::memory::Timing;
use crate::error::{Error, Result};
use crate::usage::Usage;

/// State of an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// Run completed successfully.
    Success,
    /// Run failed with an error.
    Failed,
    /// Run was interrupted.
    Interrupted,
    /// Maximum steps reached without final answer.
    MaxStepsReached,
}

impl RunState {
    /// Check if the run was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Check if the run failed.
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        matches!(self, Self::Failed)
    }
}

/// Result of an agent run with metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// The output value if successful.
    pub output: Option<Value>,
    /// The final state of the run.
    pub state: RunState,
    /// Token usage statistics.
    pub token_usage: Option<Usage>,
    /// Number of steps taken.
    pub steps_taken: usize,
    /// Timing information.
    pub timing: Timing,
    /// Error message if failed.
    pub error: Option<String>,
}

impl RunResult {
    /// Check if the run was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.state.is_success()
    }

    /// Get the output value if successful.
    #[must_use]
    pub const fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    /// Get the output as a specific type.
    pub fn output_as<T: for<'de> Deserialize<'de>>(&self) -> Option<T> {
        self.output
            .as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Convert to Result, returning the output or an error.
    pub fn into_result(self, max_steps: usize) -> Result<Value> {
        match self.state {
            RunState::Success => self
                .output
                .ok_or_else(|| Error::agent("No output produced")),
            RunState::Failed => Err(Error::agent(
                self.error.unwrap_or_else(|| "Unknown error".to_owned()),
            )),
            RunState::Interrupted => Err(Error::Interrupted),
            RunState::MaxStepsReached => Err(Error::max_steps(max_steps)),
        }
    }

    /// Get duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.timing.duration_secs.unwrap_or(0.0)
    }
}

impl Default for RunResult {
    fn default() -> Self {
        Self {
            output: None,
            state: RunState::Failed,
            token_usage: None,
            steps_taken: 0,
            timing: Timing::default(),
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_state() {
        assert!(RunState::Success.is_success());
        assert!(!RunState::Failed.is_success());
        assert!(RunState::Failed.is_failed());
    }

    #[test]
    fn test_run_result_success() {
        let result = RunResult {
            output: Some(serde_json::json!({"answer": 42})),
            state: RunState::Success,
            token_usage: Some(Usage::new(100, 50)),
            steps_taken: 3,
            timing: Timing::default(),
            error: None,
        };

        assert!(result.is_success());
        assert!(result.output().is_some());
    }

    #[test]
    fn test_run_result_into_result() {
        let success = RunResult {
            output: Some(serde_json::json!("done")),
            state: RunState::Success,
            ..Default::default()
        };
        assert!(success.into_result(20).is_ok());

        let failed = RunResult {
            output: None,
            state: RunState::Failed,
            error: Some("test error".to_owned()),
            ..Default::default()
        };
        assert!(failed.into_result(20).is_err());
    }
}
