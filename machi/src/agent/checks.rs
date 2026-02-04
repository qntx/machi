//! Final answer validation checks.

use serde_json::Value;

use crate::{Result, error::AgentError, memory::AgentMemory};

/// A function that validates the final answer before accepting it.
///
/// The check function receives:
/// - `answer`: The final answer value
/// - `memory`: The agent's memory containing all steps
///
/// Returns `Ok(())` if the answer is valid, or `Err(reason)` if invalid.
pub type FinalAnswerCheck = Box<dyn Fn(&Value, &AgentMemory) -> Result<()> + Send + Sync>;

/// Builder for creating final answer checks.
pub struct FinalAnswerChecks {
    checks: Vec<FinalAnswerCheck>,
}

impl std::fmt::Debug for FinalAnswerChecks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FinalAnswerChecks")
            .field("checks_count", &self.checks.len())
            .finish()
    }
}

impl Default for FinalAnswerChecks {
    fn default() -> Self {
        Self::new()
    }
}

impl FinalAnswerChecks {
    /// Create a new empty set of checks.
    #[must_use]
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Add a check function to the validation chain.
    #[must_use]
    pub fn with_check<F>(mut self, check: F) -> Self
    where
        F: Fn(&Value, &AgentMemory) -> Result<()> + Send + Sync + 'static,
    {
        self.checks.push(Box::new(check));
        self
    }

    /// Add a check that the answer is not null.
    #[must_use]
    pub fn not_null(self) -> Self {
        self.with_check(|answer, _| {
            if answer.is_null() {
                Err(AgentError::configuration("Final answer cannot be null"))
            } else {
                Ok(())
            }
        })
    }

    /// Add a check that the answer is not an empty string.
    #[must_use]
    pub fn not_empty(self) -> Self {
        self.with_check(|answer, _| {
            if let Some(s) = answer.as_str()
                && s.trim().is_empty()
            {
                return Err(AgentError::configuration("Final answer cannot be empty"));
            }
            Ok(())
        })
    }

    /// Add a check that the answer contains a specific substring.
    #[must_use]
    pub fn contains(self, substring: impl Into<String>) -> Self {
        let substring = substring.into();
        self.with_check(move |answer, _| {
            let text = answer.to_string();
            if text.contains(&substring) {
                Ok(())
            } else {
                Err(AgentError::configuration(format!(
                    "Final answer must contain '{substring}'"
                )))
            }
        })
    }

    /// Run all checks on the given answer.
    pub(crate) fn validate(&self, answer: &Value, memory: &AgentMemory) -> Result<()> {
        for check in &self.checks {
            check(answer, memory)?;
        }
        Ok(())
    }

    /// Check if there are any checks defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }
}
