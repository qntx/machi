//! Workflow outcomes.

use serde::{Deserialize, Serialize};

/// Why a workflow paused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PauseKind {
    /// Waiting on the user.
    User,
    /// Temporary backoff.
    BackOff,
    /// No progress detected by script.
    NoProgress,
    /// Verification / input missing.
    Verification,
    /// Infrastructure issue.
    Infra,
}

impl PauseKind {
    /// Stable string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::BackOff => "back_off",
            Self::NoProgress => "no_progress",
            Self::Verification => "verification",
            Self::Infra => "infra",
        }
    }
}

/// Terminal or pausable workflow outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
#[non_exhaustive]
pub enum WorkflowOutcome {
    /// Successful completion.
    Completed {
        /// Script result value.
        result: serde_json::Value,
    },
    /// Cooperative pause (resumable).
    Paused {
        /// Pause classification.
        kind: PauseKind,
        /// Human message.
        message: String,
    },
    /// Agent budget exhausted (resumable with higher budget).
    BudgetExceeded {
        /// Message.
        message: String,
    },
    /// Cancelled.
    Cancelled,
    /// Hard failure.
    Failed {
        /// Error text.
        error: String,
    },
}
