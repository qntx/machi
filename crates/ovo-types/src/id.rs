//! Typed identifiers for kernel entities.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ErrorCode, OvoError};

macro_rules! typed_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Generate a new random id with a stable prefix.
            #[must_use]
            pub fn generate() -> Self {
                Self(format!("{}_{}", $prefix, Uuid::new_v4().simple()))
            }

            /// Borrow the raw string.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Construct from a non-empty string without prefix validation.
            ///
            /// # Errors
            ///
            /// Returns [`OvoError`] when `value` is empty or whitespace-only.
            pub fn new(value: impl Into<String>) -> Result<Self, OvoError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(OvoError::new(
                        ErrorCode::TypesInvalidId,
                        format!("{} must be non-empty", stringify!($name)),
                    ));
                }
                Ok(Self(value))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = OvoError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::new(s)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

typed_id!(
    /// Identifies an agent instance or nested run.
    AgentId,
    "agent"
);
typed_id!(
    /// Identifies a turn or top-level run.
    RunId,
    "run"
);
typed_id!(
    /// Identifies a multi-turn session.
    SessionId,
    "session"
);
typed_id!(
    /// Identifies a model tool call within a turn.
    ToolCallId,
    "call"
);
typed_id!(
    /// Identifies a workflow orchestration run.
    WorkflowRunId,
    "wf"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_has_prefix() {
        let id = AgentId::generate();
        assert!(id.as_str().starts_with("agent_"), "{}", id);
    }

    #[test]
    fn rejects_empty() {
        let err = SessionId::new("  ").expect_err("empty");
        assert_eq!(err.code(), ErrorCode::TypesInvalidId);
    }
}
