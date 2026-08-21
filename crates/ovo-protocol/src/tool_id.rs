//! Stable tool identity used for routing and metrics.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Canonical tool identifier (name-based for v1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolId {
    name: String,
}

impl ToolId {
    /// Create from a non-empty name.
    ///
    /// # Errors
    ///
    /// Returns `None` when `name` is empty or whitespace-only.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Option<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return None;
        }
        Some(Self { name })
    }

    /// Create without validation (for compile-time constants).
    ///
    /// # Panics
    ///
    /// Panics if `name` is empty. Prefer [`Self::new`] for runtime input.
    #[must_use]
    pub fn const_new(name: &'static str) -> Self {
        assert!(!name.is_empty(), "tool id must be non-empty");
        Self {
            name: name.to_owned(),
        }
    }

    /// Underlying name string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for ToolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl AsRef<str> for ToolId {
    fn as_ref(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        assert!(ToolId::new("").is_none(), "empty rejected");
        assert!(ToolId::new("  ").is_none(), "whitespace rejected");
    }

    #[test]
    fn accepts_name() {
        let id = ToolId::new("read_file").expect("id");
        assert_eq!(id.as_str(), "read_file");
    }
}
