//! Tool behavioral metadata and capability flags.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// How a tool interacts with concurrent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConcurrencyMode {
    /// Safe to run alongside other non-exclusive tools.
    ReadOnly,
    /// May mutate; concurrent with other concurrent/read-only tools.
    #[default]
    Concurrent,
    /// Must run alone.
    Exclusive,
}

/// Destructiveness class for policy and approvals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Destructiveness {
    /// No side effects of consequence.
    #[default]
    None,
    /// Effects can be undone.
    Reversible,
    /// Permanent effects.
    Irreversible,
}

/// Cancel behavior while a tool is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InterruptBehavior {
    /// Drop / cancel immediately.
    #[default]
    Cancel,
    /// Wait for natural completion.
    WaitComplete,
}

/// Fine-grained capability flags for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CapabilityFlag {
    /// Read filesystem or data sources.
    Read,
    /// Write filesystem or mutate state.
    Write,
    /// Execute shell / process.
    Execute,
    /// Network access.
    Network,
    /// Spawn nested agents.
    Spawn,
}

/// Metadata used by dispatch and capability filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolMetadata {
    /// Concurrency class.
    pub concurrency: ConcurrencyMode,
    /// Destructiveness.
    pub destructiveness: Destructiveness,
    /// Interrupt behavior.
    pub interrupt: InterruptBehavior,
    /// Optional execution timeout.
    pub timeout: Option<Duration>,
    /// Capability flags required/advertised.
    pub capabilities: Vec<CapabilityFlag>,
    /// Optional per-tool concurrency cap (alongside [`ConcurrencyMode`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrency: Option<usize>,
}

impl Default for ToolMetadata {
    fn default() -> Self {
        Self {
            concurrency: ConcurrencyMode::Concurrent,
            destructiveness: Destructiveness::None,
            interrupt: InterruptBehavior::Cancel,
            timeout: None,
            capabilities: Vec::new(),
            max_concurrency: None,
        }
    }
}

impl ToolMetadata {
    /// Read-only tool defaults.
    #[must_use]
    pub fn read_only() -> Self {
        Self {
            concurrency: ConcurrencyMode::ReadOnly,
            capabilities: vec![CapabilityFlag::Read],
            ..Self::default()
        }
    }

    /// Exclusive mutating tool defaults.
    #[must_use]
    pub fn exclusive_write() -> Self {
        Self {
            concurrency: ConcurrencyMode::Exclusive,
            destructiveness: Destructiveness::Reversible,
            capabilities: vec![CapabilityFlag::Write],
            ..Self::default()
        }
    }

    /// Nested-agent spawn tool defaults (concurrent, non-destructive).
    #[must_use]
    pub fn spawn() -> Self {
        Self {
            concurrency: ConcurrencyMode::Concurrent,
            destructiveness: Destructiveness::None,
            capabilities: vec![CapabilityFlag::Spawn],
            ..Self::default()
        }
    }

    /// Exclusive shell / process execution defaults.
    #[must_use]
    pub fn shell_execute(timeout: Duration) -> Self {
        Self {
            concurrency: ConcurrencyMode::Exclusive,
            destructiveness: Destructiveness::Reversible,
            interrupt: InterruptBehavior::Cancel,
            timeout: Some(timeout),
            capabilities: vec![CapabilityFlag::Execute, CapabilityFlag::Write],
            max_concurrency: Some(1),
        }
    }

    /// Set per-tool concurrency cap.
    #[must_use]
    pub const fn with_max_concurrency(mut self, n: Option<usize>) -> Self {
        self.max_concurrency = n;
        self
    }

    /// True when the tool is admissible under a read-only capability mode.
    #[must_use]
    pub fn allowed_in_read_only(&self) -> bool {
        !self.capabilities.iter().any(|c| {
            matches!(
                c,
                CapabilityFlag::Write | CapabilityFlag::Execute | CapabilityFlag::Spawn
            )
        }) && self.destructiveness == Destructiveness::None
            && self.concurrency != ConcurrencyMode::Exclusive
    }
}
