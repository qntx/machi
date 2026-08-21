//! Compaction strategy trait.

use ovo_types::{Message, OvoError};

/// Result of a compaction pass.
#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    /// Messages after compaction.
    pub messages: Vec<Message>,
    /// True when the list changed.
    pub changed: bool,
    /// Strategy name for metrics.
    pub strategy: &'static str,
}

/// Pure compaction strategy.
pub trait CompactionStrategy: Send + Sync {
    /// Stable strategy id (`max_messages`, `token_threshold`, …).
    fn name(&self) -> &'static str;

    /// Whether compaction should run given current size estimates.
    fn should_compact(&self, messages: &[Message], token_estimate: u64) -> bool;

    /// Produce a compacted message list.
    ///
    /// # Errors
    ///
    /// Returns typed compaction failures (should not panic on normal input).
    fn compact(&self, messages: Vec<Message>) -> Result<CompactionOutcome, OvoError>;
}
