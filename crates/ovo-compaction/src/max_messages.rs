//! Keep at most N messages, preserving leading system and tail.

use ovo_types::{ErrorCode, Message, OvoError};

use crate::select::{apply_range, select_compaction_range, tool_pair_invariant_holds};
use crate::strategy::{CompactionOutcome, CompactionStrategy};

/// Drop oldest non-system messages until `max` remains.
#[derive(Debug, Clone, Copy)]
pub struct MaxMessages {
    /// Maximum messages retained (including system).
    pub max: usize,
}

impl MaxMessages {
    /// Construct with a positive max.
    ///
    /// # Errors
    ///
    /// Returns error when `max == 0`.
    pub fn new(max: usize) -> Result<Self, OvoError> {
        if max == 0 {
            return Err(OvoError::new(
                ErrorCode::CompactionFailed,
                "MaxMessages max must be >= 1",
            ));
        }
        Ok(Self { max })
    }
}

impl CompactionStrategy for MaxMessages {
    fn name(&self) -> &'static str {
        "max_messages"
    }

    fn should_compact(&self, messages: &[Message], _token_estimate: u64) -> bool {
        messages.len() > self.max
    }

    fn compact(&self, messages: Vec<Message>) -> Result<CompactionOutcome, OvoError> {
        if messages.len() <= self.max {
            return Ok(CompactionOutcome {
                messages,
                changed: false,
                strategy: self.name(),
            });
        }
        let before = messages.len();
        let compacted = compact_max_messages(messages, self.max);
        let changed = compacted.len() != before;
        Ok(CompactionOutcome {
            messages: compacted,
            changed,
            strategy: self.name(),
        })
    }
}

/// Shared algorithm used by runtime `VecConversationState` and this strategy.
///
/// Uses [`select_compaction_range`] so tool-result runs are never split mid-pair.
/// When no safe split exists, returns the input unchanged (same as other strategies).
#[must_use]
pub fn compact_max_messages(messages: Vec<Message>, max_messages: usize) -> Vec<Message> {
    if max_messages == 0 || messages.len() <= max_messages {
        return messages;
    }
    let Some(range) = select_compaction_range(&messages, max_messages) else {
        return messages;
    };
    let out = apply_range(messages, range, None);
    debug_assert!(
        tool_pair_invariant_holds(&out),
        "compact_max_messages must preserve tool-pair invariant"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_system_and_tail() {
        let s = MaxMessages::new(3).expect("max");
        let out = s
            .compact(vec![
                Message::system("sys"),
                Message::user("1"),
                Message::user("2"),
                Message::user("3"),
                Message::user("4"),
            ])
            .expect("compact");
        assert!(out.changed);
        assert_eq!(out.messages.len(), 3);
        assert_eq!(
            out.messages.first().map(Message::text).as_deref(),
            Some("sys")
        );
        assert_eq!(out.messages.get(2).map(Message::text).as_deref(), Some("4"));
    }
}
