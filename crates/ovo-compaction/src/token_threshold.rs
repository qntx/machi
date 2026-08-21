//! Compact when estimated tokens exceed a threshold.

use ovo_types::{ErrorCode, Message, OvoError};

use crate::max_messages::compact_max_messages;
use crate::strategy::{CompactionOutcome, CompactionStrategy};

/// Drop oldest non-system messages when `token_estimate` exceeds `max_tokens`.
///
/// Uses the same tail-preserving algorithm as [`crate::MaxMessages`], keeping
/// at most `keep_messages` after the optional leading system message.
#[derive(Debug, Clone, Copy)]
pub struct TokenThreshold {
    /// Token estimate that triggers compaction (must be >= 1).
    pub max_tokens: u64,
    /// Message count retained after compaction (must be >= 1).
    pub keep_messages: usize,
}

impl TokenThreshold {
    /// Construct a token-threshold strategy.
    ///
    /// # Errors
    ///
    /// Returns error when `max_tokens == 0` or `keep_messages == 0`.
    pub fn new(max_tokens: u64, keep_messages: usize) -> Result<Self, OvoError> {
        if max_tokens == 0 {
            return Err(OvoError::new(
                ErrorCode::CompactionFailed,
                "TokenThreshold max_tokens must be >= 1",
            ));
        }
        if keep_messages == 0 {
            return Err(OvoError::new(
                ErrorCode::CompactionFailed,
                "TokenThreshold keep_messages must be >= 1",
            ));
        }
        Ok(Self {
            max_tokens,
            keep_messages,
        })
    }
}

impl CompactionStrategy for TokenThreshold {
    fn name(&self) -> &'static str {
        "token_threshold"
    }

    fn should_compact(&self, messages: &[Message], token_estimate: u64) -> bool {
        token_estimate > self.max_tokens && messages.len() > self.keep_messages
    }

    fn compact(&self, messages: Vec<Message>) -> Result<CompactionOutcome, OvoError> {
        if messages.len() <= self.keep_messages {
            return Ok(CompactionOutcome {
                messages,
                changed: false,
                strategy: self.name(),
            });
        }
        let compacted = compact_max_messages(messages, self.keep_messages);
        Ok(CompactionOutcome {
            messages: compacted,
            changed: true,
            strategy: self.name(),
        })
    }
}

#[cfg(test)]
mod tests {
    use ovo_types::Message;

    use super::*;

    #[test]
    fn triggers_on_token_estimate() {
        let s = TokenThreshold::new(10, 3).expect("new");
        let msgs = vec![
            Message::system("sys"),
            Message::user("1"),
            Message::user("2"),
            Message::user("3"),
            Message::user("4"),
        ];
        assert!(!s.should_compact(&msgs, 5));
        assert!(s.should_compact(&msgs, 11));
        let out = s.compact(msgs).expect("compact");
        assert!(out.changed);
        assert_eq!(out.messages.len(), 3);
        assert_eq!(
            out.messages.first().map(Message::text).as_deref(),
            Some("sys")
        );
    }

    #[test]
    fn rejects_zero() {
        assert!(TokenThreshold::new(0, 3).is_err());
        assert!(TokenThreshold::new(10, 0).is_err());
    }
}
