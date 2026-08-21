//! Identical tool-call stationarity protection.
//!
//!
//! Tracks consecutive identical `(tool_name, args_hash)` batches. At
//! [`NUDGE_THRESHOLD`] injects a reminder; at [`HARD_STOP_THRESHOLD`] fails
//! with [`ErrorCode::RuntimeStationarity`].

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use ovo_types::{ErrorCode, Message, OvoError, ToolCall};

/// Consecutive identical tool rounds before a soft nudge reminder.
pub const NUDGE_THRESHOLD: u32 = 8;
/// Consecutive identical tool rounds before a hard stop.
pub const HARD_STOP_THRESHOLD: u32 = 16;

/// Tracks stationarity across tool-call steps within a turn.
#[derive(Debug, Clone, Copy, Default)]
pub struct StationarityTracker {
    last_fingerprint: Option<u64>,
    streak: u32,
    nudged: bool,
}

/// Action after observing a tool-call batch.
#[derive(Debug, Clone)]
pub enum StationarityAction {
    /// Continue normally.
    Ok,
    /// Inject reminder once at the nudge threshold.
    Nudge {
        /// Reminder text for the model.
        reminder: String,
    },
    /// Abort the turn.
    HardStop {
        /// Typed error.
        error: OvoError,
    },
}

impl StationarityTracker {
    /// Empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current streak length (for tests).
    #[must_use]
    pub const fn streak(&self) -> u32 {
        self.streak
    }

    /// Observe a non-empty tool-call batch after the assistant message is known.
    pub fn observe_tool_batch(&mut self, calls: &[ToolCall]) -> StationarityAction {
        if calls.is_empty() {
            self.reset();
            return StationarityAction::Ok;
        }
        let fp = fingerprint_batch(calls);
        if self.last_fingerprint == Some(fp) {
            self.streak = self.streak.saturating_add(1);
        } else {
            self.last_fingerprint = Some(fp);
            self.streak = 1;
            self.nudged = false;
        }

        if self.streak >= HARD_STOP_THRESHOLD {
            return StationarityAction::HardStop {
                error: OvoError::new(
                    ErrorCode::RuntimeStationarity,
                    format!(
                        "identical tool calls repeated {HARD_STOP_THRESHOLD} times (stationarity hard stop)"
                    ),
                ),
            };
        }
        if self.streak >= NUDGE_THRESHOLD && !self.nudged {
            self.nudged = true;
            return StationarityAction::Nudge {
                reminder: format!(
                    "You have called the same tool(s) with the same arguments {NUDGE_THRESHOLD} times \
                     in a row. Change strategy, use different tools, or finish without repeating."
                ),
            };
        }
        StationarityAction::Ok
    }

    /// Reset when the model produces a final (non-tool) message.
    pub fn reset(&mut self) {
        self.last_fingerprint = None;
        self.streak = 0;
        self.nudged = false;
    }
}

/// Stable fingerprint for a batch of tool calls (order-sensitive).
#[must_use]
pub fn fingerprint_batch(calls: &[ToolCall]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for call in calls {
        call.name.hash(&mut hasher);
        // Canonical-ish: JSON Display is stable enough for identity of identical Value trees.
        call.arguments.to_string().hash(&mut hasher);
    }
    hasher.finish()
}

/// User message carrying a stationarity nudge (for injection into the buffer).
#[must_use]
pub fn nudge_message(reminder: String) -> Message {
    Message::user(reminder)
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use ovo_types::ToolCallId;
    use serde_json::json;

    use super::*;

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: ToolCallId::new("c1").expect("id"),
            name: name.into(),
            arguments: args,
        }
    }

    #[test]
    fn resets_on_different_call() {
        let mut t = StationarityTracker::new();
        for _ in 0..5 {
            assert!(matches!(
                t.observe_tool_batch(&[call("a", json!({"x": 1}))]),
                StationarityAction::Ok
            ));
        }
        assert_eq!(t.streak(), 5);
        assert!(matches!(
            t.observe_tool_batch(&[call("b", json!({"x": 1}))]),
            StationarityAction::Ok
        ));
        assert_eq!(t.streak(), 1);
    }

    #[test]
    fn nudge_then_hard_stop() {
        let mut t = StationarityTracker::new();
        let batch = [call("calc", json!({"expr": "1+1"}))];
        for i in 1..NUDGE_THRESHOLD {
            let a = t.observe_tool_batch(&batch);
            assert!(matches!(a, StationarityAction::Ok), "i={i}");
        }
        assert!(matches!(
            t.observe_tool_batch(&batch),
            StationarityAction::Nudge { .. }
        ));
        for _ in (NUDGE_THRESHOLD + 1)..HARD_STOP_THRESHOLD {
            let _ = t.observe_tool_batch(&batch);
        }
        assert!(matches!(
            t.observe_tool_batch(&batch),
            StationarityAction::HardStop { .. }
        ));
    }
}
