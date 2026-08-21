//! Live turn observation surface (`TurnEvent`).
//!
//! Every effect that matters to a host should appear here (event completeness).

use ovo_types::{AgentId, RunId};
use serde::{Deserialize, Serialize};

/// One ordered event from a turn or nested spawn tree.
///
/// Invariants (per `run_id`):
/// - `seq` is strictly monotonic starting at 0
/// - every `*Start` has exactly one matching end (`*End` / `Finished` / `Aborted`)
/// - deltas never follow their terminal event for the same logical unit
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnEvent {
    /// Run that owns this event (parent turn id for spawn lifecycle on the parent stream).
    pub run_id: RunId,
    /// Monotonic sequence within `run_id`.
    pub seq: u64,
    /// Agent executing the turn when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<AgentId>,
    /// Nesting depth (`None` = top-level session turn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
    /// Payload.
    pub kind: TurnEventKind,
}

/// Payload of a [`TurnEvent`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TurnEventKind {
    /// Turn loop entered.
    TurnStarted,
    /// Agent step about to sample (`step` is one-based).
    StepStarted {
        /// One-based step index.
        step: u32,
    },
    /// Assistant text delta (stream path).
    TextDelta {
        /// Incremental text.
        text: String,
    },
    /// Reasoning / chain-of-thought delta (stream path).
    ReasoningDelta {
        /// Incremental text.
        text: String,
    },
    /// Model planned a tool call (before dispatch).
    ToolCallPlanned {
        /// Tool call id.
        id: String,
        /// Tool name.
        name: String,
    },
    /// Tool execution started.
    ToolExecutionStart {
        /// Tool call id.
        id: String,
        /// Tool name.
        name: String,
    },
    /// Tool progress frame.
    ToolExecutionUpdate {
        /// Tool call id.
        id: String,
        /// Tool name.
        name: String,
        /// Human-readable progress fragment.
        message: String,
    },
    /// Tool execution finished.
    ToolExecutionEnd {
        /// Tool call id.
        id: String,
        /// Tool name.
        name: String,
        /// Whether the tool result is an error payload.
        is_error: bool,
    },
    /// Nested agent spawn admitted (emitted on the **parent** stream).
    SpawnStarted {
        /// Child agent id.
        child_agent_id: String,
        /// Optional label.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        /// Child depth.
        depth: u32,
    },
    /// Nested agent spawn finished (emitted on the **parent** stream).
    SpawnFinished {
        /// Child agent id.
        child_agent_id: String,
        /// Optional label.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        /// Child depth.
        depth: u32,
        /// Success flag (not cancelled and completed).
        success: bool,
        /// Cancelled flag.
        cancelled: bool,
    },
    /// Compaction strategy applied to conversation state.
    CompactionApplied {
        /// Strategy name.
        strategy: String,
    },
    /// Mid-turn interjection applied.
    InterjectionApplied,
    /// Stationarity nudge injected.
    StationarityNudge,
    /// Turn completed normally (including cancelled-with-outcome paths).
    TurnFinished {
        /// Steps executed.
        steps: u32,
        /// Whether the turn ended cancelled.
        cancelled: bool,
    },
    /// Turn aborted with a typed failure (no successful outcome).
    TurnAborted {
        /// Short reason string.
        reason: String,
    },
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip_text_delta() {
        let ev = TurnEvent {
            run_id: RunId::generate(),
            seq: 3,
            agent_id: None,
            depth: Some(0),
            kind: TurnEventKind::TextDelta { text: "hi".into() },
        };
        let raw = serde_json::to_string(&ev).expect("ser");
        let back: TurnEvent = serde_json::from_str(&raw).expect("de");
        assert_eq!(back.seq, 3);
        assert_eq!(back.kind, TurnEventKind::TextDelta { text: "hi".into() });
    }
}
