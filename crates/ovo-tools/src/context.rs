//! Per-call execution context.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ovo_protocol::{TurnEvent, TurnEventKind};
use ovo_types::{AgentId, Deadline, RunId, SessionId};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Extra key: nesting depth of the agent that owns this tool call
/// (`0` = first host-spawned level). Used by `spawn_agent` to fail-closed on depth.
pub const EXTRA_SPAWN_DEPTH: &str = "ovo.spawn_depth";

/// Shared live-event bus for a turn (optional).
///
/// Clone is cheap (`Arc` seq + channel sender). When absent, tools stay silent.
#[derive(Debug, Clone)]
pub struct EventBus {
    tx: mpsc::UnboundedSender<TurnEvent>,
    run_id: RunId,
    seq: Arc<AtomicU64>,
}

impl EventBus {
    /// Create a bus bound to `run_id` with a fresh sequence counter.
    #[must_use]
    pub fn new(tx: mpsc::UnboundedSender<TurnEvent>, run_id: RunId) -> Self {
        Self {
            tx,
            run_id,
            seq: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Run id for this bus.
    #[must_use]
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// Underlying sender (for nested spawn opts).
    #[must_use]
    pub fn sender(&self) -> mpsc::UnboundedSender<TurnEvent> {
        self.tx.clone()
    }

    /// Shared sequence counter (keep one per turn across sinks).
    #[must_use]
    pub fn seq_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.seq)
    }

    /// Emit one event with monotonic `seq`.
    pub fn emit(&self, agent_id: Option<AgentId>, depth: Option<u32>, kind: TurnEventKind) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let _ = self.tx.send(TurnEvent {
            run_id: self.run_id.clone(),
            seq,
            agent_id,
            depth,
            kind,
        });
    }
}

/// Context passed into every tool invocation.
#[derive(Debug, Clone)]
pub struct ToolCallContext {
    /// Cancellation token for the call / turn.
    pub cancel: CancellationToken,
    /// Optional absolute deadline.
    pub deadline: Option<Deadline>,
    /// Working directory for relative paths.
    pub cwd: Option<PathBuf>,
    /// Session id when known.
    pub session_id: Option<SessionId>,
    /// Agent id when known.
    pub agent_id: Option<AgentId>,
    /// Host-defined extensions (stringly map).
    pub extras: Arc<HashMap<String, String>>,
    /// Optional live event bus for this turn.
    pub events: Option<EventBus>,
}

impl Default for ToolCallContext {
    fn default() -> Self {
        Self {
            cancel: CancellationToken::new(),
            deadline: None,
            cwd: None,
            session_id: None,
            agent_id: None,
            extras: Arc::new(HashMap::new()),
            events: None,
        }
    }
}

impl ToolCallContext {
    /// Builder: set cancel token.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Builder: set deadline.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Deadline) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Builder: replace extras map.
    #[must_use]
    pub fn with_extras(mut self, extras: HashMap<String, String>) -> Self {
        self.extras = Arc::new(extras);
        self
    }

    /// Builder: attach event bus.
    #[must_use]
    pub fn with_events(mut self, events: EventBus) -> Self {
        self.events = Some(events);
        self
    }

    /// Read nesting depth of the current agent (`None` = top-level session turn).
    #[must_use]
    pub fn spawn_depth(&self) -> Option<u32> {
        self.extras
            .get(EXTRA_SPAWN_DEPTH)
            .and_then(|s| s.parse().ok())
    }

    /// True when cancel requested or deadline expired.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled() || self.deadline.is_some_and(|d| d.is_expired())
    }

    /// Emit a turn event when a bus is attached.
    pub fn emit(&self, kind: TurnEventKind) {
        if let Some(bus) = &self.events {
            bus.emit(self.agent_id.clone(), self.spawn_depth(), kind);
        }
    }
}
