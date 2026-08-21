//! Optional live [`TurnEvent`] emission for embedders.

use ovo_protocol::{TurnEvent, TurnEventKind};
use ovo_tools::EventBus;
use ovo_types::{AgentId, RunId};
use tokio::sync::mpsc;

/// Sink that forwards turn events with fixed agent/depth identity.
#[derive(Debug, Clone)]
pub struct EventSink {
    bus: Option<EventBus>,
    agent_id: Option<AgentId>,
    depth: Option<u32>,
}

impl EventSink {
    /// Bind optional channel for `run_id`.
    #[must_use]
    pub fn from_tx(
        tx: Option<mpsc::UnboundedSender<TurnEvent>>,
        run_id: RunId,
        agent_id: Option<AgentId>,
        depth: Option<u32>,
    ) -> Self {
        let bus = tx.map(|t| EventBus::new(t, run_id));
        Self {
            bus,
            agent_id,
            depth,
        }
    }

    /// Use an existing bus (keeps parent `seq` monotonic).
    #[must_use]
    pub fn from_bus(bus: EventBus, agent_id: Option<AgentId>, depth: Option<u32>) -> Self {
        Self {
            bus: Some(bus),
            agent_id,
            depth,
        }
    }

    /// Clone of the bus for tool context / spawn opts.
    #[must_use]
    pub fn bus_cloned(&self) -> Option<EventBus> {
        self.bus.clone()
    }

    /// Emit one event.
    pub fn emit(&self, kind: TurnEventKind) {
        if let Some(bus) = &self.bus {
            bus.emit(self.agent_id.clone(), self.depth, kind);
        }
    }
}
