//! Persistence ports for conversation snapshots (incremental append).

use async_trait::async_trait;
use ovo_types::{Message, OvoError};

use crate::handle::ChatStateSnapshot;

/// Host-provided persistence backend.
#[async_trait]
pub trait ChatPersistence: Send + Sync {
    /// Persist a full snapshot.
    async fn save(&self, snapshot: &ChatStateSnapshot) -> Result<(), OvoError>;
    /// Load the latest snapshot when present.
    async fn load(&self) -> Result<Option<ChatStateSnapshot>, OvoError>;
    /// Append a single message (incremental). Default: load → push → save.
    ///
    /// # Errors
    ///
    /// Backend I/O failures.
    async fn persist_message(&self, message: &Message) -> Result<(), OvoError> {
        let mut snap = self.load().await?.unwrap_or_else(|| messages_only(vec![]));
        snap.messages.push(message.clone());
        if message.role == ovo_types::Role::User {
            snap.prompt_index
                .push(snap.messages.len().saturating_sub(1));
        }
        self.save(&snap).await
    }
}

/// No-op persistence.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullPersistence;

#[async_trait]
impl ChatPersistence for NullPersistence {
    async fn save(&self, _snapshot: &ChatStateSnapshot) -> Result<(), OvoError> {
        Ok(())
    }
    async fn load(&self) -> Result<Option<ChatStateSnapshot>, OvoError> {
        Ok(None)
    }
    async fn persist_message(&self, _message: &Message) -> Result<(), OvoError> {
        Ok(())
    }
}

/// In-memory persistence for tests.
#[derive(Debug, Default)]
pub struct MemoryPersistence {
    slot: tokio::sync::Mutex<Option<ChatStateSnapshot>>,
}

#[async_trait]
impl ChatPersistence for MemoryPersistence {
    async fn save(&self, snapshot: &ChatStateSnapshot) -> Result<(), OvoError> {
        *self.slot.lock().await = Some(snapshot.clone());
        Ok(())
    }
    async fn load(&self) -> Result<Option<ChatStateSnapshot>, OvoError> {
        Ok(self.slot.lock().await.clone())
    }
}

/// Helper: messages-only snapshot body.
#[must_use]
pub fn messages_only(messages: Vec<Message>) -> ChatStateSnapshot {
    let prompt_index = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == ovo_types::Role::User).then_some(i))
        .collect();
    ChatStateSnapshot {
        messages,
        usage: crate::ledger::UsageLedger::default(),
        prompt_index,
    }
}
