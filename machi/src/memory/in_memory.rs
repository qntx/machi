//! In-memory session implementation.
//!
//! [`InMemorySession`] stores conversation history in a `Vec<Message>` behind
//! a `tokio::sync::RwLock`. Data is lost when the process exits.
//!
//! Best suited for single-run agents, testing, and short-lived conversations.

use async_trait::async_trait;
use tokio::sync::RwLock;

use super::session::Session;
use crate::error::Result;
use crate::message::Message;

/// In-memory session backed by `tokio::sync::RwLock<Vec<Message>>`.
///
/// Concurrent readers may retrieve history simultaneously; writes acquire
/// exclusive access. All data is ephemeral — lost when the value is dropped.
#[derive(Debug)]
pub struct InMemorySession {
    /// Session identifier.
    id: String,
    /// Message storage.
    messages: RwLock<Vec<Message>>,
}

impl InMemorySession {
    /// Creates an empty session.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: RwLock::new(Vec::new()),
        }
    }

    /// Creates a session pre-populated with `messages`.
    #[must_use]
    pub fn with_messages(id: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            id: id.into(),
            messages: RwLock::new(messages),
        }
    }

    /// Creates an empty session with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(id: impl Into<String>, capacity: usize) -> Self {
        Self {
            id: id.into(),
            messages: RwLock::new(Vec::with_capacity(capacity)),
        }
    }
}

#[async_trait]
impl Session for InMemorySession {
    fn id(&self) -> &str {
        &self.id
    }

    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<Message>> {
        let guard = self.messages.read().await;
        match limit {
            Some(n) if n < guard.len() => Ok(guard[guard.len() - n..].to_vec()),
            _ => Ok(guard.clone()),
        }
    }

    async fn add_messages(&self, messages: &[Message]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        self.messages.write().await.extend(messages.iter().cloned());
        Ok(())
    }

    async fn pop_message(&self) -> Result<Option<Message>> {
        Ok(self.messages.write().await.pop())
    }

    async fn clear(&self) -> Result<()> {
        self.messages.write().await.clear();
        Ok(())
    }

    async fn len(&self) -> Result<usize> {
        Ok(self.messages.read().await.len())
    }
}
