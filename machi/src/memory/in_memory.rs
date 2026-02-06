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
    id: String,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::message::{Message, Role};

    /// Returns a deterministic message list for testing.
    fn sample_messages(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| match i % 3 {
                0 => Message::system(format!("system-{i}")),
                1 => Message::user(format!("user-{i}")),
                _ => Message::assistant(format!("assistant-{i}")),
            })
            .collect()
    }

    mod construction {
        use super::*;

        #[test]
        fn new_creates_empty_session() {
            let session = InMemorySession::new("s-1");
            assert_eq!(session.id, "s-1");
        }

        #[test]
        fn with_messages_populates_history() {
            let msgs = sample_messages(3);
            let session = InMemorySession::with_messages("s-2", msgs.clone());
            assert_eq!(session.id, "s-2");
            // Verify inner vec length synchronously via blocking lock.
            let inner = session.messages.blocking_read();
            assert_eq!(inner.len(), 3);
            assert_eq!(*inner, msgs);
        }

        #[test]
        fn with_capacity_preallocates() {
            let session = InMemorySession::with_capacity("s-3", 64);
            assert_eq!(session.id, "s-3");
            let inner = session.messages.blocking_read();
            assert!(inner.capacity() >= 64);
            assert!(inner.is_empty());
        }

        #[test]
        fn id_accepts_string_types() {
            // &str
            let _ = InMemorySession::new("literal");
            // String
            let _ = InMemorySession::new(String::from("owned"));
            // Format macro
            let _ = InMemorySession::new(format!("fmt-{}", 42));
        }
    }

    mod get_messages {
        use super::*;

        #[tokio::test]
        async fn returns_all_when_limit_is_none() {
            let msgs = sample_messages(5);
            let session = InMemorySession::with_messages("g1", msgs.clone());
            let result = session.get_messages(None).await.unwrap();
            assert_eq!(result, msgs);
        }

        #[tokio::test]
        async fn returns_empty_vec_for_empty_session() {
            let session = InMemorySession::new("g2");
            let result = session.get_messages(None).await.unwrap();
            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn limit_returns_latest_n_messages() {
            let msgs = sample_messages(5);
            let session = InMemorySession::with_messages("g3", msgs.clone());

            let last2 = session.get_messages(Some(2)).await.unwrap();
            assert_eq!(last2.len(), 2);
            assert_eq!(last2, msgs[3..5]);
        }

        #[tokio::test]
        async fn limit_greater_than_len_returns_all() {
            let msgs = sample_messages(3);
            let session = InMemorySession::with_messages("g4", msgs.clone());

            let result = session.get_messages(Some(100)).await.unwrap();
            assert_eq!(result, msgs);
        }

        #[tokio::test]
        async fn limit_equal_to_len_returns_all() {
            let msgs = sample_messages(4);
            let session = InMemorySession::with_messages("g5", msgs.clone());

            let result = session.get_messages(Some(4)).await.unwrap();
            assert_eq!(result, msgs);
        }

        #[tokio::test]
        async fn limit_zero_returns_empty() {
            let msgs = sample_messages(3);
            let session = InMemorySession::with_messages("g6", msgs);

            let result = session.get_messages(Some(0)).await.unwrap();
            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn limit_one_returns_last_message() {
            let msgs = sample_messages(5);
            let session = InMemorySession::with_messages("g7", msgs.clone());

            let result = session.get_messages(Some(1)).await.unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], msgs[4]);
        }
    }

    mod add_messages {
        use super::*;

        #[tokio::test]
        async fn appends_in_order() {
            let session = InMemorySession::new("a1");
            let batch = sample_messages(3);
            session.add_messages(&batch).await.unwrap();

            let stored = session.get_messages(None).await.unwrap();
            assert_eq!(stored, batch);
        }

        #[tokio::test]
        async fn multiple_adds_preserve_chronological_order() {
            let session = InMemorySession::new("a2");
            session
                .add_messages(&[Message::system("sys")])
                .await
                .unwrap();
            session.add_messages(&[Message::user("usr")]).await.unwrap();
            session
                .add_messages(&[Message::assistant("ast")])
                .await
                .unwrap();

            let stored = session.get_messages(None).await.unwrap();
            assert_eq!(stored.len(), 3);
            assert_eq!(stored[0].role, Role::System);
            assert_eq!(stored[1].role, Role::User);
            assert_eq!(stored[2].role, Role::Assistant);
        }

        #[tokio::test]
        async fn empty_slice_is_noop() {
            let session = InMemorySession::with_messages("a3", sample_messages(2));
            session.add_messages(&[]).await.unwrap();
            assert_eq!(session.len().await.unwrap(), 2);
        }

        #[tokio::test]
        async fn large_batch_add() {
            let session = InMemorySession::new("a4");
            let batch = sample_messages(1000);
            session.add_messages(&batch).await.unwrap();
            assert_eq!(session.len().await.unwrap(), 1000);
        }
    }

    mod pop_message {
        use super::*;

        #[tokio::test]
        async fn pops_last_message() {
            let msgs = sample_messages(3);
            let session = InMemorySession::with_messages("p1", msgs.clone());

            let popped = session.pop_message().await.unwrap();
            assert_eq!(popped, Some(msgs[2].clone()));
            assert_eq!(session.len().await.unwrap(), 2);
        }

        #[tokio::test]
        async fn returns_none_on_empty_session() {
            let session = InMemorySession::new("p2");
            let popped = session.pop_message().await.unwrap();
            assert_eq!(popped, None);
        }

        #[tokio::test]
        async fn successive_pops_drain_in_lifo_order() {
            let msgs = sample_messages(3);
            let session = InMemorySession::with_messages("p3", msgs.clone());

            for expected in msgs.iter().rev() {
                let popped = session.pop_message().await.unwrap().unwrap();
                assert_eq!(popped, *expected);
            }

            assert!(session.is_empty().await.unwrap());
            assert_eq!(session.pop_message().await.unwrap(), None);
        }
    }

    mod clear {
        use super::*;

        #[tokio::test]
        async fn removes_all_messages() {
            let session = InMemorySession::with_messages("c1", sample_messages(5));
            session.clear().await.unwrap();

            assert!(session.is_empty().await.unwrap());
            assert_eq!(session.len().await.unwrap(), 0);
            assert_eq!(session.get_messages(None).await.unwrap(), vec![]);
        }

        #[tokio::test]
        async fn clear_on_empty_session_is_idempotent() {
            let session = InMemorySession::new("c2");
            session.clear().await.unwrap();
            session.clear().await.unwrap();
            assert!(session.is_empty().await.unwrap());
        }
    }

    mod len_and_is_empty {
        use super::*;

        #[tokio::test]
        async fn len_reflects_mutations() {
            let session = InMemorySession::new("l1");
            assert_eq!(session.len().await.unwrap(), 0);

            session.add_messages(&sample_messages(3)).await.unwrap();
            assert_eq!(session.len().await.unwrap(), 3);

            session.pop_message().await.unwrap();
            assert_eq!(session.len().await.unwrap(), 2);

            session.clear().await.unwrap();
            assert_eq!(session.len().await.unwrap(), 0);
        }

        #[tokio::test]
        async fn is_empty_default_impl() {
            let session = InMemorySession::new("l2");
            assert!(session.is_empty().await.unwrap());

            session.add_messages(&[Message::user("hi")]).await.unwrap();
            assert!(!session.is_empty().await.unwrap());
        }
    }

    mod trait_objects {
        use super::*;

        #[tokio::test]
        async fn works_as_boxed_session() {
            let boxed: Box<dyn Session> = Box::new(InMemorySession::new("box-1"));
            boxed.add_messages(&[Message::user("hello")]).await.unwrap();
            assert_eq!(boxed.len().await.unwrap(), 1);
            assert_eq!(boxed.id(), "box-1");
        }

        #[tokio::test]
        async fn works_as_shared_session() {
            let shared: Arc<dyn Session> = Arc::new(InMemorySession::new("arc-1"));
            shared
                .add_messages(&[Message::user("hello")])
                .await
                .unwrap();
            assert_eq!(shared.len().await.unwrap(), 1);
        }
    }

    mod concurrency {
        use super::*;

        #[tokio::test]
        async fn concurrent_readers_do_not_block() {
            let session = Arc::new(InMemorySession::with_messages("cr-1", sample_messages(10)));

            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let s = Arc::clone(&session);
                    tokio::spawn(async move { s.get_messages(None).await.unwrap().len() })
                })
                .collect();

            for handle in handles {
                assert_eq!(handle.await.unwrap(), 10);
            }
        }

        #[tokio::test]
        async fn concurrent_writers_are_serialized() {
            let session = Arc::new(InMemorySession::new("cw-1"));

            let handles: Vec<_> = (0..50)
                .map(|i| {
                    let s = Arc::clone(&session);
                    tokio::spawn(async move {
                        s.add_messages(&[Message::user(format!("msg-{i}"))])
                            .await
                            .unwrap();
                    })
                })
                .collect();

            for handle in handles {
                handle.await.unwrap();
            }

            // All 50 messages must be present regardless of scheduling order.
            assert_eq!(session.len().await.unwrap(), 50);
        }

        #[tokio::test]
        async fn mixed_read_write_operations() {
            let session = Arc::new(InMemorySession::new("mx-1"));

            // Phase 1: concurrent writes
            let write_handles: Vec<_> = (0..20)
                .map(|i| {
                    let s = Arc::clone(&session);
                    tokio::spawn(async move {
                        s.add_messages(&[Message::user(format!("w-{i}"))])
                            .await
                            .unwrap();
                    })
                })
                .collect();

            for h in write_handles {
                h.await.unwrap();
            }
            assert_eq!(session.len().await.unwrap(), 20);

            // Phase 2: concurrent reads see a consistent snapshot
            let read_handles: Vec<_> = (0..10)
                .map(|_| {
                    let s = Arc::clone(&session);
                    tokio::spawn(async move { s.get_messages(Some(5)).await.unwrap().len() })
                })
                .collect();

            for h in read_handles {
                assert_eq!(h.await.unwrap(), 5);
            }
        }
    }

    mod edge_cases {
        use super::*;

        #[tokio::test]
        async fn session_with_tool_messages() {
            let session = InMemorySession::new("ec-1");
            let tool_msg = Message::tool("call-123", r#"{"result": 42}"#);
            session
                .add_messages(std::slice::from_ref(&tool_msg))
                .await
                .unwrap();

            let stored = session.get_messages(None).await.unwrap();
            assert_eq!(stored.len(), 1);
            assert_eq!(stored[0].role, Role::Tool);
            assert_eq!(stored[0].tool_call_id.as_deref(), Some("call-123"));
        }

        #[tokio::test]
        async fn add_then_pop_then_add_again() {
            let session = InMemorySession::new("ec-2");

            session
                .add_messages(&[Message::user("first")])
                .await
                .unwrap();
            session.pop_message().await.unwrap();
            assert!(session.is_empty().await.unwrap());

            session
                .add_messages(&[Message::user("second")])
                .await
                .unwrap();
            let msgs = session.get_messages(None).await.unwrap();
            assert_eq!(msgs.len(), 1);
            assert_eq!(msgs[0].text().unwrap(), "second");
        }

        #[tokio::test]
        async fn clear_then_add_works() {
            let session = InMemorySession::with_messages("ec-3", sample_messages(5));
            session.clear().await.unwrap();
            session
                .add_messages(&[Message::system("fresh")])
                .await
                .unwrap();

            let msgs = session.get_messages(None).await.unwrap();
            assert_eq!(msgs.len(), 1);
            assert_eq!(msgs[0].role, Role::System);
        }

        #[tokio::test]
        async fn empty_string_session_id() {
            let session = InMemorySession::new("");
            assert_eq!(session.id(), "");
        }

        #[tokio::test]
        async fn unicode_session_id() {
            let session = InMemorySession::new("会话-αβγ-🦀");
            assert_eq!(session.id(), "会话-αβγ-🦀");
        }
    }
}
