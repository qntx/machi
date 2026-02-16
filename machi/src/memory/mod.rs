//! Session-based memory for AI agents.
//!
//! The [`Session`] trait abstracts over storage backends, treating the
//! **message list as the single source of truth** for conversation history.
//! Agents stay stateless — all context lives in the session.
//!
//! # Available Backends
//!
//! | Type | Persistence | Feature |
//! |------|-------------|---------|
//! | [`InMemorySession`] | None (ephemeral) | always |
//! | [`SqliteSession`] | File or `:memory:` | `memory-sqlite` |
//!
//! # Quick Start
//!
//! ```rust
//! # tokio_test::block_on(async {
//! use machi::memory::{Session, InMemorySession};
//! use machi::message::Message;
//!
//! let session = InMemorySession::new("conv-1");
//!
//! session.add_messages(&[
//!     Message::system("You are a helpful assistant."),
//!     Message::user("What is Rust?"),
//! ]).await?;
//!
//! let history = session.get_messages(None).await?;
//! assert_eq!(history.len(), 2);
//!
//! let removed = session.pop_message().await?;
//! assert!(removed.is_some());
//! # Ok::<(), machi::Error>(())
//! # }).unwrap();
//! ```

mod error;
mod in_memory;
mod session;

#[cfg(feature = "memory-sqlite")]
mod sqlite;

pub use error::MemoryError;
pub use in_memory::InMemorySession;
pub use session::{BoxedSession, Session, SharedSession};
#[cfg(feature = "memory-sqlite")]
pub use sqlite::SqliteSession;
