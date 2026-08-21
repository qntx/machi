//! Conversation state handle with strict append invariants and usage ledger.
//!
//! The handle is the external API; mutations are serialized through an internal
//! actor so concurrent hosts cannot corrupt tool-call pairing.

#![forbid(unsafe_code)]

pub mod file_store;
pub mod handle;
pub mod jsonl_store;
pub mod ledger;
pub mod memory;
pub mod persistence;
pub mod strict;

pub use file_store::{DEFAULT_SESSIONS_DIR, FilePersistence, default_session_path};
pub use handle::{ChatStateHandle, ChatStateSnapshot};
pub use jsonl_store::{DEFAULT_SNAPSHOT_EVERY, EVENTS_HEADER, JsonlPersistence, session_jsonl_dir};
pub use ledger::{CompactionRecord, UsageLedger};
pub use memory::{InMemoryMemory, MemoryItem, MemoryPort, NullMemory};
pub use persistence::{ChatPersistence, MemoryPersistence, NullPersistence, messages_only};
pub use strict::{StrictAppendError, check_append, check_tool_pairing};
