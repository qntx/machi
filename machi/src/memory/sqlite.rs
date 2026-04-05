//! SQLite-backed session implementation.
//!
//! [`SqliteSession`] persists conversation history in a `SQLite` database,
//! surviving process restarts. Uses [`rusqlite`] for synchronous access,
//! bridged to async via [`tokio::task::spawn_blocking`].
//!
//! # Storage Model
//!
//! Messages are stored as JSON rows in the `messages` table, ordered by
//! auto-incrementing `id`. WAL journal mode and a composite index on
//! `(session_id, id)` ensure efficient concurrent reads.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, params};

use super::error::MemoryError;
use super::session::Session;
use crate::error::Result;
use crate::message::Message;

/// SQLite-backed session for persistent conversation history.
///
/// Cloneable via `Arc<Mutex<Connection>>` — multiple [`SqliteSession`]
/// handles (even with different session IDs) may share a single database.
///
/// Schema is auto-created on construction. All blocking I/O is offloaded
/// to the tokio blocking thread pool.
#[derive(Debug, Clone)]
pub struct SqliteSession {
    /// Session identifier.
    id: String,
    /// Shared database connection.
    conn: Arc<Mutex<Connection>>,
}

impl SqliteSession {
    /// Opens (or creates) a database at `path` and initializes the schema.
    ///
    /// Pass `":memory:"` for an ephemeral in-process database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or schema initialization fails.
    pub fn open(path: impl AsRef<Path>, session_id: impl Into<String>) -> Result<Self> {
        let conn = Connection::open(path.as_ref()).map_err(MemoryError::from)?;
        Self::from_connection(conn, session_id)
    }

    /// Opens an ephemeral in-memory database (data lost on drop).
    ///
    /// # Errors
    ///
    /// Returns an error if the in-memory database cannot be created.
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(MemoryError::from)?;
        Self::from_connection(conn, session_id)
    }

    /// Wraps an existing [`Connection`], applying pragmas and schema setup.
    ///
    /// Useful for custom connection configuration (encryption, extra pragmas).
    ///
    /// # Errors
    ///
    /// Returns an error if pragma execution or schema setup fails.
    pub fn from_connection(conn: Connection, session_id: impl Into<String>) -> Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;\
             PRAGMA foreign_keys = ON;\
             PRAGMA busy_timeout = 5000;",
        )
        .map_err(MemoryError::from)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                message_data TEXT    NOT NULL,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages (session_id, id);",
        )
        .map_err(MemoryError::from)?;

        Ok(Self {
            id: session_id.into(),
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Bridges a synchronous closure onto the tokio blocking thread pool.
    ///
    /// The closure receives a reference to the locked [`Connection`] and
    /// operates in [`MemoryError`] space; conversion to [`Result`] happens
    /// at the boundary via the double-`?` pattern.
    #[allow(clippy::shadow_unrelated)]
    async fn blocking<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> std::result::Result<T, MemoryError> + Send + 'static,
        T: Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        Ok(tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|e| MemoryError::Lock(e.to_string()))?;
            f(&guard)
        })
        .await
        .map_err(|e| MemoryError::Task(e.to_string()))??)
    }
}

#[async_trait]
impl Session for SqliteSession {
    fn id(&self) -> &str {
        &self.id
    }

    async fn get_messages(&self, limit: Option<usize>) -> Result<Vec<Message>> {
        let session_id = self.id.clone();
        self.blocking(move |conn| {
            let mut messages = if let Some(n) = limit {
                let mut stmt = conn.prepare(
                    "SELECT message_data FROM messages \
                     WHERE session_id = ?1 \
                     ORDER BY id DESC LIMIT ?2",
                )?;

                #[allow(clippy::cast_possible_wrap)]
                let row_limit = n as i64;
                stmt.query_map(params![session_id, row_limit], |row| {
                    row.get::<_, String>(0)
                })?
                .map(|r| Ok(serde_json::from_str::<Message>(&r?)?))
                .collect::<std::result::Result<Vec<_>, MemoryError>>()?
            } else {
                let mut stmt = conn.prepare(
                    "SELECT message_data FROM messages \
                     WHERE session_id = ?1 \
                     ORDER BY id ASC",
                )?;

                stmt.query_map(params![session_id], |row| row.get::<_, String>(0))?
                    .map(|r| Ok(serde_json::from_str::<Message>(&r?)?))
                    .collect::<std::result::Result<Vec<_>, MemoryError>>()?
            };

            if limit.is_some() {
                messages.reverse();
            }

            Ok(messages)
        })
        .await
    }

    async fn add_messages(&self, messages: &[Message]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        let session_id = self.id.clone();

        let serialized = messages
            .iter()
            .map(|m| serde_json::to_string(m).map_err(MemoryError::from))
            .collect::<std::result::Result<Vec<String>, MemoryError>>()?;

        self.blocking(move |conn| {
            let tx = conn.unchecked_transaction()?;

            tx.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?1)",
                params![session_id],
            )?;

            {
                let mut stmt =
                    tx.prepare("INSERT INTO messages (session_id, message_data) VALUES (?1, ?2)")?;

                for json in &serialized {
                    stmt.execute(params![session_id, json])?;
                }
            }

            tx.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP \
                 WHERE session_id = ?1",
                params![session_id],
            )?;

            tx.commit()?;
            Ok(())
        })
        .await
    }

    async fn pop_message(&self) -> Result<Option<Message>> {
        let session_id = self.id.clone();
        self.blocking(move |conn| {
            let json: Option<String> = conn
                .query_row(
                    "DELETE FROM messages \
                     WHERE id = ( \
                         SELECT id FROM messages \
                         WHERE session_id = ?1 \
                         ORDER BY id DESC LIMIT 1 \
                     ) RETURNING message_data",
                    params![session_id],
                    |row| row.get(0),
                )
                .ok();

            match json {
                Some(j) => Ok(Some(serde_json::from_str(&j)?)),
                None => Ok(None),
            }
        })
        .await
    }

    async fn clear(&self) -> Result<()> {
        let session_id = self.id.clone();
        self.blocking(move |conn| {
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "DELETE FROM messages WHERE session_id = ?1",
                params![session_id],
            )?;
            tx.execute(
                "DELETE FROM sessions WHERE session_id = ?1",
                params![session_id],
            )?;
            tx.commit()?;
            Ok(())
        })
        .await
    }

    async fn len(&self) -> Result<usize> {
        let session_id = self.id.clone();
        self.blocking(move |conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )?;

            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            Ok(count as usize)
        })
        .await
    }
}
