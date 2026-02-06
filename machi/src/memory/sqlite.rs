//! SQLite-backed session implementation.
//!
//! [`SqliteSession`] persists conversation history in a SQLite database,
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
    id: String,
    conn: Arc<Mutex<Connection>>,
}

impl SqliteSession {
    /// Opens (or creates) a database at `path` and initializes the schema.
    ///
    /// Pass `":memory:"` for an ephemeral in-process database.
    pub fn open(path: impl AsRef<Path>, session_id: impl Into<String>) -> Result<Self> {
        let conn = Connection::open(path.as_ref()).map_err(MemoryError::from)?;
        Self::from_connection(conn, session_id)
    }

    /// Opens an ephemeral in-memory database (data lost on drop).
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(MemoryError::from)?;
        Self::from_connection(conn, session_id)
    }

    /// Wraps an existing [`Connection`], applying pragmas and schema setup.
    ///
    /// Useful for custom connection configuration (encryption, extra pragmas).
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

                stmt.query_map(params![session_id, n], |row| row.get::<_, String>(0))?
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::message::{Message, Role};

    /// Creates a fresh in-memory `SqliteSession` for test isolation.
    fn new_session(id: &str) -> SqliteSession {
        SqliteSession::in_memory(id).unwrap()
    }

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
        fn in_memory_creates_session() {
            let session = SqliteSession::in_memory("s-1").unwrap();
            assert_eq!(session.id, "s-1");
        }

        #[test]
        fn from_connection_applies_schema() {
            let conn = Connection::open_in_memory().unwrap();
            let session = SqliteSession::from_connection(conn, "s-2").unwrap();
            assert_eq!(session.id, "s-2");

            // Verify tables exist by querying sqlite_master.
            let guard = session.conn.lock().unwrap();
            let tables: Vec<String> = guard
                .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .unwrap()
                .query_map([], |row| row.get(0))
                .unwrap()
                .filter_map(std::result::Result::ok)
                .collect();
            assert!(tables.contains(&"sessions".to_owned()));
            assert!(tables.contains(&"messages".to_owned()));
        }

        #[test]
        fn open_with_temp_file() {
            let dir = std::env::temp_dir().join("machi_test_sqlite");
            std::fs::create_dir_all(&dir).unwrap();
            let db_path = dir.join("test_open.db");

            let session = SqliteSession::open(&db_path, "file-1").unwrap();
            assert_eq!(session.id(), "file-1");

            // Cleanup
            drop(session);
            let _ = std::fs::remove_file(&db_path);
            let _ = std::fs::remove_dir(&dir);
        }

        #[test]
        fn clone_shares_connection() {
            let session = new_session("clone-1");
            let cloned = session.clone();

            // Both point to the same Arc<Mutex<Connection>>.
            assert!(Arc::ptr_eq(&session.conn, &cloned.conn));
            assert_eq!(cloned.id, "clone-1");
        }

        #[test]
        fn id_accepts_string_types() {
            let _ = SqliteSession::in_memory("literal").unwrap();
            let _ = SqliteSession::in_memory(String::from("owned")).unwrap();
            let _ = SqliteSession::in_memory(format!("fmt-{}", 42)).unwrap();
        }
    }

    mod get_messages {
        use super::*;

        #[tokio::test]
        async fn returns_all_when_limit_is_none() {
            let session = new_session("g1");
            let msgs = sample_messages(5);
            session.add_messages(&msgs).await.unwrap();

            let result = session.get_messages(None).await.unwrap();
            assert_eq!(result, msgs);
        }

        #[tokio::test]
        async fn returns_empty_vec_for_empty_session() {
            let session = new_session("g2");
            let result = session.get_messages(None).await.unwrap();
            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn limit_returns_latest_n_in_chronological_order() {
            let session = new_session("g3");
            let msgs = sample_messages(5);
            session.add_messages(&msgs).await.unwrap();

            let last2 = session.get_messages(Some(2)).await.unwrap();
            assert_eq!(last2.len(), 2);
            assert_eq!(last2, msgs[3..5]);
        }

        #[tokio::test]
        async fn limit_greater_than_len_returns_all() {
            let session = new_session("g4");
            let msgs = sample_messages(3);
            session.add_messages(&msgs).await.unwrap();

            let result = session.get_messages(Some(100)).await.unwrap();
            assert_eq!(result, msgs);
        }

        #[tokio::test]
        async fn limit_equal_to_len_returns_all() {
            let session = new_session("g5");
            let msgs = sample_messages(4);
            session.add_messages(&msgs).await.unwrap();

            let result = session.get_messages(Some(4)).await.unwrap();
            assert_eq!(result, msgs);
        }

        #[tokio::test]
        async fn limit_zero_returns_empty() {
            let session = new_session("g6");
            session.add_messages(&sample_messages(3)).await.unwrap();

            let result = session.get_messages(Some(0)).await.unwrap();
            assert!(result.is_empty());
        }

        #[tokio::test]
        async fn limit_one_returns_last_message() {
            let session = new_session("g7");
            let msgs = sample_messages(5);
            session.add_messages(&msgs).await.unwrap();

            let result = session.get_messages(Some(1)).await.unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], msgs[4]);
        }
    }

    mod add_messages {
        use super::*;

        #[tokio::test]
        async fn appends_in_order() {
            let session = new_session("a1");
            let batch = sample_messages(3);
            session.add_messages(&batch).await.unwrap();

            let stored = session.get_messages(None).await.unwrap();
            assert_eq!(stored, batch);
        }

        #[tokio::test]
        async fn multiple_adds_preserve_chronological_order() {
            let session = new_session("a2");
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
            let session = new_session("a3");
            session.add_messages(&sample_messages(2)).await.unwrap();
            session.add_messages(&[]).await.unwrap();
            assert_eq!(session.len().await.unwrap(), 2);
        }

        #[tokio::test]
        async fn large_batch_add() {
            let session = new_session("a4");
            let batch = sample_messages(500);
            session.add_messages(&batch).await.unwrap();
            assert_eq!(session.len().await.unwrap(), 500);
        }
    }

    mod pop_message {
        use super::*;

        #[tokio::test]
        async fn pops_last_message() {
            let session = new_session("p1");
            let msgs = sample_messages(3);
            session.add_messages(&msgs).await.unwrap();

            let popped = session.pop_message().await.unwrap();
            assert_eq!(popped, Some(msgs[2].clone()));
            assert_eq!(session.len().await.unwrap(), 2);
        }

        #[tokio::test]
        async fn returns_none_on_empty_session() {
            let session = new_session("p2");
            let popped = session.pop_message().await.unwrap();
            assert_eq!(popped, None);
        }

        #[tokio::test]
        async fn successive_pops_drain_in_lifo_order() {
            let session = new_session("p3");
            let msgs = sample_messages(3);
            session.add_messages(&msgs).await.unwrap();

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
            let session = new_session("c1");
            session.add_messages(&sample_messages(5)).await.unwrap();
            session.clear().await.unwrap();

            assert!(session.is_empty().await.unwrap());
            assert_eq!(session.len().await.unwrap(), 0);
            assert_eq!(session.get_messages(None).await.unwrap(), vec![]);
        }

        #[tokio::test]
        async fn clear_on_empty_session_is_idempotent() {
            let session = new_session("c2");
            session.clear().await.unwrap();
            session.clear().await.unwrap();
            assert!(session.is_empty().await.unwrap());
        }

        #[tokio::test]
        async fn clear_removes_session_row() {
            let session = new_session("c3");
            session.add_messages(&[Message::user("hi")]).await.unwrap();
            session.clear().await.unwrap();

            // Verify the sessions table has no row for this ID.
            let guard = session.conn.lock().unwrap();
            let count: i64 = guard
                .query_row(
                    "SELECT COUNT(*) FROM sessions WHERE session_id = ?1",
                    params!["c3"],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(count, 0);
        }
    }

    mod len_and_is_empty {
        use super::*;

        #[tokio::test]
        async fn len_reflects_mutations() {
            let session = new_session("l1");
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
            let session = new_session("l2");
            assert!(session.is_empty().await.unwrap());

            session.add_messages(&[Message::user("hi")]).await.unwrap();
            assert!(!session.is_empty().await.unwrap());
        }
    }

    mod trait_objects {
        use super::*;

        #[tokio::test]
        async fn works_as_boxed_session() {
            let boxed: Box<dyn Session> = Box::new(new_session("box-1"));
            boxed.add_messages(&[Message::user("hello")]).await.unwrap();
            assert_eq!(boxed.len().await.unwrap(), 1);
            assert_eq!(boxed.id(), "box-1");
        }

        #[tokio::test]
        async fn works_as_shared_session() {
            let shared: Arc<dyn Session> = Arc::new(new_session("arc-1"));
            shared
                .add_messages(&[Message::user("hello")])
                .await
                .unwrap();
            assert_eq!(shared.len().await.unwrap(), 1);
        }
    }

    mod isolation {
        use super::*;

        #[tokio::test]
        async fn sessions_on_shared_db_are_isolated() {
            // Two sessions sharing one in-memory database.
            let conn = Connection::open_in_memory().unwrap();
            let s1 = SqliteSession::from_connection(conn, "iso-1").unwrap();
            let s2 = SqliteSession {
                id: "iso-2".into(),
                conn: Arc::clone(&s1.conn),
            };

            s1.add_messages(&[Message::user("from-s1")]).await.unwrap();
            s2.add_messages(&[Message::user("from-s2"), Message::user("from-s2-b")])
                .await
                .unwrap();

            assert_eq!(s1.len().await.unwrap(), 1);
            assert_eq!(s2.len().await.unwrap(), 2);

            let s1_msgs = s1.get_messages(None).await.unwrap();
            assert_eq!(s1_msgs[0].text().unwrap(), "from-s1");

            let s2_msgs = s2.get_messages(None).await.unwrap();
            assert_eq!(s2_msgs[0].text().unwrap(), "from-s2");
            assert_eq!(s2_msgs[1].text().unwrap(), "from-s2-b");
        }

        #[tokio::test]
        async fn clear_one_session_does_not_affect_other() {
            let conn = Connection::open_in_memory().unwrap();
            let s1 = SqliteSession::from_connection(conn, "iso-a").unwrap();
            let s2 = SqliteSession {
                id: "iso-b".into(),
                conn: Arc::clone(&s1.conn),
            };

            s1.add_messages(&[Message::user("keep-me")]).await.unwrap();
            s2.add_messages(&[Message::user("delete-me")])
                .await
                .unwrap();

            s2.clear().await.unwrap();

            assert_eq!(s1.len().await.unwrap(), 1);
            assert_eq!(s2.len().await.unwrap(), 0);
        }
    }

    mod serialization {
        use super::*;
        use crate::message::ToolCall;

        #[tokio::test]
        async fn tool_messages_survive_roundtrip() {
            let session = new_session("ser-1");
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
        async fn assistant_with_tool_calls_survives_roundtrip() {
            let session = new_session("ser-2");
            let msg = Message::assistant_tool_calls(vec![ToolCall::function(
                "tc-1",
                "get_weather",
                r#"{"city":"Tokyo"}"#,
            )]);
            session
                .add_messages(std::slice::from_ref(&msg))
                .await
                .unwrap();

            let stored = session.get_messages(None).await.unwrap();
            assert_eq!(stored.len(), 1);
            assert!(stored[0].has_tool_calls());
            let tc = stored[0].tool_calls.as_ref().unwrap();
            assert_eq!(tc[0].name(), "get_weather");
        }

        #[tokio::test]
        async fn message_with_name_field_survives_roundtrip() {
            let session = new_session("ser-3");
            let msg = Message::user("hi").with_name("alice");
            session.add_messages(&[msg]).await.unwrap();

            let stored = session.get_messages(None).await.unwrap();
            assert_eq!(stored[0].name.as_deref(), Some("alice"));
        }
    }

    mod edge_cases {
        use super::*;

        #[tokio::test]
        async fn add_then_pop_then_add_again() {
            let session = new_session("ec-1");

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
            let session = new_session("ec-2");
            session.add_messages(&sample_messages(5)).await.unwrap();
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
        async fn unicode_session_id() {
            let session = new_session("会话-αβγ-🦀");
            assert_eq!(session.id(), "会话-αβγ-🦀");

            session
                .add_messages(&[Message::user("你好")])
                .await
                .unwrap();
            let msgs = session.get_messages(None).await.unwrap();
            assert_eq!(msgs[0].text().unwrap(), "你好");
        }

        #[tokio::test]
        async fn unicode_message_content() {
            let session = new_session("ec-4");
            let msg = Message::user("こんにちは世界 🌍 مرحبا");
            session
                .add_messages(std::slice::from_ref(&msg))
                .await
                .unwrap();

            let stored = session.get_messages(None).await.unwrap();
            assert_eq!(stored[0], msg);
        }
    }
}
