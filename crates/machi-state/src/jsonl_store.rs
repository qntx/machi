//! JSONL event log + periodic snapshot persistence (W4.4).
//!
//! Maturity: **core**
//!
//! Layout under a session directory:
//! - `events.jsonl` — append-only message events (with torn-write repair on load)
//! - `snapshot.json` — periodic full checkpoint

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use machi_types::{ErrorCode, MachiError, Message};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::handle::ChatStateSnapshot;
use crate::ledger::UsageLedger;
use crate::persistence::{ChatPersistence, messages_only};

/// Version header for the events log.
pub const EVENTS_HEADER: &str = "# machi-session-events/1";

/// How often to rewrite the full snapshot (every N message appends).
pub const DEFAULT_SNAPSHOT_EVERY: u32 = 16;

/// One durable event line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Event {
    /// Appended message.
    Message {
        /// Message body.
        message: Message,
    },
}

/// JSONL event log with optional periodic snapshot.
#[derive(Debug, Clone)]
pub struct JsonlPersistence {
    dir: PathBuf,
    snapshot_every: u32,
    appends_since_snapshot: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl JsonlPersistence {
    /// Session directory (created on first write).
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            snapshot_every: DEFAULT_SNAPSHOT_EVERY,
            appends_since_snapshot: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// Set snapshot frequency (minimum 1).
    #[must_use]
    pub fn with_snapshot_every(mut self, n: u32) -> Self {
        self.snapshot_every = if n == 0 { 1 } else { n };
        self
    }

    fn events_path(&self) -> PathBuf {
        self.dir.join("events.jsonl")
    }

    fn snapshot_path(&self) -> PathBuf {
        self.dir.join("snapshot.json")
    }

    async fn ensure_dir(&self) -> Result<(), MachiError> {
        fs::create_dir_all(&self.dir).await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("create_dir_all {}: {e}", self.dir.display()),
            )
        })
    }

    async fn append_event_line(&self, event: &Event) -> Result<(), MachiError> {
        self.ensure_dir().await?;
        let path = self.events_path();
        let mut line = if path.exists() {
            String::new()
        } else {
            format!("{EVENTS_HEADER}\n")
        };
        let body = serde_json::to_string(event).map_err(|e| {
            MachiError::new(ErrorCode::StatePersistence, format!("serde event: {e}"))
        })?;
        line.push_str(&body);
        line.push('\n');
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|e| {
                MachiError::new(
                    ErrorCode::StatePersistence,
                    format!("open {}: {e}", path.display()),
                )
            })?;
        file.write_all(line.as_bytes()).await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("write {}: {e}", path.display()),
            )
        })?;
        file.sync_data().await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("sync {}: {e}", path.display()),
            )
        })?;
        Ok(())
    }

    /// Load events with torn-write repair (drop incomplete last line).
    async fn load_events(&self) -> Result<Vec<Message>, MachiError> {
        let path = self.events_path();
        let bytes = match fs::read(&path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(MachiError::new(
                    ErrorCode::StatePersistence,
                    format!("read {}: {e}", path.display()),
                ));
            }
        };
        let mut messages = Vec::new();
        let mut offset = 0usize;
        let mut line_no = 0usize;
        while offset < bytes.len() {
            line_no = line_no.saturating_add(1);
            let tail = bytes.get(offset..).unwrap_or(&[]);
            let Some(rel) = tail.iter().position(|b| *b == b'\n') else {
                // Torn tail: rewrite known-good prefix (drop incomplete last line).
                let prefix = bytes.get(..offset).unwrap_or(&[]);
                self.truncate_events_to(prefix).await?;
                break;
            };
            let end = offset.saturating_add(rel);
            let line = bytes.get(offset..end).unwrap_or(&[]);
            offset = end.saturating_add(1);
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            if line_no == 1 && line.starts_with(b"#") {
                continue;
            }
            let event: Event = match serde_json::from_slice(line) {
                Ok(e) => e,
                Err(e) => {
                    return Err(MachiError::new(
                        ErrorCode::StatePersistence,
                        format!("parse events line {line_no}: {e}"),
                    ));
                }
            };
            match event {
                Event::Message { message } => messages.push(message),
            }
        }
        Ok(messages)
    }

    /// Truncate events file to a known-good byte prefix (torn-write repair).
    async fn truncate_events_to(&self, prefix: &[u8]) -> Result<(), MachiError> {
        let path = self.events_path();
        fs::write(&path, prefix).await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("truncate events {}: {e}", path.display()),
            )
        })
    }

    async fn write_snapshot_file(&self, snap: &ChatStateSnapshot) -> Result<(), MachiError> {
        self.ensure_dir().await?;
        let path = self.snapshot_path();
        let body = serde_json::to_vec_pretty(snap).map_err(|e| {
            MachiError::new(ErrorCode::StatePersistence, format!("serde snapshot: {e}"))
        })?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &body).await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("write {}: {e}", tmp.display()),
            )
        })?;
        fs::rename(&tmp, &path).await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("rename {}: {e}", path.display()),
            )
        })?;
        Ok(())
    }
}

#[async_trait]
impl ChatPersistence for JsonlPersistence {
    async fn save(&self, snapshot: &ChatStateSnapshot) -> Result<(), MachiError> {
        // Rewrite events from full snapshot (compaction-safe).
        self.ensure_dir().await?;
        let path = self.events_path();
        let mut body = format!("{EVENTS_HEADER}\n");
        for message in &snapshot.messages {
            let ev = Event::Message {
                message: message.clone(),
            };
            let line = serde_json::to_string(&ev)
                .map_err(|e| MachiError::new(ErrorCode::StatePersistence, format!("serde: {e}")))?;
            body.push_str(&line);
            body.push('\n');
        }
        let tmp = path.with_extension("jsonl.tmp");
        fs::write(&tmp, body.as_bytes()).await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("write {}: {e}", tmp.display()),
            )
        })?;
        fs::rename(&tmp, &path).await.map_err(|e| {
            MachiError::new(
                ErrorCode::StatePersistence,
                format!("rename {}: {e}", path.display()),
            )
        })?;
        self.write_snapshot_file(snapshot).await?;
        self.appends_since_snapshot
            .store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn load(&self) -> Result<Option<ChatStateSnapshot>, MachiError> {
        // Prefer replaying events (source of truth); fall back to snapshot.
        let messages = self.load_events().await?;
        if !messages.is_empty() {
            // Events are SoT for messages; snapshot is SoT for usage ledger.
            // Missing snapshot → empty ledger. Corrupt / unreadable snapshot → hard error
            // (do not silently zero usage when a ledger file exists).
            let usage = match fs::read(self.snapshot_path()).await {
                Ok(bytes) => {
                    let snap: ChatStateSnapshot = serde_json::from_slice(&bytes).map_err(|e| {
                        MachiError::new(
                            ErrorCode::StatePersistence,
                            format!("parse snapshot usage: {e}"),
                        )
                    })?;
                    snap.usage
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => UsageLedger::default(),
                Err(e) => {
                    return Err(MachiError::new(
                        ErrorCode::StatePersistence,
                        format!("read snapshot: {e}"),
                    ));
                }
            };
            let mut snap = messages_only(messages);
            snap.usage = usage;
            return Ok(Some(snap));
        }
        match fs::read(self.snapshot_path()).await {
            Ok(bytes) => {
                let snap = serde_json::from_slice(&bytes).map_err(|e| {
                    MachiError::new(ErrorCode::StatePersistence, format!("parse snapshot: {e}"))
                })?;
                Ok(Some(snap))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(MachiError::new(
                ErrorCode::StatePersistence,
                format!("read snapshot: {e}"),
            )),
        }
    }

    async fn persist_message(&self, message: &Message) -> Result<(), MachiError> {
        self.append_event_line(&Event::Message {
            message: message.clone(),
        })
        .await?;
        let n = self
            .appends_since_snapshot
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .saturating_add(1);
        if n >= self.snapshot_every {
            if let Some(snap) = self.load().await? {
                self.write_snapshot_file(&snap).await?;
            }
            self.appends_since_snapshot
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Convenience: `{root}/.machi/sessions/{id}/` directory store.
#[must_use]
pub fn session_jsonl_dir(root: impl AsRef<Path>, session_id: &str) -> PathBuf {
    root.as_ref().join(".machi/sessions").join(session_id)
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "unit tests")]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn append_and_reload() {
        let dir = tempdir().expect("tmp");
        let store = JsonlPersistence::new(dir.path().join("sess")).with_snapshot_every(2);
        store.persist_message(&Message::user("a")).await.expect("a");
        store
            .persist_message(&Message::assistant("b"))
            .await
            .expect("b");
        let loaded = store.load().await.expect("load").expect("some");
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(
            loaded.messages.first().map(Message::text).as_deref(),
            Some("a")
        );
    }

    #[tokio::test]
    async fn torn_tail_repaired() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("sess");
        let store = JsonlPersistence::new(&path);
        store
            .persist_message(&Message::user("ok"))
            .await
            .expect("ok");
        let events = path.join("events.jsonl");
        let mut raw = fs::read_to_string(&events).await.expect("r");
        raw.push_str("{\"kind\":\"message\",\"message\":"); // torn
        fs::write(&events, raw.as_bytes()).await.expect("w");
        let loaded = store.load().await.expect("load").expect("some");
        assert_eq!(loaded.messages.len(), 1);
    }
}
