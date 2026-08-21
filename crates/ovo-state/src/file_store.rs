//! JSON file persistence for conversation snapshots.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use ovo_types::{ErrorCode, OvoError};
use tokio::fs;

use crate::handle::ChatStateSnapshot;
use crate::persistence::ChatPersistence;

/// Default relative directory under a project root for session checkpoints.
pub const DEFAULT_SESSIONS_DIR: &str = ".ovo/sessions";

/// Canonical path: `{root}/.ovo/sessions/{session_id}.json`.
#[must_use]
pub fn default_session_path(root: impl AsRef<Path>, session_id: &str) -> PathBuf {
    root.as_ref()
        .join(DEFAULT_SESSIONS_DIR)
        .join(format!("{session_id}.json"))
}

/// Persist snapshots as a single JSON file (atomic write via temp + rename).
#[derive(Debug, Clone)]
pub struct FilePersistence {
    path: PathBuf,
}

impl FilePersistence {
    /// Target file path (parent dirs created on save).
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Default layout under `{root}/.ovo/sessions/{session_id}.json`.
    #[must_use]
    pub fn for_session(root: impl AsRef<Path>, session_id: impl AsRef<str>) -> Self {
        Self::new(default_session_path(root, session_id.as_ref()))
    }

    /// Path accessor.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the checkpoint file currently exists.
    #[must_use]
    pub fn exists(&self) -> bool {
        self.path.is_file()
    }
}

#[async_trait]
impl ChatPersistence for FilePersistence {
    async fn save(&self, snapshot: &ChatStateSnapshot) -> Result<(), OvoError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                OvoError::new(
                    ErrorCode::StatePersistence,
                    format!("create_dir_all {}: {e}", parent.display()),
                )
            })?;
        }
        let body = serde_json::to_vec_pretty(snapshot).map_err(|e| {
            OvoError::new(ErrorCode::StatePersistence, format!("serde snapshot: {e}"))
        })?;
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, &body).await.map_err(|e| {
            OvoError::new(
                ErrorCode::StatePersistence,
                format!("write {}: {e}", tmp.display()),
            )
        })?;
        fs::rename(&tmp, &self.path).await.map_err(|e| {
            OvoError::new(
                ErrorCode::StatePersistence,
                format!("rename {} -> {}: {e}", tmp.display(), self.path.display()),
            )
        })?;
        Ok(())
    }

    async fn load(&self) -> Result<Option<ChatStateSnapshot>, OvoError> {
        match fs::read(&self.path).await {
            Ok(bytes) => {
                let snap: ChatStateSnapshot = serde_json::from_slice(&bytes).map_err(|e| {
                    OvoError::new(
                        ErrorCode::StatePersistence,
                        format!("parse {}: {e}", self.path.display()),
                    )
                })?;
                Ok(Some(snap))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(OvoError::new(
                ErrorCode::StatePersistence,
                format!("read {}: {e}", self.path.display()),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use ovo_types::Message;
    use tempfile::tempdir;

    use super::*;
    use crate::persistence::messages_only;

    #[tokio::test]
    async fn round_trip() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("session.json");
        let store = FilePersistence::new(&path);
        let snap = messages_only(vec![Message::user("hi")]);
        store.save(&snap).await.expect("save");
        let loaded = store.load().await.expect("load").expect("some");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(
            loaded.messages.first().map(Message::text).as_deref(),
            Some("hi")
        );
    }

    #[tokio::test]
    async fn for_session_layout() {
        let dir = tempdir().expect("tmp");
        let store = FilePersistence::for_session(dir.path(), "sess_demo");
        assert!(
            store
                .path()
                .ends_with(Path::new(".ovo/sessions/sess_demo.json"))
        );
        assert!(!store.exists());
        store
            .save(&messages_only(vec![Message::user("x")]))
            .await
            .expect("save");
        assert!(store.exists());
    }
}
