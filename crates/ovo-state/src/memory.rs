//! Optional short-term memory port (summaries only — not a vector DB product).

use std::sync::Mutex;

use async_trait::async_trait;
use ovo_types::{ErrorCode, OvoError};
use serde::{Deserialize, Serialize};

/// One remembered fact / summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Stable key (host-defined).
    pub key: String,
    /// Text body.
    pub text: String,
    /// Optional free-form tags.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Host-injectable memory surface for session-level recall.
///
/// Kernel v1 only requires put/get/list; embedding search is out of scope.
#[async_trait]
pub trait MemoryPort: Send + Sync {
    /// Store or replace an item by key.
    async fn remember(&self, item: MemoryItem) -> Result<(), OvoError>;

    /// Fetch by exact key.
    async fn recall(&self, key: &str) -> Result<Option<MemoryItem>, OvoError>;

    /// List items (implementation-defined order); optional tag filter.
    async fn list(&self, tag: Option<&str>) -> Result<Vec<MemoryItem>, OvoError>;

    /// Delete by key; returns whether an entry existed.
    async fn forget(&self, key: &str) -> Result<bool, OvoError>;
}

/// No-op memory (always empty).
#[derive(Debug, Default, Clone, Copy)]
pub struct NullMemory;

#[async_trait]
impl MemoryPort for NullMemory {
    async fn remember(&self, _item: MemoryItem) -> Result<(), OvoError> {
        Ok(())
    }

    async fn recall(&self, _key: &str) -> Result<Option<MemoryItem>, OvoError> {
        Ok(None)
    }

    async fn list(&self, _tag: Option<&str>) -> Result<Vec<MemoryItem>, OvoError> {
        Ok(Vec::new())
    }

    async fn forget(&self, _key: &str) -> Result<bool, OvoError> {
        Ok(false)
    }
}

/// In-process map for tests and single-process hosts.
#[derive(Debug, Default)]
pub struct InMemoryMemory {
    items: Mutex<Vec<MemoryItem>>,
}

impl InMemoryMemory {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MemoryPort for InMemoryMemory {
    async fn remember(&self, item: MemoryItem) -> Result<(), OvoError> {
        if item.key.trim().is_empty() {
            return Err(OvoError::new(
                ErrorCode::TypesValidation,
                "memory key must be non-empty",
            ));
        }
        let mut guard = self
            .items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(slot) = guard.iter_mut().find(|i| i.key == item.key) {
            *slot = item;
        } else {
            guard.push(item);
        }
        drop(guard);
        Ok(())
    }

    async fn recall(&self, key: &str) -> Result<Option<MemoryItem>, OvoError> {
        let guard = self
            .items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let out = guard.iter().find(|i| i.key == key).cloned();
        drop(guard);
        Ok(out)
    }

    async fn list(&self, tag: Option<&str>) -> Result<Vec<MemoryItem>, OvoError> {
        let guard = self
            .items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let out = match tag {
            Some(t) => guard
                .iter()
                .filter(|i| i.tags.iter().any(|x| x == t))
                .cloned()
                .collect(),
            None => guard.clone(),
        };
        drop(guard);
        Ok(out)
    }

    async fn forget(&self, key: &str) -> Result<bool, OvoError> {
        let mut guard = self
            .items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = guard.len();
        guard.retain(|i| i.key != key);
        let removed = guard.len() < before;
        drop(guard);
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn remember_recall_forget() {
        let mem = InMemoryMemory::new();
        mem.remember(MemoryItem {
            key: "pref".into(),
            text: "likes dark mode".into(),
            tags: vec!["ui".into()],
        })
        .await
        .expect("remember");
        let hit = mem.recall("pref").await.expect("recall").expect("some");
        assert_eq!(hit.text, "likes dark mode");
        let listed = mem.list(Some("ui")).await.expect("list");
        assert_eq!(listed.len(), 1);
        assert!(mem.forget("pref").await.expect("forget"));
        assert!(mem.recall("pref").await.expect("r2").is_none());
    }
}
