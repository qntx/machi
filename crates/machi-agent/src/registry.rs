//! In-memory registry of [`AgentDefinition`] for host / workflow resolution.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use machi_types::{ErrorCode, MachiError};

use crate::builtin::builtin_definitions;
use crate::definition::AgentDefinition;
use crate::discovery::{discover_in_dir, discover_project, resolve_agents};

/// Shared agent definition catalogue for `agent_type` resolution.
///
/// Clone is cheap (`Arc` interior). Lookups are O(log n) on a `BTreeMap`.
/// Disabled definitions are never stored (visible == callable).
#[derive(Debug, Clone, Default)]
pub struct AgentRegistry {
    inner: Arc<BTreeMap<String, AgentDefinition>>,
}

impl AgentRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry seeded with builtin agent types only.
    #[must_use]
    pub fn with_builtins() -> Self {
        Self::from_definitions(builtin_definitions())
    }

    /// Multi-level discovery from `cwd` (builtin → user → project shadowing).
    ///
    /// # Errors
    ///
    /// Discovery I/O failures.
    pub fn resolve_from_cwd(cwd: impl AsRef<Path>) -> Result<Self, MachiError> {
        Ok(Self::from_definitions(resolve_agents(cwd)?))
    }

    /// Build from an iterator of definitions (later names overwrite earlier;
    /// disabled entries are dropped).
    #[must_use]
    pub fn from_definitions(defs: impl IntoIterator<Item = AgentDefinition>) -> Self {
        let mut map = BTreeMap::new();
        for def in defs {
            if !def.enabled {
                continue;
            }
            map.insert(def.name.clone(), def);
        }
        Self {
            inner: Arc::new(map),
        }
    }

    /// Insert or replace a definition; returns a new registry (copy-on-write).
    /// Disabled definitions remove the name from the catalogue.
    #[must_use]
    pub fn insert(&self, def: AgentDefinition) -> Self {
        let mut map = (*self.inner).clone();
        if def.enabled {
            map.insert(def.name.clone(), def);
        } else {
            map.remove(&def.name);
        }
        Self {
            inner: Arc::new(map),
        }
    }

    /// Merge another registry (other wins on key conflict).
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        let mut map = (*self.inner).clone();
        for (k, v) in other.inner.iter() {
            map.insert(k.clone(), v.clone());
        }
        Self {
            inner: Arc::new(map),
        }
    }

    /// Number of registered agents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Lookup by name (callable agents only).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&AgentDefinition> {
        self.inner.get(name).filter(|d| d.enabled)
    }

    /// Lookup by name or error.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::AgentNotFound`] when missing.
    pub fn require(&self, name: &str) -> Result<&AgentDefinition, MachiError> {
        self.get(name).ok_or_else(|| {
            MachiError::new(
                ErrorCode::AgentNotFound,
                format!("agent_type '{name}' not registered"),
            )
        })
    }

    /// Sorted names.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.inner.keys().cloned().collect()
    }

    /// Discover definitions under a directory and merge (discovered wins on conflict).
    ///
    /// # Errors
    ///
    /// Directory read / parse failures when `strict`.
    pub fn discover_dir(self, root: impl AsRef<Path>, strict: bool) -> Result<Self, MachiError> {
        let found = discover_in_dir(root, strict)?;
        Ok(self.merge(&Self::from_definitions(found)))
    }

    /// Discover `{cwd}/.machi/agents` when present.
    ///
    /// # Errors
    ///
    /// Propagates discovery failures.
    pub fn discover_project(self, cwd: impl AsRef<Path>) -> Result<Self, MachiError> {
        let found = discover_project(cwd)?;
        Ok(self.merge(&Self::from_definitions(found)))
    }

    /// Replace with multi-level resolve for `cwd` (includes builtins).
    ///
    /// # Errors
    ///
    /// Discovery I/O failures.
    pub fn resolve_layers(self, cwd: impl AsRef<Path>) -> Result<Self, MachiError> {
        // `self` discarded: resolve_agents already includes builtins + full shadowing.
        let _ = self;
        Self::resolve_from_cwd(cwd)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::definition::Instructions;

    fn def(name: &str) -> AgentDefinition {
        let mut d = AgentDefinition::new(name);
        d.instructions = Instructions::Static("hi".into());
        d.model = "mock".into();
        d.max_steps = 4;
        d
    }

    #[test]
    fn insert_and_require() {
        let reg = AgentRegistry::new().insert(def("worker"));
        assert_eq!(reg.require("worker").expect("get").name, "worker");
        assert_eq!(
            reg.require("missing").expect_err("nf").code(),
            ErrorCode::AgentNotFound
        );
    }

    #[test]
    fn discover_merges() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("helper.md");
        let mut f = fs::File::create(&path).expect("create");
        write!(f, "---\nname: helper\nmodel: m\n---\n\nHelp.\n").expect("write");
        let reg = AgentRegistry::new()
            .insert(def("base"))
            .discover_dir(dir.path(), true)
            .expect("disc");
        assert!(reg.get("base").is_some());
        assert!(reg.get("helper").is_some());
        assert_eq!(reg.len(), 2);
    }
}
