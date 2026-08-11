//! Tool registry with capability filtering.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{ToolError, codes};
use crate::metadata::CapabilityFlag;
use crate::tool::{DynTool, SharedTool, ToolDefinition};

/// How nested/session capability mode filters tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CapabilityMode {
    /// All registered tools.
    #[default]
    Full,
    /// Only tools admissible under read-only metadata rules.
    ReadOnly,
    /// Only tools that do not include execute/spawn (plan-friendly).
    Plan,
}

impl CapabilityMode {
    /// Parse common string forms (`full`, `read_only`, `plan`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "full" => Some(Self::Full),
            "read_only" | "read-only" | "readonly" => Some(Self::ReadOnly),
            "plan" => Some(Self::Plan),
            _ => None,
        }
    }

    /// More restrictive of two modes (for definition ∩ request).
    #[must_use]
    pub const fn intersect(self, other: Self) -> Self {
        use CapabilityMode::{Full, Plan, ReadOnly};
        match (self, other) {
            (ReadOnly, _) | (_, ReadOnly) => ReadOnly,
            (Plan, _) | (_, Plan) => Plan,
            (Full, Full) => Full,
        }
    }
}

/// Thread-safe tool registry.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: Arc<HashMap<String, SharedTool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<_> = self.tools.keys().map(String::as_str).collect();
        names.sort_unstable();
        f.debug_struct("ToolRegistry")
            .field("tools", &names)
            .finish()
    }
}

impl ToolRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a list of tools (last wins on name collision).
    #[must_use]
    pub fn from_tools(tools: Vec<SharedTool>) -> Self {
        let mut map = HashMap::new();
        for tool in tools {
            map.insert(tool.name().to_owned(), tool);
        }
        Self {
            tools: Arc::new(map),
        }
    }

    /// Lookup by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<SharedTool> {
        self.tools.get(name).cloned()
    }

    /// Require tool or error.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when the tool is missing.
    pub fn require(&self, name: &str) -> Result<SharedTool, ToolError> {
        self.get(name).ok_or_else(|| codes::not_found(name))
    }

    /// Definitions visible under a capability mode.
    #[must_use]
    pub fn definitions(&self, mode: CapabilityMode) -> Vec<ToolDefinition> {
        let mut defs: Vec<_> = self
            .tools
            .values()
            .filter(|t| self.allows(t.as_ref(), mode))
            .map(|t| t.definition())
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Whether a tool is allowed under mode.
    #[must_use]
    pub fn allows(&self, tool: &dyn DynTool, mode: CapabilityMode) -> bool {
        let _ = self;
        let meta = tool.metadata();
        match mode {
            CapabilityMode::Full => true,
            CapabilityMode::ReadOnly => meta.allowed_in_read_only(),
            CapabilityMode::Plan => !meta
                .capabilities
                .iter()
                .any(|c| matches!(c, CapabilityFlag::Execute | CapabilityFlag::Spawn)),
        }
    }

    /// Number of tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Merge another registry; `other` wins on name collision.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        let mut map = (*self.tools).clone();
        for (k, v) in other.tools.iter() {
            map.insert(k.clone(), Arc::clone(v));
        }
        Self {
            tools: Arc::new(map),
        }
    }

    /// Tool names (sorted).
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<_> = self.tools.keys().cloned().collect();
        n.sort_unstable();
        n
    }
}
