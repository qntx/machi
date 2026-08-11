//! Portable agent configuration.

use machi_tools::CapabilityMode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Static or deferred instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum Instructions {
    /// Fixed system prompt body.
    Static(String),
}

impl Instructions {
    /// Resolve to a string.
    #[must_use]
    pub fn resolve(&self) -> String {
        match self {
            Self::Static(s) => s.clone(),
        }
    }
}

impl From<String> for Instructions {
    fn from(value: String) -> Self {
        Self::Static(value)
    }
}

impl From<&str> for Instructions {
    fn from(value: &str) -> Self {
        Self::Static(value.to_owned())
    }
}

/// Tool allow/deny policy on a definition (applied at agent resolution / build).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolPolicy {
    /// Inherit all tools supplied at build time.
    #[default]
    InheritAll,
    /// Only these tool names.
    Allowlist(Vec<String>),
    /// All except these names.
    Denylist(Vec<String>),
}

impl ToolPolicy {
    /// Whether a tool name is admitted by this policy.
    #[must_use]
    pub fn admits(&self, name: &str) -> bool {
        match self {
            Self::InheritAll => true,
            Self::Allowlist(allow) => allow.iter().any(|n| n == name),
            Self::Denylist(deny) => !deny.iter().any(|n| n == name),
        }
    }
}

/// Require a tool call before the turn may complete.
///
/// Enforced by [`machi_runtime::TurnRuntime`] via stop gates: when the model
/// returns a final message without having called `tool`, a reminder is injected
/// and sampling continues (up to `max_retries`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequirement {
    /// Canonical tool name that must be called.
    pub tool: String,
    /// Reminder injected when the model stops without calling it.
    pub reminder: String,
    /// Max forced re-samples.
    pub max_retries: u32,
}

/// Where a definition was loaded from (for discovery precedence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentSource {
    /// Built-in catalogue (`general-purpose`, `explore`, `plan`, …).
    Builtin,
    /// User home `~/.machi/agents`.
    User,
    /// Project `.machi/agents` (cwd → repo root walk).
    #[default]
    Project,
}

/// Versionable agent definition (data only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Unique name (slug).
    pub name: String,
    /// Human description.
    pub description: String,
    /// Instructions / system prompt body.
    pub instructions: Instructions,
    /// Default model id.
    pub model: String,
    /// Tool policy (resolved at build time — definition-level `allowed_tools`).
    #[serde(default)]
    pub tools: ToolPolicy,
    /// Optional structured output schema (JSON Schema object).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Optional completion gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<CompletionRequirement>,
    /// Default max steps for turns using this agent.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// When false, definition is invisible and not callable.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Preferred capability mode (intersected with spawn request).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability: Option<CapabilityMode>,
    /// Discovery source (not required in markdown; set by resolver).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<AgentSource>,
}

fn default_max_steps() -> usize {
    32
}

const fn default_enabled() -> bool {
    true
}

impl AgentDefinition {
    /// Minimal named definition with defaults.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            instructions: Instructions::Static(String::new()),
            model: "default".into(),
            tools: ToolPolicy::InheritAll,
            output_schema: None,
            completion: None,
            max_steps: default_max_steps(),
            enabled: true,
            capability: None,
            source: None,
        }
    }

    /// Validate required fields.
    ///
    /// # Errors
    ///
    /// Returns [`machi_types::MachiError`] when name/model empty or `max_steps` is zero.
    pub fn validate(&self) -> Result<(), machi_types::MachiError> {
        use machi_types::{ErrorCode, MachiError};
        if self.name.trim().is_empty() {
            return Err(MachiError::new(
                ErrorCode::AgentInvalidDefinition,
                "agent name must be non-empty",
            ));
        }
        if self.model.trim().is_empty() {
            return Err(MachiError::new(
                ErrorCode::AgentInvalidDefinition,
                "agent model must be non-empty",
            ));
        }
        if self.max_steps == 0 {
            return Err(MachiError::new(
                ErrorCode::AgentInvalidDefinition,
                "max_steps must be >= 1",
            ));
        }
        Ok(())
    }
}
