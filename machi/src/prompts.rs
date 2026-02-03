//! Prompt templates for agent system prompts.
//!
//! This module provides configurable prompt templates that define how
//! agents interact with language models.
//!
//! Prompts are loaded from YAML files embedded at compile time using `include_str!`.
//! This ensures zero runtime overhead and guarantees prompt availability.

use serde::{Deserialize, Serialize};

/// Raw YAML content embedded at compile time.
mod embedded {
    /// Tool-calling agent prompts YAML.
    pub const TOOLCALLING_AGENT_YAML: &str = include_str!("prompts/toolcalling_agent.yaml");

    /// Code agent prompts YAML.
    pub const CODE_AGENT_YAML: &str = include_str!("prompts/code_agent.yaml");

    /// Structured code agent prompts YAML.
    pub const STRUCTURED_CODE_AGENT_YAML: &str = include_str!("prompts/structured_code_agent.yaml");
}

/// Complete prompt templates for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplates {
    /// Main system prompt.
    pub system_prompt: String,
    /// Planning prompt templates.
    #[serde(default)]
    pub planning: PlanningPrompts,
    /// Managed agent prompt templates.
    #[serde(default)]
    pub managed_agent: ManagedAgentPrompts,
    /// Final answer prompt templates.
    #[serde(default)]
    pub final_answer: FinalAnswerPrompts,
}

/// Planning-related prompt templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanningPrompts {
    /// Initial planning prompt.
    #[serde(default)]
    pub initial_plan: String,
    /// Pre-messages for plan updates.
    #[serde(default)]
    pub update_plan_pre_messages: String,
    /// Post-messages for plan updates.
    #[serde(default)]
    pub update_plan_post_messages: String,
}

/// Managed agent prompt templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManagedAgentPrompts {
    /// Task prompt for managed agents.
    #[serde(default)]
    pub task: String,
    /// Report prompt for managed agent results.
    #[serde(default)]
    pub report: String,
}

/// Final answer prompt templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FinalAnswerPrompts {
    /// Pre-messages before final answer.
    #[serde(default)]
    pub pre_messages: String,
    /// Post-messages after final answer.
    #[serde(default)]
    pub post_messages: String,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        Self::tool_calling_agent()
    }
}

impl PromptTemplates {
    /// Load prompts for a tool-calling agent from embedded YAML.
    ///
    /// # Panics
    ///
    /// Panics if the embedded YAML is malformed (should never happen in production).
    #[must_use]
    pub fn tool_calling_agent() -> Self {
        serde_yaml::from_str(embedded::TOOLCALLING_AGENT_YAML)
            .expect("Failed to parse embedded toolcalling_agent.yaml")
    }

    /// Load prompts for a code agent from embedded YAML.
    ///
    /// # Panics
    ///
    /// Panics if the embedded YAML is malformed (should never happen in production).
    #[must_use]
    pub fn code_agent() -> Self {
        serde_yaml::from_str(embedded::CODE_AGENT_YAML)
            .expect("Failed to parse embedded code_agent.yaml")
    }

    /// Load prompts for a structured code agent from embedded YAML.
    ///
    /// # Panics
    ///
    /// Panics if the embedded YAML is malformed (should never happen in production).
    #[must_use]
    pub fn structured_code_agent() -> Self {
        serde_yaml::from_str(embedded::STRUCTURED_CODE_AGENT_YAML)
            .expect("Failed to parse embedded structured_code_agent.yaml")
    }

    /// Load prompts from a YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is malformed.
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// Load prompts from a YAML file at runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the YAML is malformed.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, PromptLoadError> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&content)?)
    }

    /// Serialize prompts to YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }
}

/// Error type for prompt loading operations.
#[derive(Debug)]
pub enum PromptLoadError {
    /// IO error when reading file.
    Io(std::io::Error),
    /// YAML parsing error.
    Yaml(serde_yaml::Error),
}

impl std::fmt::Display for PromptLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Failed to read prompt file: {e}"),
            Self::Yaml(e) => write!(f, "Failed to parse prompt YAML: {e}"),
        }
    }
}

impl std::error::Error for PromptLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Yaml(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for PromptLoadError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_yaml::Error> for PromptLoadError {
    fn from(err: serde_yaml::Error) -> Self {
        Self::Yaml(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_prompts() {
        let prompts = PromptTemplates::default();
        assert!(!prompts.system_prompt.is_empty());
        assert!(!prompts.planning.initial_plan.is_empty());
    }
}
