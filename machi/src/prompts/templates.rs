//! Prompt template structures and loading logic.
//!
//! This module defines the data structures for prompt templates that match
//! the smolagents YAML format.

use serde::{Deserialize, Serialize};

use super::builtin;

/// Planning-related prompt templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanningPrompt {
    /// Initial planning prompt for the first step.
    #[serde(default)]
    pub initial_plan: String,

    /// Pre-messages for updating the plan (shown before history).
    #[serde(default)]
    pub update_plan_pre_messages: String,

    /// Post-messages for updating the plan (shown after history).
    #[serde(default)]
    pub update_plan_post_messages: String,
}

/// Managed agent prompt templates (for multi-agent scenarios).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManagedAgentPrompt {
    /// Task prompt for delegating work to managed agents.
    #[serde(default)]
    pub task: String,

    /// Report prompt for receiving results from managed agents.
    #[serde(default)]
    pub report: String,
}

/// Final answer prompt templates (for fallback scenarios).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FinalAnswerPrompt {
    /// Pre-messages shown before agent memory dump.
    #[serde(default)]
    pub pre_messages: String,

    /// Post-messages shown after agent memory dump.
    #[serde(default)]
    pub post_messages: String,
}

/// Complete set of prompt templates for an agent.
///
/// This structure mirrors the smolagents `PromptTemplates` TypedDict,
/// providing all necessary prompts for agent operation.
///
/// # YAML Format
///
/// Templates are stored in YAML files with Jinja2 template syntax:
///
/// ```yaml
/// system_prompt: |-
///   You are an expert assistant...
///   {%- for tool in tools %}
///   - {{ tool.name }}: {{ tool.description }}
///   {%- endfor %}
///
/// planning:
///   initial_plan: |-
///     Analyze the task and create a plan...
///   update_plan_pre_messages: |-
///     ...
///   update_plan_post_messages: |-
///     ...
///
/// managed_agent:
///   task: |-
///     ...
///   report: |-
///     ...
///
/// final_answer:
///   pre_messages: |-
///     ...
///   post_messages: |-
///     ...
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptTemplates {
    /// Main system prompt template with Jinja2 syntax.
    #[serde(default)]
    pub system_prompt: String,

    /// Planning-related prompt templates.
    #[serde(default)]
    pub planning: PlanningPrompt,

    /// Managed agent prompt templates.
    #[serde(default)]
    pub managed_agent: ManagedAgentPrompt,

    /// Final answer fallback prompt templates.
    #[serde(default)]
    pub final_answer: FinalAnswerPrompt,
}

impl PromptTemplates {
    /// Create a new empty prompt templates instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for custom prompt templates.
    #[must_use]
    pub fn builder() -> PromptTemplatesBuilder {
        PromptTemplatesBuilder::new()
    }

    /// Load the built-in tool-calling agent templates.
    ///
    /// These templates are optimized for agents that use function/tool calling
    /// to interact with the environment.
    #[must_use]
    pub fn toolcalling_agent() -> Self {
        Self::from_yaml(builtin::TOOLCALLING_AGENT_YAML)
            .expect("Built-in toolcalling_agent.yaml should be valid")
    }

    /// Load the built-in code agent templates.
    ///
    /// These templates are optimized for agents that generate and execute
    /// Python code to accomplish tasks.
    #[must_use]
    pub fn code_agent() -> Self {
        Self::from_yaml(builtin::CODE_AGENT_YAML).expect("Built-in code_agent.yaml should be valid")
    }

    /// Load the built-in structured code agent templates.
    ///
    /// These templates use structured JSON output format for code generation.
    #[must_use]
    pub fn structured_code_agent() -> Self {
        Self::from_yaml(builtin::STRUCTURED_CODE_AGENT_YAML)
            .expect("Built-in structured_code_agent.yaml should be valid")
    }

    /// Load templates from a YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is malformed or doesn't match
    /// the expected schema.
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// Load templates from a YAML file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, TemplateLoadError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(TemplateLoadError::Io)?;
        Self::from_yaml(&content).map_err(TemplateLoadError::Yaml)
    }

    /// Check if all required templates are present and non-empty.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        !self.system_prompt.is_empty() && !self.planning.initial_plan.is_empty()
    }

    /// Merge another template set into this one, filling in empty fields.
    pub fn merge_defaults(&mut self, defaults: &Self) {
        if self.system_prompt.is_empty() {
            self.system_prompt.clone_from(&defaults.system_prompt);
        }
        if self.planning.initial_plan.is_empty() {
            self.planning
                .initial_plan
                .clone_from(&defaults.planning.initial_plan);
        }
        if self.planning.update_plan_pre_messages.is_empty() {
            self.planning
                .update_plan_pre_messages
                .clone_from(&defaults.planning.update_plan_pre_messages);
        }
        if self.planning.update_plan_post_messages.is_empty() {
            self.planning
                .update_plan_post_messages
                .clone_from(&defaults.planning.update_plan_post_messages);
        }
        if self.managed_agent.task.is_empty() {
            self.managed_agent
                .task
                .clone_from(&defaults.managed_agent.task);
        }
        if self.managed_agent.report.is_empty() {
            self.managed_agent
                .report
                .clone_from(&defaults.managed_agent.report);
        }
        if self.final_answer.pre_messages.is_empty() {
            self.final_answer
                .pre_messages
                .clone_from(&defaults.final_answer.pre_messages);
        }
        if self.final_answer.post_messages.is_empty() {
            self.final_answer
                .post_messages
                .clone_from(&defaults.final_answer.post_messages);
        }
    }
}

/// Builder for constructing custom [`PromptTemplates`].
#[derive(Debug, Clone, Default)]
pub struct PromptTemplatesBuilder {
    templates: PromptTemplates,
}

impl PromptTemplatesBuilder {
    /// Create a new builder with empty templates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start from the built-in tool-calling agent templates.
    #[must_use]
    pub fn from_toolcalling_agent() -> Self {
        Self {
            templates: PromptTemplates::toolcalling_agent(),
        }
    }

    /// Start from the built-in code agent templates.
    #[must_use]
    pub fn from_code_agent() -> Self {
        Self {
            templates: PromptTemplates::code_agent(),
        }
    }

    /// Set the system prompt template.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.templates.system_prompt = prompt.into();
        self
    }

    /// Set the initial planning prompt.
    #[must_use]
    pub fn initial_plan(mut self, prompt: impl Into<String>) -> Self {
        self.templates.planning.initial_plan = prompt.into();
        self
    }

    /// Set the update plan pre-messages prompt.
    #[must_use]
    pub fn update_plan_pre_messages(mut self, prompt: impl Into<String>) -> Self {
        self.templates.planning.update_plan_pre_messages = prompt.into();
        self
    }

    /// Set the update plan post-messages prompt.
    #[must_use]
    pub fn update_plan_post_messages(mut self, prompt: impl Into<String>) -> Self {
        self.templates.planning.update_plan_post_messages = prompt.into();
        self
    }

    /// Set the managed agent task prompt.
    #[must_use]
    pub fn managed_agent_task(mut self, prompt: impl Into<String>) -> Self {
        self.templates.managed_agent.task = prompt.into();
        self
    }

    /// Set the managed agent report prompt.
    #[must_use]
    pub fn managed_agent_report(mut self, prompt: impl Into<String>) -> Self {
        self.templates.managed_agent.report = prompt.into();
        self
    }

    /// Set the final answer pre-messages prompt.
    #[must_use]
    pub fn final_answer_pre_messages(mut self, prompt: impl Into<String>) -> Self {
        self.templates.final_answer.pre_messages = prompt.into();
        self
    }

    /// Set the final answer post-messages prompt.
    #[must_use]
    pub fn final_answer_post_messages(mut self, prompt: impl Into<String>) -> Self {
        self.templates.final_answer.post_messages = prompt.into();
        self
    }

    /// Build the final [`PromptTemplates`].
    #[must_use]
    pub fn build(self) -> PromptTemplates {
        self.templates
    }
}

/// Error type for template loading operations.
#[derive(Debug)]
pub enum TemplateLoadError {
    /// IO error reading the template file.
    Io(std::io::Error),
    /// YAML parsing error.
    Yaml(serde_yaml::Error),
}

impl std::fmt::Display for TemplateLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Failed to read template file: {e}"),
            Self::Yaml(e) => write!(f, "Failed to parse template YAML: {e}"),
        }
    }
}

impl std::error::Error for TemplateLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Yaml(e) => Some(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_toolcalling_agent() {
        let templates = PromptTemplates::toolcalling_agent();
        assert!(templates.system_prompt.contains("expert assistant"));
        assert!(templates.system_prompt.contains("tool calls"));
        assert!(!templates.planning.initial_plan.is_empty());
    }

    #[test]
    fn test_builder() {
        let templates = PromptTemplates::builder()
            .system_prompt("Custom system prompt")
            .initial_plan("Custom planning prompt")
            .build();

        assert_eq!(templates.system_prompt, "Custom system prompt");
        assert_eq!(templates.planning.initial_plan, "Custom planning prompt");
    }

    #[test]
    fn test_merge_defaults() {
        let mut custom = PromptTemplates::builder()
            .system_prompt("My custom system prompt")
            .build();

        let defaults = PromptTemplates::toolcalling_agent();
        custom.merge_defaults(&defaults);

        // Custom value should be preserved
        assert_eq!(custom.system_prompt, "My custom system prompt");
        // Empty values should be filled from defaults
        assert!(!custom.planning.initial_plan.is_empty());
    }

    #[test]
    fn test_is_complete() {
        let empty = PromptTemplates::new();
        assert!(!empty.is_complete());

        let complete = PromptTemplates::toolcalling_agent();
        assert!(complete.is_complete());
    }
}
