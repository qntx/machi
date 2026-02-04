//! Prompt template system for AI agents.
//!
//! This module provides a Jinja2-compatible template engine for dynamic prompt generation,
//! following the smolagents architecture pattern.
//!
//! # Architecture
//!
//! The prompt system consists of:
//! - [`PromptTemplates`] - Container for all agent prompt templates
//! - [`PromptEngine`] - Jinja2-compatible template rendering engine
//! - [`TemplateContext`] - Context data for template rendering
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::prompts::{PromptTemplates, PromptEngine, TemplateContext};
//!
//! // Load built-in templates
//! let templates = PromptTemplates::toolcalling_agent();
//!
//! // Create rendering context
//! let context = TemplateContext::new()
//!     .with_tools(&tools)
//!     .with_task("What is 2 + 2?");
//!
//! // Render the system prompt
//! let engine = PromptEngine::new();
//! let system_prompt = engine.render(&templates.system_prompt, &context)?;
//! ```

mod engine;
mod renderer;
mod templates;

pub use engine::{PromptEngine, RenderError, TemplateContext};
pub use renderer::{PromptRender, SummaryAppender};
pub use templates::{
    FinalAnswerPrompt, ManagedAgentPrompt, PlanningPrompt, PromptTemplates, PromptTemplatesBuilder,
};

/// Built-in template YAML files embedded at compile time.
pub mod builtin {
    /// Tool-calling agent template (Jinja2 format).
    pub const TOOLCALLING_AGENT_YAML: &str = include_str!("toolcalling_agent.yaml");

    /// Code agent template (Jinja2 format).
    pub const CODE_AGENT_YAML: &str = include_str!("code_agent.yaml");

    /// Structured code agent template (Jinja2 format).
    pub const STRUCTURED_CODE_AGENT_YAML: &str = include_str!("structured_code_agent.yaml");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_builtin_templates() {
        let templates = PromptTemplates::toolcalling_agent();
        assert!(!templates.system_prompt.is_empty());
        assert!(!templates.planning.initial_plan.is_empty());
    }

    #[test]
    fn test_render_simple_template() {
        let engine = PromptEngine::new();
        let ctx = TemplateContext::new().with_task("Test task");

        let template = "Task: {{ task }}";
        let result = engine.render(template, &ctx).expect("render failed");
        assert_eq!(result, "Task: Test task");
    }

    #[test]
    fn test_render_with_tools() {
        use crate::tool::ToolDefinition;

        let engine = PromptEngine::new();

        let tools = vec![
            ToolDefinition {
                name: "calculator".to_string(),
                description: "Performs math calculations".to_string(),
                parameters: serde_json::json!({}),
                output_type: Some("number".to_string()),
                output_schema: None,
            },
            ToolDefinition {
                name: "search".to_string(),
                description: "Searches the web".to_string(),
                parameters: serde_json::json!({}),
                output_type: Some("string".to_string()),
                output_schema: None,
            },
        ];

        let ctx = TemplateContext::new().with_tools(&tools);

        let template = r"Tools:
{%- for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{%- endfor %}";

        let result = engine.render(template, &ctx).expect("render failed");
        assert!(result.contains("calculator: Performs math calculations"));
        assert!(result.contains("search: Searches the web"));
    }
}
