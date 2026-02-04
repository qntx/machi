//! Unified prompt renderer for AI agents.
//!
//! This module provides a high-level API for rendering all agent prompts,
//! consolidating template management and rendering logic in one place.

use std::collections::HashMap;
use std::fmt::Write;

use tracing::warn;

use crate::managed::ManagedAgentInfo as ManagedInfo;
use crate::message::ChatMessage;
use crate::tool::ToolDefinition;

use super::engine::{PromptEngine, TemplateContext};
use super::templates::PromptTemplates;

/// Unified prompt renderer that encapsulates template engine and templates.
///
/// This struct provides a clean API for rendering all agent prompts,
/// hiding the complexity of template context construction.
///
/// # Example
///
/// ```rust,ignore
/// use machi::prompts::{PromptRender, PromptTemplates};
///
/// let renderer = PromptRender::new(PromptTemplates::toolcalling_agent());
///
/// let system_prompt = renderer.render_system_prompt(
///     &tool_definitions,
///     &managed_agent_infos,
///     Some("Be concise."),
/// );
/// ```
#[derive(Debug, Clone)]
pub struct PromptRender {
    engine: PromptEngine,
    templates: PromptTemplates,
}

impl PromptRender {
    /// Create a new prompt renderer with the given templates.
    #[must_use]
    pub fn new(templates: PromptTemplates) -> Self {
        Self {
            engine: PromptEngine::new(),
            templates,
        }
    }

    /// Create a renderer with the default tool-calling agent templates.
    #[must_use]
    pub fn toolcalling_agent() -> Self {
        Self::new(PromptTemplates::toolcalling_agent())
    }

    /// Create a renderer with the code agent templates.
    #[must_use]
    pub fn code_agent() -> Self {
        Self::new(PromptTemplates::code_agent())
    }

    /// Get a reference to the underlying templates.
    #[must_use]
    pub const fn templates(&self) -> &PromptTemplates {
        &self.templates
    }

    /// Get a mutable reference to the underlying templates.
    pub const fn templates_mut(&mut self) -> &mut PromptTemplates {
        &mut self.templates
    }

    /// Render the system prompt for an agent.
    ///
    /// # Arguments
    ///
    /// * `tools` - Tool definitions available to the agent
    /// * `managed_agents` - Managed agent information (for multi-agent scenarios)
    /// * `custom_instructions` - Optional custom instructions to include
    ///
    /// # Returns
    ///
    /// The rendered system prompt string. Falls back to a default prompt if rendering fails.
    #[must_use]
    pub fn render_system_prompt(
        &self,
        tools: &[ToolDefinition],
        managed_agents: &HashMap<String, ManagedInfo>,
        custom_instructions: Option<&str>,
    ) -> String {
        let ctx = TemplateContext::new()
            .with_tools(tools)
            .with_managed_agents(managed_agents)
            .with_custom_instructions_opt(custom_instructions);

        self.engine
            .render(&self.templates.system_prompt, &ctx)
            .unwrap_or_else(|e| {
                warn!(error = %e, "Failed to render system prompt template, using fallback");
                Self::default_system_prompt(tools)
            })
    }

    /// Render the task prompt for a managed agent.
    ///
    /// This is used when delegating a task to a sub-agent.
    #[must_use]
    pub fn render_managed_agent_task(&self, agent_name: &str, task: &str) -> String {
        let ctx = TemplateContext::new().with_name(agent_name).with_task(task);

        self.engine
            .render(&self.templates.managed_agent.task, &ctx)
            .unwrap_or_else(|_| Self::default_task_prompt(agent_name, task))
    }

    /// Render the report prompt for a managed agent's result.
    ///
    /// This formats the result from a sub-agent for the parent agent.
    #[must_use]
    pub fn render_managed_agent_report(&self, agent_name: &str, final_answer: &str) -> String {
        let ctx = TemplateContext::new()
            .with_name(agent_name)
            .with_final_answer(final_answer);

        self.engine
            .render(&self.templates.managed_agent.report, &ctx)
            .unwrap_or_else(|_| Self::default_report(agent_name, final_answer))
    }

    /// Render the initial planning prompt.
    #[must_use]
    pub fn render_initial_plan(
        &self,
        task: &str,
        tools: &[ToolDefinition],
        managed_agents: &HashMap<String, ManagedInfo>,
    ) -> String {
        let ctx = TemplateContext::new()
            .with_task(task)
            .with_tools(tools)
            .with_managed_agents(managed_agents);

        self.engine
            .render(&self.templates.planning.initial_plan, &ctx)
            .unwrap_or_else(|e| {
                warn!(error = %e, "Failed to render initial plan template");
                format!("Create a step-by-step plan to accomplish: {task}")
            })
    }

    /// Render the plan update prompt (pre-messages part).
    #[must_use]
    pub fn render_update_plan_pre(&self, remaining_steps: usize) -> String {
        let ctx = TemplateContext::new().with_remaining_steps(remaining_steps);

        self.engine
            .render(&self.templates.planning.update_plan_pre_messages, &ctx)
            .unwrap_or_default()
    }

    /// Render the plan update prompt (post-messages part).
    #[must_use]
    pub fn render_update_plan_post(&self, remaining_steps: usize) -> String {
        let ctx = TemplateContext::new().with_remaining_steps(remaining_steps);

        self.engine
            .render(&self.templates.planning.update_plan_post_messages, &ctx)
            .unwrap_or_default()
    }

    /// Render the final answer pre-messages prompt.
    #[must_use]
    pub fn render_final_answer_pre(&self) -> String {
        self.engine
            .render(
                &self.templates.final_answer.pre_messages,
                &TemplateContext::new(),
            )
            .unwrap_or_default()
    }

    /// Render the final answer post-messages prompt.
    #[must_use]
    pub fn render_final_answer_post(&self, task: &str) -> String {
        let ctx = TemplateContext::new().with_task(task);

        self.engine
            .render(&self.templates.final_answer.post_messages, &ctx)
            .unwrap_or_default()
    }

    /// Render a custom template with the given context.
    ///
    /// This is a low-level API for advanced use cases.
    ///
    /// # Errors
    ///
    /// Returns an error if template syntax is invalid or rendering fails.
    pub fn render_custom(
        &self,
        template: &str,
        ctx: &TemplateContext,
    ) -> Result<String, super::engine::RenderError> {
        self.engine.render(template, ctx)
    }

    /// Generate a default system prompt when template rendering fails.
    fn default_system_prompt(tools: &[ToolDefinition]) -> String {
        let mut result = String::with_capacity(512 + tools.len() * 64);

        result.push_str(
            "You are a helpful AI assistant that can use tools to accomplish tasks.\n\n\
             Available tools:\n",
        );

        for def in tools {
            let _ = writeln!(result, "- {}: {}", def.name, def.description);
        }

        result.push_str(
            "\nWhen you need to use a tool, respond with a tool call. \
             When you have the final answer, use the 'final_answer' tool to provide it.\n\n\
             Think step by step about what you need to do to accomplish the task.",
        );

        result
    }

    /// Generate a default task prompt for managed agents.
    fn default_task_prompt(name: &str, task: &str) -> String {
        format!(
            "You're a helpful agent named '{name}'.\n\
             You have been submitted this task by your manager.\n\
             ---\n\
             Task:\n{task}\n\
             ---\n\
             You're helping your manager solve a wider task: so make sure to not provide \
             a one-line answer, but give as much information as possible."
        )
    }

    /// Generate a default report format for managed agent results.
    fn default_report(name: &str, final_answer: &str) -> String {
        format!("Here is the final answer from your managed agent '{name}':\n{final_answer}")
    }
}

/// Extension trait for appending work summaries to answers.
///
/// This is used when managed agents need to provide detailed summaries
/// of their work to parent agents.
pub trait SummaryAppender {
    /// Append a summary of work to the answer string.
    fn append_summary(&self, answer: &mut String, messages: &[ChatMessage]);
}

impl SummaryAppender for PromptRender {
    fn append_summary(&self, answer: &mut String, messages: &[ChatMessage]) {
        answer.push_str(
            "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n",
        );

        for msg in messages {
            if let Some(content) = msg.text_content() {
                // Truncate long content for readability
                if content.len() > 1000 {
                    let _ = write!(answer, "\n{}...\n---", &content[..1000]);
                } else {
                    let _ = write!(answer, "\n{content}\n---");
                }
            }
        }

        answer.push_str("\n</summary_of_work>");
    }
}

impl Default for PromptRender {
    fn default() -> Self {
        Self::toolcalling_agent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_renderer() {
        let renderer = PromptRender::default();
        assert!(!renderer.templates().system_prompt.is_empty());
    }

    #[test]
    fn test_render_system_prompt_empty() {
        let renderer = PromptRender::default();
        let prompt = renderer.render_system_prompt(&[], &HashMap::new(), None);
        assert!(prompt.contains("tool"));
    }

    #[test]
    fn test_render_managed_agent_task() {
        let renderer = PromptRender::default();
        let prompt = renderer.render_managed_agent_task("researcher", "Find information about AI");
        assert!(prompt.contains("researcher") || prompt.contains("Find information"));
    }

    #[test]
    fn test_render_managed_agent_report() {
        let renderer = PromptRender::default();
        let report = renderer.render_managed_agent_report("helper", "The answer is 42");
        assert!(report.contains("42") || report.contains("helper"));
    }

    #[test]
    fn test_fallback_prompts() {
        let tools = vec![ToolDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({}),
            output_type: Some("string".to_string()),
            output_schema: None,
        }];

        let fallback = PromptRender::default_system_prompt(&tools);
        assert!(fallback.contains("test_tool"));
        assert!(fallback.contains("A test tool"));
    }
}
