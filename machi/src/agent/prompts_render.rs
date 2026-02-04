//! Prompt rendering methods for the Agent.
//!
//! This module contains methods for rendering system prompts, task prompts,
//! and other template-based text generation.

use std::fmt::Write;

use tracing::warn;

use crate::prompts::TemplateContext;

use super::Agent;

impl Agent {
    /// Render the system prompt using templates.
    pub(crate) fn render_system_prompt(&self) -> String {
        let defs = self.tools.definitions();
        let managed_agent_infos = self.managed_agents.infos();
        let ctx = TemplateContext::new()
            .with_tools(&defs)
            .with_managed_agents(&managed_agent_infos)
            .with_custom_instructions_opt(self.custom_instructions.as_deref());

        self.prompt_engine
            .render(&self.prompt_templates.system_prompt, &ctx)
            .unwrap_or_else(|e| {
                warn!(error = %e, "Failed to render system prompt template, using fallback");
                self.default_system_prompt()
            })
    }

    /// Generate a default system prompt when template rendering fails.
    fn default_system_prompt(&self) -> String {
        let defs = self.tools.definitions();
        let mut result = String::with_capacity(512 + defs.len() * 64);

        result.push_str(
            "You are a helpful AI assistant that can use tools to accomplish tasks.\n\n\
             Available tools:\n",
        );

        for def in &defs {
            let _ = writeln!(result, "- {}: {}", def.name, def.description);
        }

        result.push_str(
            "\nWhen you need to use a tool, respond with a tool call. \
             When you have the final answer, use the 'final_answer' tool to provide it.\n\n\
             Think step by step about what you need to do to accomplish the task.",
        );

        result
    }

    /// Format a task prompt for managed agent execution.
    pub(crate) fn format_task_prompt(&self, name: &str, task: &str) -> String {
        let ctx = TemplateContext::new().with_name(name).with_task(task);

        self.prompt_engine
            .render(&self.prompt_templates.managed_agent.task, &ctx)
            .unwrap_or_else(|_| {
                format!(
                    "You're a helpful agent named '{name}'.\n\
                     You have been submitted this task by your manager.\n\
                     ---\n\
                     Task:\n{task}\n\
                     ---\n\
                     You're helping your manager solve a wider task: so make sure to not provide \
                     a one-line answer, but give as much information as possible."
                )
            })
    }

    /// Format a report from a managed agent.
    pub(crate) fn format_report(&self, name: &str, final_answer: &str) -> String {
        let ctx = TemplateContext::new()
            .with_name(name)
            .with_final_answer(final_answer);

        self.prompt_engine
            .render(&self.prompt_templates.managed_agent.report, &ctx)
            .unwrap_or_else(|_| {
                format!(
                    "Here is the final answer from your managed agent '{name}':\n{final_answer}"
                )
            })
    }

    /// Append a summary of the agent's work to an answer.
    pub(crate) fn append_summary(&self, answer: &mut String) {
        answer.push_str(
            "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n",
        );
        for msg in self.memory.to_messages(true) {
            if let Some(content) = msg.text_content() {
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
