//! Prompt rendering methods for the Agent.
//!
//! This module provides thin delegation to [`PromptRender`] for backward compatibility.
//! All actual rendering logic is centralized in the `prompts` module.

use crate::prompts::SummaryAppender;

use super::Agent;

impl Agent {
    /// Render the system prompt using the unified renderer.
    ///
    /// Delegates to [`PromptRender::render_system_prompt`].
    #[inline]
    pub(crate) fn render_system_prompt(&self) -> String {
        self.prompt_renderer.render_system_prompt(
            &self.tools.definitions(),
            &self.managed_agents.infos(),
            self.custom_instructions.as_deref(),
        )
    }

    /// Format a task prompt for managed agent execution.
    ///
    /// Delegates to [`PromptRender::render_managed_agent_task`].
    #[inline]
    pub(crate) fn format_task_prompt(&self, name: &str, task: &str) -> String {
        self.prompt_renderer.render_managed_agent_task(name, task)
    }

    /// Format a report from a managed agent.
    ///
    /// Delegates to [`PromptRender::render_managed_agent_report`].
    #[inline]
    pub(crate) fn format_report(&self, name: &str, final_answer: &str) -> String {
        self.prompt_renderer
            .render_managed_agent_report(name, final_answer)
    }

    /// Append a summary of the agent's work to an answer.
    ///
    /// Uses the [`SummaryAppender`] trait implementation on `PromptRender`.
    #[inline]
    pub(crate) fn append_summary(&self, answer: &mut String) {
        let messages = self.memory.to_messages(true);
        self.prompt_renderer.append_summary(answer, &messages);
    }
}
