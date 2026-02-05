//! Agent builder for fluent construction.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::chat::ChatProvider;
use crate::error::{Error, Result};
use crate::tool::{BoxedConfirmationHandler, BoxedTool, Tool, ToolBox, ToolExecutionPolicy};

use super::Agent;
use super::config::AgentConfig;
use super::memory::AgentMemory;

/// Default system prompt for agents.
const DEFAULT_SYSTEM_PROMPT: &str = r"You are a helpful AI assistant that can use tools to accomplish tasks.

When you need to perform an action, use the available tools. Always think step by step about what you need to do.

When you have completed the task or have a final answer, call the `final_answer` tool with your response.";

/// Builder for creating agents with a fluent API.
#[derive(Default)]
pub struct AgentBuilder {
    provider: Option<Arc<dyn ChatProvider>>,
    tools: ToolBox,
    /// Agent configuration.
    pub config: AgentConfig,
    system_prompt: Option<String>,
    confirmation_handler: Option<BoxedConfirmationHandler>,
}

impl AgentBuilder {
    /// Create a new agent builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            tools: ToolBox::new(),
            config: AgentConfig::default(),
            system_prompt: None,
            confirmation_handler: None,
        }
    }

    /// Set the LLM provider.
    #[must_use]
    pub fn provider(mut self, provider: impl ChatProvider + 'static) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Set the LLM provider from an Arc.
    #[must_use]
    pub fn provider_arc(mut self, provider: Arc<dyn ChatProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Add a tool to the agent with default policy (Auto).
    #[must_use]
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self
    where
        T::Output: 'static,
    {
        self.tools.add(tool);
        self
    }

    /// Add a tool with a specific execution policy.
    #[must_use]
    pub fn tool_with_policy<T: Tool + 'static>(
        mut self,
        tool: T,
        policy: ToolExecutionPolicy,
    ) -> Self
    where
        T::Output: 'static,
    {
        self.tools.add_with_policy(tool, policy);
        self
    }

    /// Add a boxed tool to the agent with default policy (Auto).
    #[must_use]
    pub fn tool_boxed(mut self, tool: BoxedTool) -> Self {
        self.tools.add_boxed(tool);
        self
    }

    /// Add a boxed tool with a specific execution policy.
    #[must_use]
    pub fn tool_boxed_with_policy(mut self, tool: BoxedTool, policy: ToolExecutionPolicy) -> Self {
        self.tools.add_boxed_with_policy(tool, policy);
        self
    }

    /// Add multiple tools to the agent.
    #[must_use]
    pub fn tools(mut self, tools: Vec<BoxedTool>) -> Self {
        for tool in tools {
            self.tools.add_boxed(tool);
        }
        self
    }

    /// Set the tool execution policy for a specific tool.
    #[must_use]
    pub fn tool_policy(
        mut self,
        tool_name: impl Into<String>,
        policy: ToolExecutionPolicy,
    ) -> Self {
        self.tools.set_policy(tool_name, policy);
        self
    }

    /// Set the system prompt.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum number of steps.
    #[must_use]
    pub const fn max_steps(mut self, max_steps: usize) -> Self {
        self.config.max_steps = max_steps;
        self
    }

    /// Set the agent name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = Some(name.into());
        self
    }

    /// Set the agent description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.config.description = Some(description.into());
        self
    }

    /// Set custom instructions.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.config.instructions = Some(instructions.into());
        self
    }

    /// Set the temperature for LLM calls.
    #[must_use]
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set the max tokens for LLM calls.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set the confirmation handler.
    #[must_use]
    pub fn confirmation_handler(mut self, handler: BoxedConfirmationHandler) -> Self {
        self.confirmation_handler = Some(handler);
        self
    }

    /// Set the full configuration.
    #[must_use]
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the agent.
    ///
    /// # Errors
    ///
    /// Returns an error if no provider is set.
    pub fn build(self) -> Result<Agent> {
        let provider = self
            .provider
            .ok_or_else(|| Error::agent("Provider is required"))?;

        let system_prompt = self
            .system_prompt
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_owned());

        Ok(Agent {
            provider,
            tools: self.tools,
            config: self.config,
            memory: AgentMemory::new(),
            system_prompt,
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            step_number: 0,
            state: HashMap::new(),
            confirmation_handler: self.confirmation_handler,
            run_start: std::time::Instant::now(),
        })
    }
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("has_provider", &self.provider.is_some())
            .field("tools", &self.tools)
            .field("config", &self.config)
            .field("system_prompt", &self.system_prompt)
            .field(
                "has_confirmation_handler",
                &self.confirmation_handler.is_some(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let builder = AgentBuilder::new();
        assert!(builder.provider.is_none());
        assert!(builder.tools.is_empty());
    }

    #[test]
    fn test_builder_config() {
        let builder = AgentBuilder::new()
            .max_steps(10)
            .name("test_agent")
            .temperature(0.7);

        assert_eq!(builder.config.max_steps, 10);
        assert_eq!(builder.config.name, Some("test_agent".to_owned()));
        assert_eq!(builder.config.temperature, Some(0.7));
    }

    #[test]
    fn test_builder_no_provider_error() {
        let result = AgentBuilder::new().build();
        assert!(result.is_err());
    }
}
