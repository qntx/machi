//! Agent builder for constructing agents with a fluent API.

use crate::{
    callback::CallbackRegistry,
    error::{AgentError, Result},
    managed::{BoxedManagedAgent, ManagedAgentRegistry},
    memory::AgentMemory,
    prompts::{PromptRender, PromptTemplates},
    providers::common::Model,
    tool::{BoxedTool, ToolBox},
    tools::FinalAnswerTool,
};

use std::{collections::HashMap, sync::Arc};

use super::{Agent, AgentConfig, FinalAnswerChecks};

/// Builder for [`Agent`].
///
/// # Example
///
/// ```rust,ignore
/// let agent = Agent::builder()
///     .model(my_model)
///     .tool(Box::new(MyTool))
///     .max_steps(10)
///     .build();
/// ```
#[derive(Default)]
pub struct AgentBuilder {
    model: Option<Box<dyn Model>>,
    tools: Vec<BoxedTool>,
    managed_agents: Vec<BoxedManagedAgent>,
    config: AgentConfig,
    prompt_templates: Option<PromptTemplates>,
    custom_instructions: Option<String>,
    final_answer_checks: FinalAnswerChecks,
    callbacks: CallbackRegistry,
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("has_model", &self.model.is_some())
            .field("tools", &self.tools.len())
            .field("managed_agents", &self.managed_agents.len())
            .finish_non_exhaustive()
    }
}

impl AgentBuilder {
    /// Create a new builder with default settings.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            managed_agents: Vec::new(),
            config: AgentConfig::new(),
            prompt_templates: None,
            custom_instructions: None,
            final_answer_checks: FinalAnswerChecks::new(),
            callbacks: CallbackRegistry::new(),
        }
    }

    /// Set the language model.
    #[must_use]
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Box::new(model));
        self
    }

    /// Add a tool to the agent.
    #[must_use]
    pub fn tool(mut self, tool: BoxedTool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools to the agent.
    #[must_use]
    pub fn tools(mut self, tools: impl IntoIterator<Item = BoxedTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add base tools commonly used by agents.
    ///
    /// Similar to smolagents' `add_base_tools=True` option, this adds:
    /// - `FinalAnswerTool` - for providing final answers
    /// - `VisitWebpageTool` - for reading webpage content
    ///
    /// Note: `FinalAnswerTool` is always added automatically, so this mainly
    /// adds `VisitWebpageTool` for web browsing capability.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder()
    ///     .model(model)
    ///     .add_base_tools()
    ///     .build();
    /// ```
    #[must_use]
    pub fn add_base_tools(mut self) -> Self {
        self.tools.extend(crate::tools::base_tools());
        self
    }

    /// Set the maximum number of steps (default: 20).
    #[must_use]
    pub const fn max_steps(mut self, max: usize) -> Self {
        self.config.max_steps = max;
        self
    }

    /// Set the planning interval.
    #[must_use]
    pub const fn planning_interval(mut self, interval: usize) -> Self {
        self.config.planning_interval = Some(interval);
        self
    }

    /// Set the maximum number of concurrent tool calls.
    ///
    /// When a model returns multiple tool calls in a single response,
    /// they can be executed in parallel. This setting controls the
    /// maximum concurrency level.
    ///
    /// - `None` (default): Unlimited parallelism
    /// - `Some(1)`: Sequential execution (tools run one at a time)
    /// - `Some(n)`: Up to `n` tools run concurrently
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Allow up to 4 concurrent tool executions
    /// let agent = Agent::builder()
    ///     .model(model)
    ///     .max_parallel_tool_calls(4)
    ///     .build();
    ///
    /// // Force sequential execution
    /// let agent = Agent::builder()
    ///     .model(model)
    ///     .max_parallel_tool_calls(1)
    ///     .build();
    /// ```
    #[must_use]
    pub const fn max_parallel_tool_calls(mut self, max: usize) -> Self {
        self.config.max_parallel_tool_calls = Some(max);
        self
    }

    /// Set the agent's name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = Some(name.into());
        self
    }

    /// Set the agent's description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.config.description = Some(desc.into());
        self
    }

    /// Set custom prompt templates.
    ///
    /// By default, uses the built-in tool-calling agent templates.
    #[must_use]
    pub fn prompt_templates(mut self, templates: PromptTemplates) -> Self {
        self.prompt_templates = Some(templates);
        self
    }

    /// Set custom instructions to be included in the system prompt.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.custom_instructions = Some(instructions.into());
        self
    }

    /// Add a managed agent to delegate tasks to.
    ///
    /// Managed agents can be called by the parent agent as if they were tools,
    /// allowing for multi-agent collaboration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let researcher = Agent::builder()
    ///     .model(model.clone())
    ///     .name("researcher")
    ///     .description("Expert at finding information")
    ///     .tool(Box::new(WebSearchTool::new()))
    ///     .build();
    ///
    /// let agent = Agent::builder()
    ///     .model(model)
    ///     .managed_agent(Box::new(researcher))
    ///     .build();
    /// ```
    #[must_use]
    pub fn managed_agent(mut self, agent: BoxedManagedAgent) -> Self {
        self.managed_agents.push(agent);
        self
    }

    /// Add multiple managed agents.
    #[must_use]
    pub fn managed_agents(mut self, agents: impl IntoIterator<Item = BoxedManagedAgent>) -> Self {
        self.managed_agents.extend(agents);
        self
    }

    /// Set final answer validation checks.
    ///
    /// These checks run before accepting a final answer from the agent.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let agent = Agent::builder()
    ///     .model(model)
    ///     .final_answer_checks(
    ///         FinalAnswerChecks::new()
    ///             .not_null()
    ///             .not_empty()
    ///     )
    ///     .build();
    /// ```
    #[must_use]
    pub fn final_answer_checks(mut self, checks: FinalAnswerChecks) -> Self {
        self.final_answer_checks = checks;
        self
    }

    /// Set callback registry for step events.
    ///
    /// Callbacks are invoked when steps complete, allowing for monitoring,
    /// logging, and custom event handling.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use machi::callback::{CallbackRegistry, CallbackContext};
    /// use machi::memory::ActionStep;
    ///
    /// let agent = Agent::builder()
    ///     .model(model)
    ///     .callbacks(
    ///         CallbackRegistry::builder()
    ///             .on_action(|step, ctx| {
    ///                 println!("Step {} completed", step.step_number);
    ///             })
    ///             .build()
    ///     )
    ///     .build();
    /// ```
    #[must_use]
    pub fn callbacks(mut self, registry: CallbackRegistry) -> Self {
        self.callbacks = registry;
        self
    }

    /// Build the agent.
    ///
    /// # Panics
    ///
    /// Panics if no model is provided. Use [`try_build`](Self::try_build) for
    /// a fallible alternative.
    #[must_use]
    pub fn build(self) -> Agent {
        self.try_build().expect("Model is required")
    }

    /// Try to build the agent, returning an error if configuration is invalid.
    pub fn try_build(self) -> Result<Agent> {
        let model = self
            .model
            .ok_or_else(|| AgentError::configuration("Model is required"))?;

        let mut tools = ToolBox::new();
        for tool in self.tools {
            tools.add_boxed(tool);
        }
        tools.add(FinalAnswerTool);

        let mut managed_agents = ManagedAgentRegistry::new();
        for agent in self.managed_agents {
            if agent.name().is_empty() {
                return Err(AgentError::configuration("All managed agents need a name"));
            }
            if agent.description().is_empty() {
                return Err(AgentError::configuration(
                    "All managed agents need a description",
                ));
            }
            if tools.get(agent.name()).is_some() {
                return Err(AgentError::configuration(format!(
                    "Managed agent name '{}' conflicts with a tool name",
                    agent.name()
                )));
            }
            managed_agents.try_add(agent)?;
        }

        for tool in managed_agents.as_tools() {
            tools.add_boxed(tool);
        }

        let prompt_renderer = self
            .prompt_templates
            .map_or_else(PromptRender::default, PromptRender::new);

        Ok(Agent {
            model,
            tools,
            managed_agents,
            config: self.config,
            memory: AgentMemory::default(),
            system_prompt: String::new(),
            prompt_renderer,
            interrupt_flag: Arc::default(),
            step_number: 0,
            state: HashMap::new(),
            custom_instructions: self.custom_instructions,
            final_answer_checks: self.final_answer_checks,
            callbacks: self.callbacks,
            run_start: std::time::Instant::now(),
        })
    }
}
