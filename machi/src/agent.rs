//! Agent implementations for executing tasks with tools.
//!
//! This module provides the core agent types that can execute tasks
//! using language models and tools.

use crate::error::{AgentError, Result};
use crate::memory::{
    ActionStep, AgentMemory, FinalAnswerStep, TaskStep, Timing, TokenUsage, ToolCall,
};
use crate::providers::common::{GenerateOptions, Model};
use crate::tool::{BoxedTool, ToolBox, ToolDefinition};
use crate::tools::FinalAnswerTool;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info, warn};

/// Result of an agent run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Final output of the agent.
    pub output: Option<Value>,
    /// State of the run ("success" or "`max_steps_error`").
    pub state: String,
    /// Steps taken during the run.
    pub steps: Vec<Value>,
    /// Token usage during the run.
    pub token_usage: Option<TokenUsage>,
    /// Timing information.
    pub timing: Timing,
}

/// Configuration for an agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Planning interval (run planning every N steps).
    pub planning_interval: Option<usize>,
    /// Agent name (for managed agents).
    pub name: Option<String>,
    /// Agent description (for managed agents).
    pub description: Option<String>,
    /// Whether to stream outputs.
    pub stream_outputs: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 20,
            planning_interval: None,
            name: None,
            description: None,
            stream_outputs: false,
        }
    }
}

/// AI agent that uses LLM function calling to execute tasks with tools.
pub struct Agent {
    /// The language model to use.
    model: Box<dyn Model>,
    /// Available tools.
    tools: ToolBox,
    /// Agent configuration.
    config: AgentConfig,
    /// Agent memory.
    memory: AgentMemory,
    /// System prompt template.
    system_prompt: String,
    /// Interrupt flag.
    interrupt_flag: Arc<AtomicBool>,
    /// Current step number.
    step_number: usize,
    /// Current task.
    current_task: Option<String>,
    /// Additional state.
    state: HashMap<String, Value>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("config", &self.config)
            .field("tools", &self.tools)
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Create a new agent builder.
    #[must_use]
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Initialize the system prompt.
    fn initialize_system_prompt(&self) -> String {
        let tool_descriptions: Vec<String> = self
            .tools
            .definitions()
            .iter()
            .map(|t| format!("- {}: {}", t.name, t.description))
            .collect();

        format!(
            r"You are a helpful AI assistant that can use tools to accomplish tasks.

Available tools:
{}

When you need to use a tool, respond with a tool call. When you have the final answer, use the 'final_answer' tool to provide it.

Think step by step about what you need to do to accomplish the task.",
            tool_descriptions.join("\n")
        )
    }

    /// Perform a single step.
    async fn step(&self, action_step: &mut ActionStep) -> Result<Option<Value>> {
        // Build messages from memory
        let messages = self.memory.to_messages(false);
        action_step.model_input_messages = Some(messages.clone());

        // Get tool definitions
        let tool_defs: Vec<ToolDefinition> = self.tools.definitions();

        // Generate response
        let options = GenerateOptions::new().with_tools(tool_defs);

        debug!(
            "Generating model response for step {}",
            action_step.step_number
        );
        let response = self.model.generate(messages, options).await?;

        action_step.model_output_message = Some(response.message.clone());
        action_step.token_usage = response.token_usage;

        if let Some(text) = response.message.text_content() {
            action_step.model_output = Some(text);
        }

        // Check for tool calls
        if let Some(tool_calls) = &response.message.tool_calls {
            let mut observations = Vec::new();
            let mut final_answer: Option<Value> = None;

            for tc in tool_calls {
                let tool_call = ToolCall::new(&tc.id, tc.name(), tc.arguments().clone());
                action_step
                    .tool_calls
                    .get_or_insert_with(Vec::new)
                    .push(tool_call);

                // Check for final answer
                if tc.name() == "final_answer" {
                    if let Ok(args) = tc.parse_arguments::<crate::tools::FinalAnswerArgs>() {
                        final_answer = Some(args.answer);
                        action_step.is_final_answer = true;
                    }
                    continue;
                }

                // Execute tool
                match self.tools.call(tc.name(), tc.arguments().clone()).await {
                    Ok(result) => {
                        let obs = format!("Tool '{}' returned: {}", tc.name(), result);
                        observations.push(obs);
                    }
                    Err(e) => {
                        let err_msg = format!("Tool '{}' failed: {}", tc.name(), e);
                        observations.push(err_msg.clone());
                        action_step.error = Some(err_msg);
                    }
                }
            }

            if !observations.is_empty() {
                action_step.observations = Some(observations.join("\n"));
            }

            if let Some(answer) = final_answer {
                action_step.action_output = Some(answer.clone());
                return Ok(Some(answer));
            }
        }

        Ok(None)
    }

    /// Run the agent with a task.
    pub async fn run(&mut self, task: &str) -> Result<Value> {
        self.run_with_args(task, HashMap::new()).await
    }

    /// Run the agent with a task and additional arguments.
    pub async fn run_with_args(
        &mut self,
        task: &str,
        args: HashMap<String, Value>,
    ) -> Result<Value> {
        // Reset state
        self.memory.reset();
        self.step_number = 0;
        self.interrupt_flag.store(false, Ordering::SeqCst);
        self.state = args;

        // Set up system prompt
        self.system_prompt = self.initialize_system_prompt();
        self.memory
            .system_prompt
            .system_prompt
            .clone_from(&self.system_prompt);

        // Store task
        let task_with_args = if self.state.is_empty() {
            task.to_string()
        } else {
            format!(
                "{}\n\nAdditional context provided:\n{}",
                task,
                serde_json::to_string_pretty(&self.state).unwrap_or_default()
            )
        };
        self.current_task = Some(task_with_args.clone());

        // Add task step
        self.memory.add_step(TaskStep {
            task: task_with_args,
            task_images: None,
        });

        info!(
            "Starting agent run with max {} steps",
            self.config.max_steps
        );

        // Main loop
        let mut final_answer: Option<Value> = None;

        while self.step_number < self.config.max_steps {
            // Check interrupt
            if self.interrupt_flag.load(Ordering::SeqCst) {
                return Err(AgentError::Interrupted);
            }

            self.step_number += 1;
            info!("Executing step {}", self.step_number);

            let mut action_step = ActionStep {
                step_number: self.step_number,
                timing: Timing::start_now(),
                ..Default::default()
            };

            match self.step(&mut action_step).await {
                Ok(Some(answer)) => {
                    action_step.timing.complete();
                    self.memory.add_step(action_step);
                    final_answer = Some(answer);
                    break;
                }
                Ok(None) => {
                    action_step.timing.complete();
                    self.memory.add_step(action_step);
                }
                Err(e) => {
                    action_step.error = Some(e.to_string());
                    action_step.timing.complete();
                    self.memory.add_step(action_step);
                    warn!("Step {} error: {}", self.step_number, e);
                }
            }
        }

        // Handle max steps reached
        if final_answer.is_none() {
            warn!("Reached maximum steps ({})", self.config.max_steps);
            self.memory.add_step(FinalAnswerStep {
                output: Value::String("Maximum steps reached without final answer".to_string()),
            });
            return Err(AgentError::max_steps(
                self.step_number,
                self.config.max_steps,
            ));
        }

        // Add final answer step
        if let Some(ref answer) = final_answer {
            self.memory.add_step(FinalAnswerStep {
                output: answer.clone(),
            });
        }

        info!("Agent completed successfully");
        final_answer.ok_or_else(|| AgentError::internal("No final answer produced"))
    }

    /// Get the agent's name.
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    /// Get the agent's description.
    pub fn description(&self) -> Option<&str> {
        self.config.description.as_deref()
    }

    /// Interrupt the agent's execution.
    pub fn interrupt(&self) {
        self.interrupt_flag.store(true, Ordering::SeqCst);
    }

    /// Get the agent's memory.
    pub const fn memory(&self) -> &AgentMemory {
        &self.memory
    }

    /// Reset the agent's state and memory.
    pub fn reset(&mut self) {
        self.memory.reset();
        self.step_number = 0;
        self.current_task = None;
        self.state.clear();
    }
}

/// Builder for [`Agent`].
#[derive(Default)]
pub struct AgentBuilder {
    model: Option<Box<dyn Model>>,
    tools: Vec<BoxedTool>,
    config: AgentConfig,
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("has_model", &self.model.is_some())
            .field("tools_count", &self.tools.len())
            .field("config", &self.config)
            .finish()
    }
}

impl AgentBuilder {
    /// Set the model.
    #[must_use]
    pub fn model<M: Model + 'static>(mut self, model: M) -> Self {
        self.model = Some(Box::new(model));
        self
    }

    /// Add a tool.
    #[must_use]
    pub fn tool(mut self, tool: BoxedTool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools.
    #[must_use]
    pub fn tools(mut self, tools: Vec<BoxedTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set max steps.
    #[must_use]
    pub const fn max_steps(mut self, max: usize) -> Self {
        self.config.max_steps = max;
        self
    }

    /// Set planning interval.
    #[must_use]
    pub const fn planning_interval(mut self, interval: usize) -> Self {
        self.config.planning_interval = Some(interval);
        self
    }

    /// Set agent name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = Some(name.into());
        self
    }

    /// Set agent description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.config.description = Some(desc.into());
        self
    }

    /// Build the agent.
    ///
    /// # Panics
    ///
    /// Panics if no model is provided.
    #[must_use]
    pub fn build(self) -> Agent {
        let model = self.model.expect("Model is required");
        let mut toolbox = ToolBox::new();

        // Add user tools
        for tool in self.tools {
            toolbox.add_boxed(tool);
        }

        // Add final answer tool
        toolbox.add(FinalAnswerTool);

        Agent {
            model,
            tools: toolbox,
            config: self.config,
            memory: AgentMemory::default(),
            system_prompt: String::new(),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            step_number: 0,
            current_task: None,
            state: HashMap::new(),
        }
    }
}
