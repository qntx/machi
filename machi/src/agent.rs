//! AI Agent for executing tasks with tools.
//!
//! This module provides a lightweight, ergonomic agent that uses LLM function
//! calling to accomplish tasks through tool invocations.
//!
//! # Example
//!
//! ```rust,ignore
//! let mut agent = Agent::builder()
//!     .model(model)
//!     .tool(Box::new(MyTool))
//!     .build();
//!
//! let result = agent.run("What is 2 + 2?").await?;
//! ```

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_stream::stream;
use futures::Stream;
use serde_json::Value;
use tracing::{debug, info, instrument, warn};

use crate::{
    error::{AgentError, Result},
    memory::{ActionStep, AgentMemory, FinalAnswerStep, TaskStep, Timing, ToolCall},
    prompts::{PromptEngine, PromptTemplates, TemplateContext},
    providers::common::{GenerateOptions, Model, TokenUsage},
    tool::{BoxedTool, ToolBox},
    tools::FinalAnswerTool,
};

/// Configuration for an agent.
#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    /// Maximum number of steps (default: 20).
    #[doc(hidden)]
    pub max_steps: usize,
    /// Planning interval (run planning every N steps).
    pub planning_interval: Option<usize>,
    /// Agent name.
    pub name: Option<String>,
    /// Agent description.
    pub description: Option<String>,
}

impl AgentConfig {
    const DEFAULT_MAX_STEPS: usize = 20;

    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_steps: Self::DEFAULT_MAX_STEPS,
            ..Default::default()
        }
    }
}

/// Events emitted during streaming agent execution.
///
/// These events allow real-time observation of agent progress, including
/// model output chunks, tool calls, and step completions.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text content from the model.
    TextDelta(String),

    /// A tool call is being made.
    ToolCallStart {
        /// Tool call ID.
        id: String,
        /// Name of the tool being called.
        name: String,
    },

    /// Tool execution completed.
    ToolCallComplete {
        /// Tool call ID.
        id: String,
        /// Name of the tool.
        name: String,
        /// Result of the tool execution.
        result: std::result::Result<String, String>,
    },

    /// An action step has completed.
    StepComplete {
        /// Step number.
        step: usize,
        /// The completed action step data.
        action_step: ActionStep,
    },

    /// The agent has produced a final answer.
    FinalAnswer {
        /// The final answer value.
        answer: Value,
    },

    /// Token usage information.
    TokenUsage(TokenUsage),

    /// An error occurred.
    Error(String),
}

/// Type alias for the streaming event result.
pub type StreamItem = std::result::Result<StreamEvent, AgentError>;

/// Type alias for the boxed stream of events.
pub type AgentStream = Pin<Box<dyn Stream<Item = StreamItem> + Send>>;

/// Result of a streaming step execution.
#[derive(Debug)]
enum StepResult {
    /// Continue to next step.
    Continue,
    /// Agent produced a final answer.
    FinalAnswer(Value),
}

/// AI agent that uses LLM function calling to execute tasks with tools.
///
/// The agent follows a ReAct-style loop:
/// 1. Receive a task
/// 2. Think and decide which tool to call
/// 3. Execute the tool and observe the result
/// 4. Repeat until `final_answer` is called or max steps reached
pub struct Agent {
    model: Box<dyn Model>,
    tools: ToolBox,
    config: AgentConfig,
    memory: AgentMemory,
    system_prompt: String,
    prompt_templates: PromptTemplates,
    prompt_engine: PromptEngine,
    interrupt_flag: Arc<AtomicBool>,
    step_number: usize,
    state: HashMap<String, Value>,
    custom_instructions: Option<String>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("config", &self.config)
            .field("tools", &self.tools)
            .field("step", &self.step_number)
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Create a new agent builder.
    #[inline]
    #[must_use]
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Build the system prompt from available tools using Jinja2 templates.
    ///
    /// This method renders the system_prompt template with the current context,
    /// including tool definitions and custom instructions.
    fn build_system_prompt(&self) -> String {
        let defs = self.tools.definitions();
        let ctx = TemplateContext::new()
            .with_tools(&defs)
            .with_custom_instructions_opt(self.custom_instructions.as_deref());

        match self
            .prompt_engine
            .render(&self.prompt_templates.system_prompt, &ctx)
        {
            Ok(rendered) => rendered,
            Err(e) => {
                warn!(error = %e, "Failed to render system prompt template, using fallback");
                self.build_fallback_system_prompt()
            }
        }
    }

    /// Fallback system prompt when template rendering fails.
    fn build_fallback_system_prompt(&self) -> String {
        let tools = self
            .tools
            .definitions()
            .iter()
            .map(|t| format!("- {}: {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "You are a helpful AI assistant that can use tools to accomplish tasks.\n\n\
             Available tools:\n{tools}\n\n\
             When you need to use a tool, respond with a tool call. \
             When you have the final answer, use the 'final_answer' tool to provide it.\n\n\
             Think step by step about what you need to do to accomplish the task."
        )
    }

    /// Execute a single reasoning step.
    async fn execute_step(&self, step: &mut ActionStep) -> Result<Option<Value>> {
        let messages = self.memory.to_messages(false);
        step.model_input_messages = Some(messages.clone());

        let options = GenerateOptions::new().with_tools(self.tools.definitions());
        debug!(step = step.step_number, "Generating model response");

        let response = self.model.generate(messages, options).await?;
        step.model_output_message = Some(response.message.clone());
        step.token_usage = response.token_usage;
        step.model_output = response.message.text_content();

        let Some(tool_calls) = &response.message.tool_calls else {
            return Ok(None);
        };

        let mut observations = Vec::with_capacity(tool_calls.len());
        let mut final_answer = None;

        for tc in tool_calls {
            step.tool_calls
                .get_or_insert_with(Vec::new)
                .push(ToolCall::new(&tc.id, tc.name(), tc.arguments().clone()));

            if tc.name() == "final_answer" {
                if let Ok(args) = tc.parse_arguments::<crate::tools::FinalAnswerArgs>() {
                    final_answer = Some(args.answer);
                    step.is_final_answer = true;
                }
                continue;
            }

            match self.tools.call(tc.name(), tc.arguments().clone()).await {
                Ok(result) => observations.push(format!("Tool '{}' returned: {result}", tc.name())),
                Err(e) => {
                    let msg = format!("Tool '{}' failed: {e}", tc.name());
                    observations.push(msg.clone());
                    step.error = Some(msg);
                }
            }
        }

        if !observations.is_empty() {
            step.observations = Some(observations.join("\n"));
        }

        if let Some(answer) = final_answer {
            step.action_output = Some(answer.clone());
            return Ok(Some(answer));
        }

        Ok(None)
    }

    /// Run the agent with a task.
    #[inline]
    pub async fn run(&mut self, task: &str) -> Result<Value> {
        self.run_with_args(task, HashMap::new()).await
    }

    /// Run the agent with a task and additional context.
    #[instrument(skip(self, args), fields(max_steps = self.config.max_steps))]
    pub async fn run_with_args(
        &mut self,
        task: &str,
        args: HashMap<String, Value>,
    ) -> Result<Value> {
        self.prepare_run(task, args);
        info!("Starting agent run");

        let result = self.run_loop().await;

        match &result {
            Ok(answer) => {
                self.memory.add_step(FinalAnswerStep {
                    output: answer.clone(),
                });
                info!("Agent completed successfully");
            }
            Err(e) => warn!(error = %e, "Agent run failed"),
        }

        result
    }

    /// Run the agent with streaming output.
    ///
    /// Returns a stream of [`StreamEvent`]s that yields events as the agent executes.
    /// The stream completes after a `FinalAnswer` event or when an error occurs.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = agent.run_stream("What is 2 + 2?");
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         StreamEvent::StepComplete { step, .. } => println!("Step {step} done"),
    ///         StreamEvent::FinalAnswer { answer } => println!("Answer: {answer}"),
    ///         StreamEvent::Error(e) => eprintln!("Error: {e}"),
    ///         _ => {}
    ///     }
    /// }
    ///
    /// // Or use Stream combinators
    /// let events: Vec<_> = agent.run_stream("task").collect().await;
    /// ```
    #[instrument(skip(self), fields(max_steps = self.config.max_steps))]
    pub fn run_stream(&mut self, task: &str) -> impl Stream<Item = StreamItem> + '_ {
        self.run_stream_with_args(task, HashMap::new())
    }

    /// Run the agent with streaming output and additional context.
    #[instrument(skip(self, args), fields(max_steps = self.config.max_steps))]
    #[allow(tail_expr_drop_order)]
    pub fn run_stream_with_args(
        &mut self,
        task: &str,
        args: HashMap<String, Value>,
    ) -> impl Stream<Item = StreamItem> + '_ {
        self.prepare_run(task, args);
        info!("Starting streaming agent run");

        stream! {
            loop {
                match self.next_stream_event().await {
                    Some(Ok(event)) => {
                        let is_final = matches!(event, StreamEvent::FinalAnswer { .. });
                        yield Ok(event);
                        if is_final {
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        yield Err(e);
                        break;
                    }
                    None => break,
                }
            }
        }
    }

    /// Get the next streaming event from the agent execution.
    ///
    /// Returns `None` when the agent run is complete.
    async fn next_stream_event(&mut self) -> Option<StreamItem> {
        // Check if we've reached max steps
        if self.step_number >= self.config.max_steps {
            self.memory.add_step(FinalAnswerStep {
                output: Value::String("Maximum steps reached".into()),
            });
            return Some(Err(AgentError::max_steps(
                self.step_number,
                self.config.max_steps,
            )));
        }

        // Check for interrupt
        if self.interrupt_flag.load(Ordering::SeqCst) {
            return Some(Err(AgentError::Interrupted));
        }

        self.step_number += 1;

        let mut step = ActionStep {
            step_number: self.step_number,
            timing: Timing::start_now(),
            ..Default::default()
        };

        // Execute step with streaming
        match self.execute_step_streaming(&mut step).await {
            Ok(StepResult::Continue) => {
                step.timing.complete();
                let step_clone = step.clone();
                self.memory.add_step(step);
                Some(Ok(StreamEvent::StepComplete {
                    step: self.step_number,
                    action_step: step_clone,
                }))
            }
            Ok(StepResult::FinalAnswer(answer)) => {
                step.timing.complete();
                step.is_final_answer = true;
                step.action_output = Some(answer.clone());
                self.memory.add_step(step);
                self.memory.add_step(FinalAnswerStep {
                    output: answer.clone(),
                });
                info!("Agent completed successfully");
                // Return final answer - stream will end after this
                Some(Ok(StreamEvent::FinalAnswer { answer }))
            }
            Err(e) => {
                step.timing.complete();
                step.error = Some(e.to_string());
                self.memory.add_step(step);
                warn!(step = self.step_number, error = %e, "Step failed");
                Some(Ok(StreamEvent::Error(e.to_string())))
            }
        }
    }

    /// Execute a single step with streaming support.
    async fn execute_step_streaming(&mut self, step: &mut ActionStep) -> Result<StepResult> {
        let messages = self.memory.to_messages(false);
        step.model_input_messages = Some(messages.clone());

        let options = GenerateOptions::new().with_tools(self.tools.definitions());
        debug!(
            step = step.step_number,
            "Generating model response (streaming)"
        );

        // Use streaming if the model supports it
        if self.model.supports_streaming() {
            let mut stream = self.model.generate_stream(messages, options).await?;
            let mut deltas = Vec::new();

            use futures::StreamExt;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(delta) => {
                        if let Some(usage) = &delta.token_usage {
                            step.token_usage = Some(usage.clone());
                        }
                        deltas.push(delta);
                    }
                    Err(e) => return Err(e),
                }
            }

            // Aggregate deltas into a complete message
            let message = crate::message::aggregate_stream_deltas(&deltas);
            step.model_output_message = Some(message.clone());
            step.model_output = message.text_content();

            self.process_tool_calls(step, &message).await
        } else {
            // Fall back to non-streaming
            let response = self.model.generate(messages, options).await?;
            step.model_output_message = Some(response.message.clone());
            step.token_usage = response.token_usage;
            step.model_output = response.message.text_content();

            self.process_tool_calls(step, &response.message).await
        }
    }

    /// Process tool calls from the model response.
    async fn process_tool_calls(
        &self,
        step: &mut ActionStep,
        message: &crate::message::ChatMessage,
    ) -> Result<StepResult> {
        let Some(tool_calls) = &message.tool_calls else {
            return Ok(StepResult::Continue);
        };

        let mut observations = Vec::with_capacity(tool_calls.len());
        let mut final_answer = None;

        for tc in tool_calls {
            step.tool_calls
                .get_or_insert_with(Vec::new)
                .push(ToolCall::new(&tc.id, tc.name(), tc.arguments().clone()));

            if tc.name() == "final_answer" {
                if let Ok(args) = tc.parse_arguments::<crate::tools::FinalAnswerArgs>() {
                    final_answer = Some(args.answer);
                }
                continue;
            }

            match self.tools.call(tc.name(), tc.arguments().clone()).await {
                Ok(result) => observations.push(format!("Tool '{}' returned: {result}", tc.name())),
                Err(e) => {
                    let msg = format!("Tool '{}' failed: {e}", tc.name());
                    observations.push(msg.clone());
                    step.error = Some(msg);
                }
            }
        }

        if !observations.is_empty() {
            step.observations = Some(observations.join("\n"));
        }

        if let Some(answer) = final_answer {
            return Ok(StepResult::FinalAnswer(answer));
        }

        Ok(StepResult::Continue)
    }

    /// Prepare the agent for a new run.
    fn prepare_run(&mut self, task: &str, args: HashMap<String, Value>) {
        self.memory.reset();
        self.step_number = 0;
        self.interrupt_flag.store(false, Ordering::SeqCst);
        self.state = args;

        self.system_prompt = self.build_system_prompt();
        self.memory
            .system_prompt
            .system_prompt
            .clone_from(&self.system_prompt);

        let task_text = if self.state.is_empty() {
            task.to_string()
        } else {
            format!(
                "{task}\n\nAdditional context provided:\n{}",
                serde_json::to_string_pretty(&self.state).unwrap_or_default()
            )
        };

        self.memory.add_step(TaskStep {
            task: task_text,
            task_images: None,
        });
    }

    /// Main execution loop.
    async fn run_loop(&mut self) -> Result<Value> {
        while self.step_number < self.config.max_steps {
            if self.interrupt_flag.load(Ordering::SeqCst) {
                return Err(AgentError::Interrupted);
            }

            self.step_number += 1;

            let mut step = ActionStep {
                step_number: self.step_number,
                timing: Timing::start_now(),
                ..Default::default()
            };

            let result = self.execute_step(&mut step).await;
            step.timing.complete();

            match result {
                Ok(Some(answer)) => {
                    self.memory.add_step(step);
                    return Ok(answer);
                }
                Ok(None) => self.memory.add_step(step),
                Err(e) => {
                    step.error = Some(e.to_string());
                    self.memory.add_step(step);
                    warn!(step = self.step_number, error = %e, "Step failed");
                }
            }
        }

        self.memory.add_step(FinalAnswerStep {
            output: Value::String("Maximum steps reached".into()),
        });
        Err(AgentError::max_steps(
            self.step_number,
            self.config.max_steps,
        ))
    }

    /// Get the agent's name.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }

    /// Get the agent's description.
    #[inline]
    pub fn description(&self) -> Option<&str> {
        self.config.description.as_deref()
    }

    /// Request the agent to stop after the current step.
    #[inline]
    pub fn interrupt(&self) {
        self.interrupt_flag.store(true, Ordering::SeqCst);
    }

    /// Check if an interrupt has been requested.
    #[inline]
    pub fn is_interrupted(&self) -> bool {
        self.interrupt_flag.load(Ordering::SeqCst)
    }

    /// Get the agent's memory.
    #[inline]
    pub const fn memory(&self) -> &AgentMemory {
        &self.memory
    }

    /// Get mutable access to the agent's memory.
    #[inline]
    pub const fn memory_mut(&mut self) -> &mut AgentMemory {
        &mut self.memory
    }

    /// Reset the agent for a new task.
    pub fn reset(&mut self) {
        self.memory.reset();
        self.step_number = 0;
        self.state.clear();
        self.interrupt_flag.store(false, Ordering::SeqCst);
    }

    /// Get the current step number.
    #[inline]
    pub const fn current_step(&self) -> usize {
        self.step_number
    }
}

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
    config: AgentConfig,
    prompt_templates: Option<PromptTemplates>,
    custom_instructions: Option<String>,
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("has_model", &self.model.is_some())
            .field("tools", &self.tools.len())
            .finish_non_exhaustive()
    }
}

impl AgentBuilder {
    /// Create a new builder with default settings.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AgentConfig::new(),
            ..Default::default()
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

        let prompt_templates = self
            .prompt_templates
            .unwrap_or_else(PromptTemplates::toolcalling_agent);

        Ok(Agent {
            model,
            tools,
            config: self.config,
            memory: AgentMemory::default(),
            system_prompt: String::new(),
            prompt_templates,
            prompt_engine: PromptEngine::new(),
            interrupt_flag: Arc::default(),
            step_number: 0,
            state: HashMap::new(),
            custom_instructions: self.custom_instructions,
        })
    }
}
