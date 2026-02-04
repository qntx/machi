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

/// The state of an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunState {
    /// The run completed successfully with a final answer.
    Success,
    /// The run reached the maximum number of steps without a final answer.
    MaxStepsReached,
    /// The run was interrupted.
    Interrupted,
    /// The run failed with an error.
    Failed,
}

impl std::fmt::Display for RunState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::MaxStepsReached => write!(f, "max_steps_reached"),
            Self::Interrupted => write!(f, "interrupted"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Extended result of an agent run, containing detailed execution information.
///
/// Use `agent.run_with_result()` to get this instead of just the final answer.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// The final output of the agent run.
    pub output: Option<Value>,
    /// The state of the run (success, max_steps_reached, etc.).
    pub state: RunState,
    /// Total token usage during the run.
    pub token_usage: TokenUsage,
    /// Number of steps executed.
    pub steps_taken: usize,
    /// Timing information.
    pub timing: Timing,
    /// Error message if the run failed.
    pub error: Option<String>,
}

impl RunResult {
    /// Check if the run was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self.state, RunState::Success)
    }

    /// Get the output value, if available.
    #[must_use]
    pub fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    /// Generate a summary of the run.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Run State: {}\n", self.state));
        summary.push_str(&format!("Steps Taken: {}\n", self.steps_taken));
        summary.push_str(&format!(
            "Duration: {:.2}s\n",
            self.timing.duration_secs().unwrap_or_default()
        ));
        summary.push_str(&format!(
            "Tokens: {} (in: {}, out: {})\n",
            self.token_usage.total(),
            self.token_usage.input_tokens,
            self.token_usage.output_tokens
        ));
        if let Some(output) = &self.output {
            summary.push_str(&format!("Output: {output}\n"));
        }
        if let Some(error) = &self.error {
            summary.push_str(&format!("Error: {error}\n"));
        }
        summary
    }
}

/// A function that validates the final answer before accepting it.
///
/// The check function receives:
/// - `answer`: The final answer value
/// - `memory`: The agent's memory containing all steps
///
/// Returns `Ok(())` if the answer is valid, or `Err(reason)` if invalid.
pub type FinalAnswerCheck = Box<dyn Fn(&Value, &AgentMemory) -> Result<()> + Send + Sync>;

/// Builder for creating final answer checks.
pub struct FinalAnswerChecks {
    checks: Vec<FinalAnswerCheck>,
}

impl std::fmt::Debug for FinalAnswerChecks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FinalAnswerChecks")
            .field("checks_count", &self.checks.len())
            .finish()
    }
}

impl Default for FinalAnswerChecks {
    fn default() -> Self {
        Self::new()
    }
}

impl FinalAnswerChecks {
    /// Create a new empty set of checks.
    #[must_use]
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Add a check function.
    #[must_use]
    pub fn add<F>(mut self, check: F) -> Self
    where
        F: Fn(&Value, &AgentMemory) -> Result<()> + Send + Sync + 'static,
    {
        self.checks.push(Box::new(check));
        self
    }

    /// Add a check that the answer is not null.
    #[must_use]
    pub fn not_null(self) -> Self {
        self.add(|answer, _| {
            if answer.is_null() {
                Err(AgentError::configuration("Final answer cannot be null"))
            } else {
                Ok(())
            }
        })
    }

    /// Add a check that the answer is not an empty string.
    #[must_use]
    pub fn not_empty(self) -> Self {
        self.add(|answer, _| {
            if let Some(s) = answer.as_str() {
                if s.trim().is_empty() {
                    return Err(AgentError::configuration("Final answer cannot be empty"));
                }
            }
            Ok(())
        })
    }

    /// Add a check that the answer contains a specific substring.
    #[must_use]
    pub fn contains(self, substring: impl Into<String>) -> Self {
        let substring = substring.into();
        self.add(move |answer, _| {
            let text = answer.to_string();
            if !text.contains(&substring) {
                Err(AgentError::configuration(format!(
                    "Final answer must contain '{substring}'"
                )))
            } else {
                Ok(())
            }
        })
    }

    /// Run all checks on the given answer.
    pub(crate) fn validate(&self, answer: &Value, memory: &AgentMemory) -> Result<()> {
        for check in &self.checks {
            check(answer, memory)?;
        }
        Ok(())
    }

    /// Check if there are any checks defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }
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
    final_answer_checks: FinalAnswerChecks,
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
        let result = self.run_with_result_args(task, args).await;
        match result.state {
            RunState::Success => result.output.ok_or_else(|| {
                AgentError::configuration("Run succeeded but no output was produced")
            }),
            RunState::MaxStepsReached => Err(AgentError::max_steps(
                result.steps_taken,
                self.config.max_steps,
            )),
            RunState::Interrupted => Err(AgentError::Interrupted),
            RunState::Failed => Err(AgentError::configuration(
                result.error.unwrap_or_else(|| "Unknown error".to_string()),
            )),
        }
    }

    /// Run the agent and return a detailed result including summary.
    ///
    /// This method returns a [`RunResult`] containing:
    /// - The final output (if successful)
    /// - Run state (success, max_steps_reached, etc.)
    /// - Token usage statistics
    /// - Timing information
    /// - Number of steps taken
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = agent.run_with_result("What is 2 + 2?").await;
    /// println!("{}", result.summary());
    /// if result.is_success() {
    ///     println!("Answer: {}", result.output.unwrap());
    /// }
    /// ```
    #[inline]
    pub async fn run_with_result(&mut self, task: &str) -> RunResult {
        self.run_with_result_args(task, HashMap::new()).await
    }

    /// Run the agent with additional context and return a detailed result.
    #[instrument(skip(self, args), fields(max_steps = self.config.max_steps))]
    pub async fn run_with_result_args(
        &mut self,
        task: &str,
        args: HashMap<String, Value>,
    ) -> RunResult {
        self.prepare_run(task, args);
        info!("Starting agent run");

        let timing = Timing::start_now();
        let result = self.run_loop_with_checks().await;
        let mut final_timing = timing;
        final_timing.complete();

        let token_usage = self.memory.total_token_usage();
        let steps_taken = self.step_number;

        match result {
            Ok(answer) => {
                self.memory.add_step(FinalAnswerStep {
                    output: answer.clone(),
                });
                info!("Agent completed successfully");
                RunResult {
                    output: Some(answer),
                    state: RunState::Success,
                    token_usage,
                    steps_taken,
                    timing: final_timing,
                    error: None,
                }
            }
            Err(AgentError::MaxSteps { .. }) => {
                self.memory.add_step(FinalAnswerStep {
                    output: Value::String("Maximum steps reached".into()),
                });
                RunResult {
                    output: None,
                    state: RunState::MaxStepsReached,
                    token_usage,
                    steps_taken,
                    timing: final_timing,
                    error: Some("Maximum steps reached".to_string()),
                }
            }
            Err(AgentError::Interrupted) => RunResult {
                output: None,
                state: RunState::Interrupted,
                token_usage,
                steps_taken,
                timing: final_timing,
                error: Some("Agent was interrupted".to_string()),
            },
            Err(e) => {
                warn!(error = %e, "Agent run failed");
                RunResult {
                    output: None,
                    state: RunState::Failed,
                    token_usage,
                    steps_taken,
                    timing: final_timing,
                    error: Some(e.to_string()),
                }
            }
        }
    }

    /// Main execution loop with final answer checks.
    async fn run_loop_with_checks(&mut self) -> Result<Value> {
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
                    // Run final answer checks
                    if !self.final_answer_checks.is_empty() {
                        if let Err(e) = self.final_answer_checks.validate(&answer, &self.memory) {
                            warn!(error = %e, "Final answer check failed");
                            step.error = Some(format!("Final answer check failed: {e}"));
                            self.memory.add_step(step);
                            // Continue to next step instead of failing
                            continue;
                        }
                    }
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

        Err(AgentError::max_steps(
            self.step_number,
            self.config.max_steps,
        ))
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
    final_answer_checks: FinalAnswerChecks,
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
            final_answer_checks: self.final_answer_checks,
        })
    }
}
