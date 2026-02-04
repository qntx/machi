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

mod builder;
mod checks;
mod config;
mod events;
mod result;

pub use builder::AgentBuilder;
pub use checks::{FinalAnswerCheck, FinalAnswerChecks};
pub use config::AgentConfig;
pub use events::{AgentStream, RunState, StreamEvent, StreamItem};
pub use result::RunResult;

use events::StepResult;

use std::{
    collections::HashMap,
    fmt::Write as _,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_stream::stream;
use futures::{Stream, StreamExt};
use serde_json::Value;
use tracing::{debug, info, instrument, warn};

use crate::{
    error::{AgentError, Result},
    managed_agent::{ManagedAgent, ManagedAgentRegistry},
    memory::{ActionStep, AgentMemory, FinalAnswerStep, TaskStep, Timing, ToolCall},
    message::{ChatMessage, ChatMessageToolCall},
    multimodal::AgentImage,
    prompts::{PromptEngine, PromptTemplates, TemplateContext},
    providers::common::{GenerateOptions, Model},
    telemetry::{RunMetrics, Telemetry},
    tool::ToolBox,
};

/// AI agent that uses LLM function calling to execute tasks with tools.
///
/// The agent follows a ReAct-style loop:
/// 1. Receive a task
/// 2. Think and decide which tool to call
/// 3. Execute the tool and observe the result
/// 4. Repeat until `final_answer` is called or max steps reached
pub struct Agent {
    pub(crate) model: Box<dyn Model>,
    pub(crate) tools: ToolBox,
    pub(crate) managed_agents: ManagedAgentRegistry,
    pub(crate) config: AgentConfig,
    pub(crate) memory: AgentMemory,
    pub(crate) system_prompt: String,
    pub(crate) prompt_templates: PromptTemplates,
    pub(crate) prompt_engine: PromptEngine,
    pub(crate) interrupt_flag: Arc<AtomicBool>,
    pub(crate) step_number: usize,
    pub(crate) state: HashMap<String, Value>,
    pub(crate) custom_instructions: Option<String>,
    pub(crate) final_answer_checks: FinalAnswerChecks,
    pub(crate) telemetry: Telemetry,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("config", &self.config)
            .field("tools", &self.tools)
            .field("managed_agents", &self.managed_agents)
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

    /// Run the agent with a task.
    #[inline]
    pub async fn run(&mut self, task: &str) -> Result<Value> {
        self.run_with_args(task, HashMap::new()).await
    }

    /// Run the agent with images (for vision models).
    #[inline]
    pub async fn run_with_images(&mut self, task: &str, images: Vec<AgentImage>) -> Result<Value> {
        self.invoke(task, images, HashMap::new()).await
    }

    /// Run the agent with additional context arguments.
    #[instrument(skip(self, args), fields(max_steps = self.config.max_steps))]
    pub async fn run_with_args(
        &mut self,
        task: &str,
        args: HashMap<String, Value>,
    ) -> Result<Value> {
        self.execute_with_args(task, args)
            .await
            .into_result(self.config.max_steps)
    }

    /// Run the agent with images and context arguments.
    #[instrument(skip(self, images, args), fields(max_steps = self.config.max_steps, image_count = images.len()))]
    pub async fn invoke(
        &mut self,
        task: &str,
        images: Vec<AgentImage>,
        args: HashMap<String, Value>,
    ) -> Result<Value> {
        self.execute_full(task, images, args)
            .await
            .into_result(self.config.max_steps)
    }

    /// Execute and return detailed [`RunResult`].
    #[inline]
    pub async fn execute(&mut self, task: &str) -> RunResult {
        self.execute_with_args(task, HashMap::new()).await
    }

    /// Execute with images and return detailed [`RunResult`].
    #[inline]
    pub async fn execute_with_images(&mut self, task: &str, images: Vec<AgentImage>) -> RunResult {
        self.execute_full(task, images, HashMap::new()).await
    }

    /// Execute with context arguments and return detailed [`RunResult`].
    #[instrument(skip(self, args), fields(max_steps = self.config.max_steps))]
    pub async fn execute_with_args(
        &mut self,
        task: &str,
        args: HashMap<String, Value>,
    ) -> RunResult {
        self.execute_full(task, Vec::new(), args).await
    }

    /// Execute with images and context arguments, return detailed [`RunResult`].
    #[instrument(skip(self, images, args), fields(max_steps = self.config.max_steps, image_count = images.len()))]
    pub async fn execute_full(
        &mut self,
        task: &str,
        images: Vec<AgentImage>,
        args: HashMap<String, Value>,
    ) -> RunResult {
        self.init_run(task, images, args);
        info!("Starting agent run");

        let timing = Timing::start_now();
        let result = self.step_loop().await;
        let mut final_timing = timing;
        final_timing.complete();

        self.finalize(result, final_timing)
    }

    /// Stream execution events.
    #[instrument(skip(self), fields(max_steps = self.config.max_steps))]
    pub fn stream(&mut self, task: &str) -> impl Stream<Item = StreamItem> + '_ {
        self.stream_with_args(task, HashMap::new())
    }

    /// Stream execution events with context arguments.
    #[instrument(skip(self, args), fields(max_steps = self.config.max_steps))]
    #[expect(
        tail_expr_drop_order,
        reason = "stream yields control flow intentionally"
    )]
    pub fn stream_with_args(
        &mut self,
        task: &str,
        args: HashMap<String, Value>,
    ) -> impl Stream<Item = StreamItem> + '_ {
        self.init_run(task, Vec::new(), args);
        info!("Starting streaming agent run");

        stream! {
            loop {
                match self.poll_event().await {
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

    /// Execute as a managed sub-agent.
    pub async fn call_as_managed(&mut self, task: &str) -> Result<String> {
        let agent_name = self.config.name.clone().unwrap_or_else(|| "agent".into());
        let full_task = self.format_task_prompt(&agent_name, task);
        let result: Value = self.run(&full_task).await?;

        let report = match result {
            Value::Null => "No result produced".to_string(),
            Value::String(s) => s,
            other => other.to_string(),
        };

        let mut answer = self.format_report(&agent_name, &report);

        if self.config.provide_run_summary.unwrap_or(false) {
            self.append_summary(&mut answer);
        }

        Ok(answer)
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

    /// Get the current step number.
    #[inline]
    pub const fn current_step(&self) -> usize {
        self.step_number
    }

    /// Get a reference to the telemetry collector.
    #[inline]
    pub const fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    /// Get the telemetry metrics for the current/last run.
    #[inline]
    #[must_use]
    pub fn metrics(&mut self) -> RunMetrics {
        self.telemetry.complete()
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

    /// Reset the agent for a new task.
    pub fn reset(&mut self) {
        self.memory.reset();
        self.step_number = 0;
        self.state.clear();
        self.interrupt_flag.store(false, Ordering::SeqCst);
        self.telemetry.reset();
    }
}

impl Agent {
    fn init_run(&mut self, task: &str, images: Vec<AgentImage>, args: HashMap<String, Value>) {
        self.memory.reset();
        self.step_number = 0;
        self.interrupt_flag.store(false, Ordering::SeqCst);
        self.state = args;

        self.system_prompt = self.render_system_prompt();
        self.memory
            .system_prompt
            .system_prompt
            .clone_from(&self.system_prompt);

        let task_text = self.format_task(task);
        let task_step = if images.is_empty() {
            TaskStep::new(task_text)
        } else {
            TaskStep::with_images(task_text, images)
        };
        self.memory.add_step(task_step);
    }

    fn format_task(&self, task: &str) -> String {
        if self.state.is_empty() {
            task.into()
        } else {
            let context = serde_json::to_string_pretty(&self.state).unwrap_or_default();
            let mut text = String::with_capacity(task.len() + 32 + context.len());
            text.push_str(task);
            text.push_str("\n\nAdditional context provided:\n");
            text.push_str(&context);
            text
        }
    }

    fn finalize(&mut self, result: Result<Value>, timing: Timing) -> RunResult {
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
                    timing,
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
                    timing,
                    error: Some("Maximum steps reached".to_string()),
                }
            }
            Err(AgentError::Interrupted) => RunResult {
                output: None,
                state: RunState::Interrupted,
                token_usage,
                steps_taken,
                timing,
                error: Some("Agent was interrupted".to_string()),
            },
            Err(e) => {
                warn!(error = %e, "Agent run failed");
                RunResult {
                    output: None,
                    state: RunState::Failed,
                    token_usage,
                    steps_taken,
                    timing,
                    error: Some(e.to_string()),
                }
            }
        }
    }
}

impl Agent {
    async fn step_loop(&mut self) -> Result<Value> {
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

            let result = self.run_step(&mut step).await;
            step.timing.complete();
            self.record_telemetry(&step);

            match result {
                Ok(StepResult::FinalAnswer(answer)) => {
                    if let Err(e) = self.check_answer(&answer) {
                        warn!(error = %e, "Final answer check failed");
                        step.error = Some(format!("Final answer check failed: {e}"));
                        self.telemetry.record_error(&e.to_string());
                        self.memory.add_step(step);
                        continue;
                    }
                    self.memory.add_step(step);
                    return Ok(answer);
                }
                Ok(StepResult::Continue) => {
                    self.memory.add_step(step);
                }
                Err(e) => {
                    let err_msg = e.to_string();
                    step.error = Some(err_msg.clone());
                    self.telemetry.record_error(&err_msg);
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

    async fn run_step(&self, step: &mut ActionStep) -> Result<StepResult> {
        let messages = self.memory.to_messages(false);
        step.model_input_messages = Some(messages.clone());

        let options = GenerateOptions::new().with_tools(self.tools.definitions());
        debug!(step = step.step_number, "Generating model response");

        let message = self.call_model(messages, options, step).await?;
        self.handle_response(step, &message).await
    }

    async fn poll_event(&mut self) -> Option<StreamItem> {
        if self.step_number >= self.config.max_steps {
            self.memory.add_step(FinalAnswerStep {
                output: Value::String("Maximum steps reached".into()),
            });
            return Some(Err(AgentError::max_steps(
                self.step_number,
                self.config.max_steps,
            )));
        }

        if self.interrupt_flag.load(Ordering::SeqCst) {
            return Some(Err(AgentError::Interrupted));
        }

        self.step_number += 1;
        let mut step = ActionStep {
            step_number: self.step_number,
            timing: Timing::start_now(),
            ..Default::default()
        };

        let result = self.run_step(&mut step).await;
        step.timing.complete();

        match result {
            Ok(StepResult::Continue) => {
                let step_clone = step.clone();
                self.memory.add_step(step);
                Some(Ok(StreamEvent::StepComplete {
                    step: self.step_number,
                    action_step: Box::new(step_clone),
                }))
            }
            Ok(StepResult::FinalAnswer(answer)) => {
                step.is_final_answer = true;
                step.action_output = Some(answer.clone());
                self.memory.add_step(step);
                self.memory.add_step(FinalAnswerStep {
                    output: answer.clone(),
                });
                info!("Agent completed successfully");
                Some(Ok(StreamEvent::FinalAnswer { answer }))
            }
            Err(e) => {
                step.error = Some(e.to_string());
                self.memory.add_step(step);
                warn!(step = self.step_number, error = %e, "Step failed");
                Some(Ok(StreamEvent::Error(e.to_string())))
            }
        }
    }

    fn check_answer(&self, answer: &Value) -> Result<()> {
        if self.final_answer_checks.is_empty() {
            return Ok(());
        }
        self.final_answer_checks.validate(answer, &self.memory)
    }

    fn record_telemetry(&mut self, step: &ActionStep) {
        if let Some(ref tool_calls) = step.tool_calls {
            for tc in tool_calls {
                self.telemetry.record_tool_call(&tc.name);
            }
        }
        self.telemetry
            .record_step(self.step_number, step.token_usage.as_ref());
    }
}

impl Agent {
    async fn call_model(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
        step: &mut ActionStep,
    ) -> Result<ChatMessage> {
        if self.model.supports_streaming() {
            let mut stream = self.model.generate_stream(messages, options).await?;
            let mut deltas = Vec::new();

            while let Some(result) = stream.next().await {
                match result {
                    Ok(delta) => {
                        if let Some(usage) = &delta.token_usage {
                            step.token_usage = Some(*usage);
                        }
                        deltas.push(delta);
                    }
                    Err(e) => return Err(e),
                }
            }

            let message = crate::message::aggregate_stream_deltas(&deltas);
            step.model_output_message = Some(message.clone());
            step.model_output = message.text_content();
            Ok(message)
        } else {
            let response = self.model.generate(messages, options).await?;
            step.model_output_message = Some(response.message.clone());
            step.token_usage = response.token_usage;
            step.model_output = response.message.text_content();
            Ok(response.message)
        }
    }

    async fn handle_response(
        &self,
        step: &mut ActionStep,
        message: &ChatMessage,
    ) -> Result<StepResult> {
        let tool_calls = self.get_tool_calls(step, message)?;

        let Some(tool_calls) = tool_calls else {
            return Ok(StepResult::Continue);
        };

        let mut observations = Vec::with_capacity(tool_calls.len());
        let mut final_answer = None;

        for tc in &tool_calls {
            let tool_name = tc.name();
            step.tool_calls
                .get_or_insert_with(Vec::new)
                .push(ToolCall::new(&tc.id, tool_name, tc.arguments().clone()));

            if tool_name == "final_answer" {
                if let Ok(args) = tc.parse_arguments::<crate::tools::FinalAnswerArgs>() {
                    final_answer = Some(args.answer);
                    step.is_final_answer = true;
                }
                continue;
            }

            let observation = self.call_tool(tool_name, tc.arguments().clone()).await;
            if let Err(ref e) = observation {
                step.error = Some(e.clone());
            }
            observations.push(observation.unwrap_or_else(|e| e));
        }

        if !observations.is_empty() {
            step.observations = Some(observations.join("\n"));
        }

        match final_answer {
            Some(answer) => {
                step.action_output = Some(answer.clone());
                Ok(StepResult::FinalAnswer(answer))
            }
            None => Ok(StepResult::Continue),
        }
    }

    fn get_tool_calls(
        &self,
        step: &ActionStep,
        message: &ChatMessage,
    ) -> Result<Option<Vec<ChatMessageToolCall>>> {
        if let Some(tc) = &message.tool_calls {
            return Ok(Some(tc.clone()));
        }

        if let Some(text) = &step.model_output {
            if let Some(parsed) = Self::parse_text_tool_call(text) {
                debug!(step = step.step_number, tool = %parsed.name(), "Parsed tool call from text");
                return Ok(Some(vec![parsed]));
            }
            debug!(step = step.step_number, output = %text, "Model returned text without tool call");
        } else {
            debug!(step = step.step_number, "Model returned empty response");
        }

        Ok(None)
    }

    async fn call_tool(&self, name: &str, args: Value) -> std::result::Result<String, String> {
        match self.tools.call(name, args).await {
            Ok(result) => Ok(format!("Tool '{name}' returned: {result}")),
            Err(e) => Err(format!("Tool '{name}' failed: {e}")),
        }
    }

    fn parse_text_tool_call(text: &str) -> Option<ChatMessageToolCall> {
        let json_str = text.find('{').map(|start| {
            let mut depth = 0;
            let mut end = start;
            for (i, c) in text[start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            &text[start..end]
        })?;

        let json: Value = serde_json::from_str(json_str).ok()?;
        let name = json.get("name")?.as_str()?;
        let arguments = json
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(serde_json::Map::default()));

        Some(ChatMessageToolCall::new(
            format!("text_parsed_{}", uuid::Uuid::new_v4().simple()),
            name.to_string(),
            arguments,
        ))
    }
}

impl Agent {
    fn render_system_prompt(&self) -> String {
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
                self.fallback_prompt()
            })
    }

    fn fallback_prompt(&self) -> String {
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

    fn format_task_prompt(&self, name: &str, task: &str) -> String {
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

    fn format_report(&self, name: &str, final_answer: &str) -> String {
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

    fn append_summary(&self, answer: &mut String) {
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

#[async_trait::async_trait]
impl ManagedAgent for Agent {
    fn name(&self) -> &str {
        self.config.name.as_deref().unwrap_or("agent")
    }

    fn description(&self) -> &str {
        self.config
            .description
            .as_deref()
            .unwrap_or("A helpful AI agent")
    }

    async fn call(
        &self,
        task: &str,
        _additional_args: Option<HashMap<String, Value>>,
    ) -> Result<String> {
        let agent_name = ManagedAgent::name(self).to_string();
        let agent_desc = ManagedAgent::description(self).to_string();
        let _full_task = self.format_task_prompt(&agent_name, task);

        let report = format!(
            "### 1. Task outcome (short version):\n\
             Received task: {task}\n\n\
             ### 2. Task outcome (extremely detailed version):\n\
             The managed agent '{agent_name}' received the task. The task has been delegated.\n\n\
             ### 3. Additional context:\n\
             Agent description: {agent_desc}"
        );

        Ok(self.format_report(&agent_name, &report))
    }

    fn provide_run_summary(&self) -> bool {
        self.config.provide_run_summary.unwrap_or(false)
    }
}
