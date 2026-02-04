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
mod executor;
mod options;
mod prompts_render;
mod result;
mod streaming;
mod tool_processor;

pub use builder::AgentBuilder;
pub use checks::{FinalAnswerCheck, FinalAnswerChecks};
pub use config::AgentConfig;
pub use events::{AgentStream, RunState, StreamEvent, StreamItem};
pub use options::RunOptions;
pub use result::RunResult;

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use futures::Stream;
use serde_json::Value;
use tracing::{info, instrument, warn};

use crate::{
    callback::{CallbackContext, CallbackRegistry},
    error::{AgentError, Result},
    managed::{ManagedAgent, ManagedAgentRegistry},
    memory::{AgentMemory, FinalAnswerStep, TaskStep, Timing},
    multimodal::AgentImage,
    prompts::PromptRender,
    providers::common::Model,
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
    pub(crate) prompt_renderer: PromptRender,
    pub(crate) interrupt_flag: Arc<AtomicBool>,
    pub(crate) step_number: usize,
    pub(crate) state: HashMap<String, Value>,
    pub(crate) custom_instructions: Option<String>,
    pub(crate) final_answer_checks: FinalAnswerChecks,
    pub(crate) telemetry: Telemetry,
    pub(crate) callbacks: CallbackRegistry,
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
    ///
    /// This is the unified entry point for agent execution. Use [`RunOptions`]
    /// to configure images, context variables, and other settings.
    /// ```
    #[instrument(skip(self, options), fields(max_steps = self.config.max_steps))]
    pub async fn run(&mut self, options: impl Into<RunOptions>) -> Result<Value> {
        let opts = options.into();
        if opts.detailed {
            self.run_internal(opts)
                .await
                .into_result(self.config.max_steps)
        } else {
            self.run_internal(opts)
                .await
                .into_result(self.config.max_steps)
        }
    }

    /// Run the agent and return detailed [`RunResult`] with metrics.
    ///
    /// This always returns the full [`RunResult`] including token usage,
    /// timing, and step information.
    #[instrument(skip(self, options), fields(max_steps = self.config.max_steps))]
    pub async fn run_detailed(&mut self, options: impl Into<RunOptions>) -> RunResult {
        self.run_internal(options.into()).await
    }

    /// Internal run implementation.
    async fn run_internal(&mut self, options: RunOptions) -> RunResult {
        self.prepare_run(&options.task, options.images, options.context);
        info!("Starting agent run");

        let timing = Timing::start_now();
        let result = self.execute_loop().await;
        let mut final_timing = timing;
        final_timing.complete();

        self.complete_run(result, final_timing)
    }

    /// Stream execution events.
    ///
    /// This streams events at both step-level and token-level granularity,
    /// enabling real-time feedback during agent runs.
    #[instrument(skip(self, options), fields(max_steps = self.config.max_steps))]
    pub fn stream(
        &mut self,
        options: impl Into<RunOptions>,
    ) -> impl Stream<Item = StreamItem> + '_ {
        let opts = options.into();
        self.prepare_run(&opts.task, opts.images, opts.context);
        self.stream_execution()
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
    /// Create a callback context for the current state.
    fn create_callback_context(&self) -> CallbackContext {
        CallbackContext::new(self.step_number, self.config.max_steps)
            .with_agent_name(self.config.name.clone().unwrap_or_default())
    }

    fn prepare_run(
        &mut self,
        task: &str,
        images: Vec<AgentImage>,
        context: HashMap<String, Value>,
    ) {
        self.memory.reset();
        self.step_number = 0;
        self.interrupt_flag.store(false, Ordering::SeqCst);
        self.state = context;

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

    fn complete_run(&mut self, result: Result<Value>, timing: Timing) -> RunResult {
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
