//! AI Agent for executing tasks with tools.
//!
//! This module provides a lightweight, ergonomic agent that uses LLM function
//! calling to accomplish tasks through tool invocations.
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! let provider = OpenAI::from_env()?;
//! let mut agent = Agent::builder()
//!     .provider(provider)
//!     .tool(Box::new(MyTool))
//!     .build()?;
//!
//! let result = agent.run("What is 2 + 2?").await?;
//! ```

mod builder;
mod config;
mod executor;
mod memory;
mod result;

pub use builder::AgentBuilder;
pub use config::AgentConfig;
pub use memory::{ActionStep, AgentMemory, MemoryStep, PlanningStep, TaskStep, Timing};
pub use result::{RunResult, RunState};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::Value;
use tracing::{info, instrument, warn};

use crate::chat::ChatProvider;
use crate::error::Result;
use crate::tool::{BoxedConfirmationHandler, ToolBox};
use crate::usage::Usage;

/// AI agent that uses LLM function calling to execute tasks with tools.
///
/// The agent follows a ReAct-style loop:
/// 1. Receive a task
/// 2. Think and decide which tool to call
/// 3. Execute the tool and observe the result
/// 4. Repeat until task is complete or max steps reached
pub struct Agent {
    /// The LLM provider.
    provider: Arc<dyn ChatProvider>,
    /// Available tools.
    tools: ToolBox,
    /// Agent configuration.
    config: AgentConfig,
    /// Agent memory (conversation history).
    memory: AgentMemory,
    /// System prompt.
    system_prompt: String,
    /// Interrupt flag for stopping execution.
    interrupt_flag: Arc<AtomicBool>,
    /// Current step number.
    step_number: usize,
    /// Context variables.
    state: HashMap<String, Value>,
    /// Optional confirmation handler for tools.
    confirmation_handler: Option<BoxedConfirmationHandler>,
    /// Run start time.
    run_start: std::time::Instant,
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

    /// Run the agent with a task.
    #[instrument(skip(self, task), fields(max_steps = self.config.max_steps))]
    pub async fn run(&mut self, task: impl Into<String>) -> Result<Value> {
        let task = task.into();
        self.prepare_run(&task);
        info!("Starting agent run");

        let timing = Timing::start_now();
        let result = self.execute_loop().await;
        let mut final_timing = timing;
        final_timing.complete();

        self.complete_run(result, final_timing)
            .into_result(self.config.max_steps)
    }

    /// Run the agent and return detailed [`RunResult`] with metrics.
    #[instrument(skip(self, task), fields(max_steps = self.config.max_steps))]
    pub async fn run_detailed(&mut self, task: impl Into<String>) -> RunResult {
        let task = task.into();
        self.prepare_run(&task);
        info!("Starting agent run");

        let timing = Timing::start_now();
        let result = self.execute_loop().await;
        let mut final_timing = timing;
        final_timing.complete();

        self.complete_run(result, final_timing)
    }

    /// Get the agent's memory.
    #[inline]
    pub const fn memory(&self) -> &AgentMemory {
        &self.memory
    }

    /// Get the current step number.
    #[inline]
    pub const fn current_step(&self) -> usize {
        self.step_number
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
        self.run_start = std::time::Instant::now();
    }

    /// Get total token usage.
    #[must_use]
    pub fn total_usage(&self) -> Usage {
        self.memory.total_usage()
    }
}

impl Agent {
    fn prepare_run(&mut self, task: &str) {
        self.memory.reset();
        self.step_number = 0;
        self.interrupt_flag.store(false, Ordering::SeqCst);
        self.run_start = std::time::Instant::now();

        // Add system prompt
        self.memory.set_system_prompt(&self.system_prompt);

        // Add task
        let task_step = TaskStep::new(task);
        self.memory.add_step(MemoryStep::Task(task_step));
    }

    fn complete_run(&self, result: Result<Value>, timing: Timing) -> RunResult {
        let token_usage = self.memory.total_usage();
        let steps_taken = self.step_number;

        match result {
            Ok(answer) => {
                info!("Agent completed successfully");
                RunResult {
                    output: Some(answer),
                    state: RunState::Success,
                    token_usage: Some(token_usage),
                    steps_taken,
                    timing,
                    error: None,
                }
            }
            Err(crate::error::Error::MaxSteps { .. }) => RunResult {
                output: None,
                state: RunState::MaxStepsReached,
                token_usage: Some(token_usage),
                steps_taken,
                timing,
                error: Some("Maximum steps reached".to_owned()),
            },
            Err(crate::error::Error::Interrupted) => RunResult {
                output: None,
                state: RunState::Interrupted,
                token_usage: Some(token_usage),
                steps_taken,
                timing,
                error: Some("Agent was interrupted".to_owned()),
            },
            Err(e) => {
                warn!(error = %e, "Agent run failed");
                RunResult {
                    output: None,
                    state: RunState::Failed,
                    token_usage: Some(token_usage),
                    steps_taken,
                    timing,
                    error: Some(e.to_string()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_builder() {
        // Basic builder test - actual creation requires a provider
        let builder = Agent::builder();
        assert!(builder.config.max_steps > 0);
    }
}
