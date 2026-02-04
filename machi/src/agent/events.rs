//! Streaming events and execution state types.

use std::pin::Pin;

use futures::Stream;
use serde_json::Value;

use crate::{error::AgentError, memory::ActionStep, providers::common::TokenUsage};

/// Events emitted during streaming agent execution.
///
/// These events allow real-time observation of agent progress, including
/// model output chunks, tool calls, and step completions.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
        result: Result<String, String>,
    },

    /// An action step has completed.
    StepComplete {
        /// Step number.
        step: usize,
        /// The completed action step data (boxed to reduce enum size).
        action_step: Box<ActionStep>,
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
pub type StreamItem = Result<StreamEvent, AgentError>;

/// Type alias for the boxed stream of events.
pub type AgentStream = Pin<Box<dyn Stream<Item = StreamItem> + Send>>;

/// Result of a streaming step execution.
#[derive(Debug)]
pub enum StepResult {
    /// Continue to next step.
    Continue,
    /// Agent produced a final answer.
    FinalAnswer(Value),
}

/// The state of an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
