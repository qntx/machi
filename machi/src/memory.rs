//! Memory system for tracking agent steps and state.
//!
//! This module provides the memory infrastructure for agents, allowing them
//! to track their execution history, tool calls, and observations.

use crate::message::{ChatMessage, MessageRole};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;

pub use crate::providers::common::TokenUsage;

/// Timing information for a step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Timing {
    /// Start time of the step.
    pub start_time: DateTime<Utc>,
    /// End time of the step (if completed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<DateTime<Utc>>,
}

impl Timing {
    /// Create a new timing starting now.
    #[must_use]
    pub fn start_now() -> Self {
        Self {
            start_time: Utc::now(),
            end_time: None,
        }
    }

    /// Mark the timing as complete.
    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// Get the duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> Option<f64> {
        self.end_time
            .map(|end| (end - self.start_time).num_milliseconds() as f64 / 1000.0)
    }
}

impl Default for Timing {
    fn default() -> Self {
        Self::start_now()
    }
}

/// A tool call made during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for the tool call.
    pub id: String,
    /// Name of the tool.
    pub name: String,
    /// Arguments passed to the tool.
    pub arguments: Value,
}

impl ToolCall {
    /// Create a new tool call.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// Base trait for memory steps.
pub trait MemoryStep: Send + Sync + std::fmt::Debug {
    /// Convert the step to messages for the model.
    fn to_messages(&self, summary_mode: bool) -> Vec<ChatMessage>;

    /// Get the step as a serializable value.
    fn to_value(&self) -> Value;

    /// Downcast to Any for type checking.
    fn as_any(&self) -> &dyn Any;
}

/// System prompt step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptStep {
    /// The system prompt.
    pub system_prompt: String,
}

impl MemoryStep for SystemPromptStep {
    fn to_messages(&self, summary_mode: bool) -> Vec<ChatMessage> {
        if summary_mode {
            return vec![];
        }
        vec![ChatMessage::system(&self.system_prompt)]
    }

    fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Task step representing a new task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    /// The task description.
    pub task: String,
    /// Optional images associated with the task.
    #[serde(skip)]
    pub task_images: Option<Vec<Vec<u8>>>,
}

impl MemoryStep for TaskStep {
    fn to_messages(&self, _summary_mode: bool) -> Vec<ChatMessage> {
        vec![ChatMessage::user(format!("New task:\n{}", self.task))]
    }

    fn to_value(&self) -> Value {
        serde_json::json!({
            "task": self.task,
            "has_images": self.task_images.as_ref().is_some_and(|i| !i.is_empty())
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Action step representing an agent action.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionStep {
    /// Step number.
    pub step_number: usize,
    /// Timing information.
    pub timing: Timing,
    /// Input messages to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_input_messages: Option<Vec<ChatMessage>>,
    /// Tool calls made in this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Error that occurred (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Model output message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_output_message: Option<ChatMessage>,
    /// Raw model output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_output: Option<String>,
    /// Code action (for code agents).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_action: Option<String>,
    /// Observations from tool execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observations: Option<String>,
    /// Output of the action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_output: Option<Value>,
    /// Token usage for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
    /// Whether this is the final answer step.
    #[serde(default)]
    pub is_final_answer: bool,
}

impl MemoryStep for ActionStep {
    fn to_messages(&self, summary_mode: bool) -> Vec<ChatMessage> {
        let mut messages = vec![];

        // Add model output
        if let Some(output) = &self.model_output
            && !summary_mode {
                messages.push(ChatMessage::assistant(output.trim()));
            }

        // Add tool calls
        if let Some(tool_calls) = &self.tool_calls {
            let calls_str = serde_json::to_string(tool_calls).unwrap_or_default();
            messages.push(ChatMessage {
                role: MessageRole::ToolCall,
                content: Some(vec![crate::message::MessageContent::text(format!(
                    "Calling tools:\n{calls_str}"
                ))]),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Add observations
        if let Some(obs) = &self.observations {
            messages.push(ChatMessage {
                role: MessageRole::ToolResponse,
                content: Some(vec![crate::message::MessageContent::text(format!(
                    "Observation:\n{obs}"
                ))]),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Add error
        if let Some(err) = &self.error {
            let error_msg =
                format!("Error:\n{err}\nNow let's retry: take care not to repeat previous errors!");
            messages.push(ChatMessage {
                role: MessageRole::ToolResponse,
                content: Some(vec![crate::message::MessageContent::text(error_msg)]),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        messages
    }

    fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Planning step for agent planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningStep {
    /// Input messages to the model.
    pub model_input_messages: Vec<ChatMessage>,
    /// Model output message.
    pub model_output_message: ChatMessage,
    /// The plan text.
    pub plan: String,
    /// Timing information.
    pub timing: Timing,
    /// Token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
}

impl MemoryStep for PlanningStep {
    fn to_messages(&self, summary_mode: bool) -> Vec<ChatMessage> {
        if summary_mode {
            return vec![];
        }
        vec![
            ChatMessage::assistant(self.plan.trim()),
            ChatMessage::user("Now proceed and carry out this plan."),
        ]
    }

    fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Final answer step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalAnswerStep {
    /// The final output.
    pub output: Value,
}

impl MemoryStep for FinalAnswerStep {
    fn to_messages(&self, _summary_mode: bool) -> Vec<ChatMessage> {
        vec![]
    }

    fn to_value(&self) -> Value {
        serde_json::json!({ "output": self.output })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Agent memory containing system prompt and all steps.
#[derive(Debug)]
pub struct AgentMemory {
    /// System prompt step.
    pub system_prompt: SystemPromptStep,
    /// List of steps taken by the agent.
    pub steps: Vec<Box<dyn MemoryStep>>,
}

impl AgentMemory {
    /// Create a new agent memory with the given system prompt.
    #[must_use]
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: SystemPromptStep {
                system_prompt: system_prompt.into(),
            },
            steps: Vec::new(),
        }
    }

    /// Reset the memory, clearing all steps.
    pub fn reset(&mut self) {
        self.steps.clear();
    }

    /// Add a step to memory.
    pub fn add_step<S: MemoryStep + 'static>(&mut self, step: S) {
        self.steps.push(Box::new(step));
    }

    /// Convert memory to messages for the model.
    #[must_use]
    pub fn to_messages(&self, summary_mode: bool) -> Vec<ChatMessage> {
        let mut messages = self.system_prompt.to_messages(summary_mode);
        for step in &self.steps {
            messages.extend(step.to_messages(summary_mode));
        }
        messages
    }

    /// Get all steps as values.
    #[must_use]
    pub fn get_steps(&self) -> Vec<Value> {
        self.steps.iter().map(|s| s.to_value()).collect()
    }

    /// Get total token usage.
    #[must_use]
    pub fn total_token_usage(&self) -> TokenUsage {
        let mut total = TokenUsage::default();
        for step in &self.steps {
            if let Some(action) = step.as_any().downcast_ref::<ActionStep>() {
                if let Some(usage) = action.token_usage {
                    total += usage;
                }
            } else if let Some(planning) = step.as_any().downcast_ref::<PlanningStep>()
                && let Some(usage) = planning.token_usage {
                    total += usage;
                }
        }
        total
    }

    /// Get all code actions concatenated.
    #[must_use]
    pub fn return_full_code(&self) -> String {
        self.steps
            .iter()
            .filter_map(|s| {
                s.as_any()
                    .downcast_ref::<ActionStep>()
                    .and_then(|a| a.code_action.clone())
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self::new("")
    }
}
