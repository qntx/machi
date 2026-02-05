//! Agent memory and conversation history.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Message;
use crate::usage::Usage;

/// Timing information for an operation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Timing {
    /// Start time as Unix timestamp (seconds).
    pub start_time: f64,
    /// End time as Unix timestamp (seconds).
    pub end_time: Option<f64>,
    /// Duration in seconds.
    pub duration_secs: Option<f64>,
}

impl Timing {
    /// Create a new timing starting now.
    #[must_use]
    pub fn start_now() -> Self {
        Self {
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |d| d.as_secs_f64()),
            end_time: None,
            duration_secs: None,
        }
    }

    /// Mark the timing as complete.
    pub fn complete(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        self.end_time = Some(now);
        self.duration_secs = Some(now - self.start_time);
    }
}

/// A step in the agent's memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryStep {
    /// Task assignment step.
    Task(TaskStep),
    /// Planning step.
    Planning(PlanningStep),
    /// Action/tool execution step.
    Action(ActionStep),
    /// Final answer step.
    FinalAnswer(FinalAnswerStep),
}

impl MemoryStep {
    /// Get the step type as a string.
    #[must_use]
    pub const fn step_type(&self) -> &'static str {
        match self {
            Self::Task(_) => "task",
            Self::Planning(_) => "planning",
            Self::Action(_) => "action",
            Self::FinalAnswer(_) => "final_answer",
        }
    }
}

/// Task assignment step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    /// The task description.
    pub task: String,
    /// Additional context.
    pub context: Option<Value>,
}

impl TaskStep {
    /// Create a new task step.
    #[must_use]
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            context: None,
        }
    }

    /// Create a task step with context.
    #[must_use]
    pub fn with_context(task: impl Into<String>, context: Value) -> Self {
        Self {
            task: task.into(),
            context: Some(context),
        }
    }
}

/// Planning step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningStep {
    /// The plan content.
    pub plan: String,
    /// Token usage for this step.
    pub usage: Option<Usage>,
}

impl PlanningStep {
    /// Create a new planning step.
    #[must_use]
    pub fn new(plan: impl Into<String>) -> Self {
        Self {
            plan: plan.into(),
            usage: None,
        }
    }
}

/// Action/tool execution step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionStep {
    /// The thought/reasoning before action.
    pub thought: Option<String>,
    /// Tool calls made.
    pub tool_calls: Vec<ToolCallInfo>,
    /// Token usage for this step.
    pub usage: Option<Usage>,
    /// Step timing.
    pub timing: Option<Timing>,
}

/// Information about a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    /// Tool call ID.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool arguments.
    pub arguments: Value,
    /// Tool output.
    pub output: Option<Value>,
    /// Error message if failed.
    pub error: Option<String>,
}

impl ActionStep {
    /// Create a new action step.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            thought: None,
            tool_calls: Vec::new(),
            usage: None,
            timing: None,
        }
    }

    /// Set the thought.
    #[must_use]
    pub fn with_thought(mut self, thought: impl Into<String>) -> Self {
        self.thought = Some(thought.into());
        self
    }

    /// Add a tool call.
    pub fn add_tool_call(&mut self, info: ToolCallInfo) {
        self.tool_calls.push(info);
    }
}

impl Default for ActionStep {
    fn default() -> Self {
        Self::new()
    }
}

/// Final answer step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalAnswerStep {
    /// The final answer.
    pub output: Value,
}

impl FinalAnswerStep {
    /// Create a new final answer step.
    #[must_use]
    pub const fn new(output: Value) -> Self {
        Self { output }
    }
}

/// Agent memory storing conversation history and steps.
#[derive(Debug, Clone, Default)]
pub struct AgentMemory {
    /// System prompt.
    system_prompt: Option<String>,
    /// Memory steps.
    steps: Vec<MemoryStep>,
    /// Raw messages for LLM context.
    messages: Vec<Message>,
}

impl AgentMemory {
    /// Create a new empty memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the system prompt.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = Some(prompt.to_owned());
        // Add as first message
        if self.messages.is_empty() || !self.messages[0].role.is_system() {
            self.messages.insert(0, Message::system(prompt));
        } else {
            self.messages[0] = Message::system(prompt);
        }
    }

    /// Get the system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Add a memory step.
    pub fn add_step(&mut self, step: MemoryStep) {
        self.steps.push(step);
    }

    /// Add a message to the conversation.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get all steps.
    #[must_use]
    pub fn steps(&self) -> &[MemoryStep] {
        &self.steps
    }

    /// Get all messages for LLM context.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get the number of steps.
    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Calculate total token usage.
    #[must_use]
    pub fn total_usage(&self) -> Usage {
        let mut total = Usage::zero();
        for step in &self.steps {
            if let MemoryStep::Action(action) = step {
                if let Some(usage) = action.usage {
                    total += usage;
                }
            } else if let MemoryStep::Planning(planning) = step
                && let Some(usage) = planning.usage
            {
                total += usage;
            }
        }
        total
    }

    /// Reset the memory.
    pub fn reset(&mut self) {
        self.steps.clear();
        self.messages.clear();
        if let Some(prompt) = &self.system_prompt {
            self.messages.push(Message::system(prompt));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing() {
        let mut timing = Timing::start_now();
        assert!(timing.end_time.is_none());
        timing.complete();
        assert!(timing.end_time.is_some());
        assert!(timing.duration_secs.is_some());
    }

    #[test]
    fn test_memory_step() {
        let task = MemoryStep::Task(TaskStep::new("Test task"));
        assert_eq!(task.step_type(), "task");
    }

    #[test]
    fn test_agent_memory() {
        let mut memory = AgentMemory::new();
        memory.set_system_prompt("You are helpful.");
        memory.add_step(MemoryStep::Task(TaskStep::new("Do something")));

        assert_eq!(memory.step_count(), 1);
        assert_eq!(memory.system_prompt(), Some("You are helpful."));
    }

    #[test]
    fn test_memory_reset() {
        let mut memory = AgentMemory::new();
        memory.set_system_prompt("System prompt");
        memory.add_step(MemoryStep::Task(TaskStep::new("Task")));
        memory.reset();

        assert_eq!(memory.step_count(), 0);
        assert_eq!(memory.messages().len(), 1); // System prompt preserved
    }
}
