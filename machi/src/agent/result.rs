//! Run result types and step resolution.
//!
//! This module defines the core state-machine types that drive the agent's
//! reasoning loop:
//!
//! - [`NextStep`]: Determines what happens after each LLM turn.
//! - [`RunConfig`]: Configures a single agent run (hooks, session, limits).
//! - [`RunResult`]: The final outcome of a completed agent run.
//! - [`StepInfo`]: Metadata about a single reasoning step for observability.

use std::fmt;

use serde_json::Value;

use crate::callback::SharedRunHooks;
use crate::chat::ChatResponse;
use crate::memory::SharedSession;
use crate::message::{Content, ContentPart, ImageMime, Message, Role, ToolCall};
use crate::tool::SharedConfirmationHandler;
use crate::usage::Usage;

/// Outcome of processing an LLM response within the agent loop.
///
/// The Runner evaluates the model's output and classifies it into one of these
/// variants, each of which drives a different continuation path.
#[derive(Debug)]
pub enum NextStep {
    /// The LLM produced a final text answer — the run is complete.
    FinalOutput {
        /// The final output value from the agent.
        output: Value,
    },

    /// The LLM requested one or more tool calls (including managed agents).
    ///
    /// The Runner executes these concurrently via [`futures::stream::buffered`],
    /// respecting [`RunConfig::max_tool_concurrency`], then appends results as
    /// tool messages and loops back for another LLM turn.
    ToolCalls {
        /// Tool calls extracted from the LLM response.
        calls: Vec<ToolCallRequest>,
    },

    /// One or more tool calls require human approval before execution.
    ///
    /// The Runner will invoke the configured [`ConfirmationHandler`](crate::tool::ConfirmationHandler)
    /// and either proceed or abort based on the user's decision.
    NeedsApproval {
        /// Tool calls pending human approval.
        pending_approval: Vec<ToolCallRequest>,
        /// Tool calls already approved (auto policy).
        approved: Vec<ToolCallRequest>,
    },

    /// The maximum step count was reached without a final answer.
    MaxStepsExceeded,
}

/// A parsed tool call request extracted from the LLM response.
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    /// The tool call ID from the model (used to correlate results).
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The arguments as a JSON value.
    pub arguments: Value,
}

impl From<&ToolCall> for ToolCallRequest {
    fn from(tc: &ToolCall) -> Self {
        let arguments: Value = serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
        Self {
            id: tc.id.clone(),
            name: tc.function.name.clone(),
            arguments,
        }
    }
}

/// Configuration for a single agent run.
///
/// Passed to [`Runner::run`](super::Runner::run) to control execution behavior
/// without modifying the [`Agent`](super::Agent) definition itself.
#[derive(Clone, Default)]
pub struct RunConfig {
    /// Global run-level lifecycle hooks.
    pub hooks: Option<SharedRunHooks>,

    /// Session for message persistence across runs.
    pub session: Option<SharedSession>,

    /// Maximum number of reasoning steps (overrides `Agent::max_steps`).
    pub max_steps: Option<usize>,

    /// Maximum number of concurrent tool executions.
    ///
    /// Defaults to unlimited (all tool calls run in parallel).
    pub max_tool_concurrency: Option<usize>,

    /// Handler for tool execution confirmation requests.
    ///
    /// Required when any tool has [`ToolExecutionPolicy::RequireConfirmation`](crate::tool::ToolExecutionPolicy::RequireConfirmation).
    /// If absent and a tool requires confirmation, the runner returns an error.
    pub confirmation_handler: Option<SharedConfirmationHandler>,
}

impl fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunConfig")
            .field("hooks", &self.hooks.is_some())
            .field("session", &self.session.is_some())
            .field("max_steps", &self.max_steps)
            .field("max_tool_concurrency", &self.max_tool_concurrency)
            .field("confirmation_handler", &self.confirmation_handler.is_some())
            .finish()
    }
}

impl RunConfig {
    /// Create a new default run configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set global run-level hooks.
    #[must_use]
    pub fn hooks(mut self, hooks: SharedRunHooks) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set a session for message persistence.
    #[must_use]
    pub fn session(mut self, session: SharedSession) -> Self {
        self.session = Some(session);
        self
    }

    /// Override the agent's max_steps for this run.
    #[must_use]
    pub const fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Set the maximum number of concurrent tool executions.
    #[must_use]
    pub const fn max_tool_concurrency(mut self, max: usize) -> Self {
        self.max_tool_concurrency = Some(max);
        self
    }

    /// Set the confirmation handler for tools requiring approval.
    #[must_use]
    pub fn confirmation_handler(mut self, handler: SharedConfirmationHandler) -> Self {
        self.confirmation_handler = Some(handler);
        self
    }
}

/// The final result of a completed agent run.
#[derive(Debug)]
pub struct RunResult {
    /// The final output produced by the agent.
    pub output: Value,

    /// Cumulative token usage across all LLM calls in this run.
    pub usage: Usage,

    /// Number of reasoning steps taken.
    pub steps: usize,

    /// Detailed information about each step (for observability).
    pub step_history: Vec<StepInfo>,

    /// The name of the agent that produced the final output.
    ///
    /// In a multi-agent scenario this may differ from the starting agent
    /// if managed agents were involved (though each managed agent runs
    /// its own sub-run).
    pub agent_name: String,
}

impl RunResult {
    /// Get the final output as a string, if it is one.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.output.as_str()
    }

    /// Deserialize the output into a concrete Rust type.
    ///
    /// This is the companion to [`Agent::output_type`](super::Agent::output_type)
    /// and [`Agent::output_schema`](super::Agent::output_schema). When the agent
    /// produces structured JSON output, this method deserializes it into `T`.
    ///
    /// # Errors
    ///
    /// Returns [`serde_json::Error`] if the output cannot be deserialized into `T`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let result = agent.run("Tell me about France", config).await?;
    /// let country: Country = result.parse()?;
    /// ```
    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> serde_json::Result<T> {
        serde_json::from_value(self.output.clone())
    }
}

/// Metadata about a single reasoning step in the agent loop.
///
/// Captures both the LLM interaction and any tool calls that were executed,
/// providing a complete audit trail for debugging and observability.
#[derive(Debug, Clone)]
pub struct StepInfo {
    /// Step number (1-indexed).
    pub step: usize,

    /// The LLM response for this step.
    pub response: ChatResponse,

    /// Tool calls executed during this step (empty if the LLM produced text only).
    pub tool_calls: Vec<ToolCallRecord>,
}

/// Record of a single tool call execution within a step.
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    /// The tool call ID.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The arguments passed to the tool.
    pub arguments: Value,
    /// The result (serialized output or error message).
    pub result: String,
    /// Whether the call succeeded.
    pub success: bool,
}

/// An observable event emitted during a streamed agent run.
///
/// `RunEvent` provides fine-grained visibility into the agent's execution,
/// enabling real-time UIs, progress indicators, and streaming text display.
/// Events are yielded in chronological order through the stream returned by
/// [`Runner::run_streamed`](super::Runner::run_streamed).
///
/// # Event Flow
///
/// A typical successful run emits events in this order:
///
/// ```text
/// RunStarted → StepStarted → TextDelta* → StepCompleted → RunCompleted
///                           → ToolCallStarted* → ToolCallCompleted* ↩ (loop)
/// ```
///
/// # Error Handling
///
/// Errors are delivered as `Err(...)` through the `Result<RunEvent>` stream
/// rather than as a dedicated event variant, following Rust's idiomatic
/// `try_stream` pattern.
#[derive(Debug)]
pub enum RunEvent {
    /// The agent run has started.
    RunStarted {
        /// Name of the agent being executed.
        agent_name: String,
    },

    /// A new reasoning step has begun.
    StepStarted {
        /// Step number (1-indexed).
        step: usize,
    },

    /// Incremental text output from the LLM (for real-time display).
    TextDelta(String),

    /// Incremental reasoning/thinking content from the LLM (o1/o3 models).
    ReasoningDelta(String),

    /// Audio data delta from the LLM (for audio-capable models).
    AudioDelta {
        /// Base64-encoded audio data.
        data: String,
        /// Audio transcript (if available).
        transcript: Option<String>,
    },

    /// A tool call has been identified from the LLM stream.
    ToolCallStarted {
        /// The tool call ID.
        id: String,
        /// The tool name.
        name: String,
    },

    /// A tool call execution has completed.
    ToolCallCompleted {
        /// The completed tool call record.
        record: ToolCallRecord,
    },

    /// A reasoning step has completed.
    StepCompleted {
        /// Metadata about the completed step.
        step_info: Box<StepInfo>,
    },

    /// The agent run completed successfully with a final result.
    RunCompleted {
        /// The final run result.
        result: Box<RunResult>,
    },
}

/// Flexible input for an agent run, supporting text and multimodal content.
///
/// `UserInput` abstracts over plain text and multimodal payloads so that
/// [`Runner::run`](super::Runner::run) and [`Agent::run`](super::Agent::run)
/// accept both simple strings and rich content (images, audio) through
/// a single, ergonomic API.
///
/// # Examples
///
/// ```rust,ignore
/// use machi::agent::{Agent, RunConfig, UserInput};
/// use machi::message::ContentPart;
///
/// // Simple text — most common case.
/// agent.run("Hello!", RunConfig::default()).await?;
///
/// // Text + image via convenience constructor.
/// agent.run(
///     UserInput::with_image("Describe this", "https://example.com/img.png"),
///     RunConfig::default(),
/// ).await?;
///
/// // Full multimodal via Vec<ContentPart>.
/// agent.run(
///     vec![
///         ContentPart::text("What's in these images?"),
///         ContentPart::image_url("https://example.com/a.png"),
///         ContentPart::image_url("https://example.com/b.png"),
///     ],
///     RunConfig::default(),
/// ).await?;
/// ```
#[derive(Debug, Clone)]
pub enum UserInput {
    /// Plain text input (the most common case).
    Text(String),

    /// Multimodal content parts (text, images, audio, etc.).
    Parts(Vec<ContentPart>),
}

impl UserInput {
    /// Creates a text-only input.
    #[inline]
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Creates a multimodal input from raw content parts.
    #[inline]
    #[must_use]
    pub const fn parts(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }

    /// Convenience: text + a single image URL.
    #[must_use]
    pub fn with_image(text: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self::Parts(vec![
            ContentPart::text(text),
            ContentPart::image_url(image_url),
        ])
    }

    /// Convenience: text + a single image from raw bytes with explicit MIME.
    #[must_use]
    pub fn with_image_bytes(text: impl Into<String>, data: &[u8], mime: ImageMime) -> Self {
        Self::Parts(vec![
            ContentPart::text(text),
            ContentPart::image_bytes(data, mime),
        ])
    }

    /// Convenience: text + a single image from raw bytes with auto-detected MIME.
    #[must_use]
    pub fn with_image_auto(text: impl Into<String>, data: &[u8]) -> Self {
        Self::Parts(vec![
            ContentPart::text(text),
            ContentPart::image_bytes_auto(data),
        ])
    }

    /// Converts this input into a user-role [`Message`].
    #[must_use]
    pub fn into_message(self) -> Message {
        match self {
            Self::Text(text) => Message::user(text),
            Self::Parts(parts) => Message::new(Role::User, Content::Parts(parts)),
        }
    }

    /// Returns `true` if this input contains any images.
    #[must_use]
    pub fn has_images(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(ContentPart::is_image),
        }
    }

    /// Returns `true` if this input contains any audio.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(ContentPart::is_audio),
        }
    }

    /// Returns `true` if this is a multimodal input (contains non-text parts).
    #[must_use]
    pub fn is_multimodal(&self) -> bool {
        self.has_images() || self.has_audio()
    }
}

impl From<&str> for UserInput {
    #[inline]
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<String> for UserInput {
    #[inline]
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<Vec<ContentPart>> for UserInput {
    #[inline]
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
}
