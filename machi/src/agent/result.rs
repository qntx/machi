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

use crate::chat::ChatResponse;
use crate::context::SharedContextStrategy;
use crate::guardrail::{
    InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult,
};
use crate::hooks::SharedHooks;
use crate::memory::SharedSession;
use crate::message::{Content, ContentPart, ImageMime, Message, Role, ToolCall};
use crate::middleware::SharedMiddleware;
use crate::tool::SharedConfirmationHandler;
use crate::usage::Usage;

/// Outcome of processing an LLM response within the agent loop.
///
/// The Runner evaluates the model's output and classifies it into one of these
/// variants, each of which drives a different continuation path.
#[derive(Debug)]
#[non_exhaustive]
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
}

/// A parsed tool call request extracted from the LLM response.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct RunConfig {
    /// Global run-level lifecycle hooks.
    pub hooks: Option<SharedHooks>,

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

    /// Additional input guardrails applied at the run level.
    ///
    /// These are combined with the agent's own [`input_guardrails`](crate::agent::Agent::input_guardrails)
    /// and executed together during the first step.
    pub input_guardrails: Vec<InputGuardrail>,

    /// Additional output guardrails applied at the run level.
    ///
    /// These are combined with the agent's own [`output_guardrails`](crate::agent::Agent::output_guardrails)
    /// and executed together after the agent produces a final output.
    pub output_guardrails: Vec<OutputGuardrail>,

    /// Context compaction strategy applied before each LLM call.
    ///
    /// When set, the runner compacts the message list using this strategy
    /// before building the [`ChatRequest`](crate::chat::ChatRequest), keeping
    /// the conversation within the LLM's context window.
    ///
    /// Defaults to [`None`] (no compaction — messages accumulate unboundedly).
    pub context_strategy: Option<SharedContextStrategy>,

    /// Middleware pipeline for intercepting and modifying execution.
    ///
    /// Unlike [`Hooks`](crate::hooks::Hooks) which are purely observational,
    /// middleware can reject tool calls, transform requests, and short-circuit
    /// execution. Middleware runs in order for pre-execution events and in
    /// reverse for post-execution events (onion pattern).
    pub middleware: Vec<SharedMiddleware>,
}

impl fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunConfig")
            .field("hooks", &self.hooks.is_some())
            .field("session", &self.session.is_some())
            .field("max_steps", &self.max_steps)
            .field("max_tool_concurrency", &self.max_tool_concurrency)
            .field("confirmation_handler", &self.confirmation_handler.is_some())
            .field("input_guardrails", &self.input_guardrails.len())
            .field("output_guardrails", &self.output_guardrails.len())
            .field("context_strategy", &self.context_strategy.is_some())
            .field("middleware", &self.middleware.len())
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
    pub fn hooks(mut self, hooks: SharedHooks) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set a session for message persistence.
    #[must_use]
    pub fn session(mut self, session: SharedSession) -> Self {
        self.session = Some(session);
        self
    }

    /// Override the agent's `max_steps` for this run.
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

    /// Add an input guardrail at the run level.
    #[must_use]
    pub fn input_guardrail(mut self, guardrail: InputGuardrail) -> Self {
        self.input_guardrails.push(guardrail);
        self
    }

    /// Add an output guardrail at the run level.
    #[must_use]
    pub fn output_guardrail(mut self, guardrail: OutputGuardrail) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }

    /// Set a context compaction strategy.
    ///
    /// The strategy is applied before each LLM call to keep the message
    /// list within the model's context window.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use machi::agent::RunConfig;
    /// use machi::context::SlidingWindow;
    ///
    /// let config = RunConfig::new()
    ///     .context_strategy(Arc::new(SlidingWindow::new(50)));
    /// ```
    #[must_use]
    pub fn context_strategy(mut self, strategy: SharedContextStrategy) -> Self {
        self.context_strategy = Some(strategy);
        self
    }

    /// Add a middleware to the pipeline.
    ///
    /// Middleware runs in the order added for pre-execution events,
    /// and in reverse order for post-execution events (onion pattern).
    #[must_use]
    pub fn middleware(mut self, mw: SharedMiddleware) -> Self {
        self.middleware.push(mw);
        self
    }
}

/// The final result of a completed agent run.
#[derive(Debug)]
#[non_exhaustive]
#[allow(clippy::module_name_repetitions)]
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

    /// Results from input guardrail checks (empty if no guardrails configured).
    pub input_guardrail_results: Vec<InputGuardrailResult>,

    /// Results from output guardrail checks (empty if no guardrails configured).
    pub output_guardrail_results: Vec<OutputGuardrailResult>,
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
    /// ```rust
    /// use machi::agent::RunResult;
    /// use serde::Deserialize;
    /// use serde_json::json;
    ///
    /// #[derive(Deserialize)]
    /// struct Info { name: String }
    ///
    /// let result = RunResult {
    ///     output: json!({"name": "Rust"}),
    ///     usage: Default::default(),
    ///     steps: 1,
    ///     step_history: vec![],
    ///     agent_name: "test".into(),
    ///     input_guardrail_results: vec![],
    ///     output_guardrail_results: vec![],
    /// };
    /// let info: Info = result.parse().unwrap();
    /// assert_eq!(info.name, "Rust");
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
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// Token usage from managed sub-agent runs (zero for regular tools).
    pub sub_usage: Usage,
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
#[non_exhaustive]
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
/// ```rust
/// use machi::agent::UserInput;
/// use machi::message::ContentPart;
///
/// // Simple text — most common case.
/// let input: UserInput = "Hello!".into();
///
/// // Text + image via convenience constructor.
/// let input = UserInput::with_image(
///     "Describe this",
///     "https://example.com/img.png",
/// );
///
/// // Full multimodal via Vec<ContentPart>.
/// let input: UserInput = vec![
///     ContentPart::text("What's in these images?"),
///     ContentPart::image_url("https://example.com/a.png"),
///     ContentPart::image_url("https://example.com/b.png"),
/// ].into();
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
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
