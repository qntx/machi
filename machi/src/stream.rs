//! Streaming response types for LLM operations.
//!
//! This module provides types for handling streaming responses from LLM providers,
//! enabling real-time display of generated content.

use serde::{Deserialize, Serialize};

use crate::usage::Usage;

/// A chunk of streaming response from an LLM.
///
/// # `OpenAI` API Alignment
/// This enum represents the various types of content that can be streamed
/// from an LLM provider, following `OpenAI`'s streaming response format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
#[allow(clippy::module_name_repetitions)]
pub enum StreamChunk {
    /// Text content chunk.
    Text(String),

    /// Reasoning content chunk (for o1/o3 reasoning models).
    ReasoningContent(String),

    /// Audio content chunk (base64 encoded, for audio-capable models).
    Audio {
        /// Base64-encoded audio data.
        data: String,
        /// Audio transcript (if available).
        #[serde(skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
    },

    /// Start of a tool use/function call.
    ToolUseStart {
        /// Index of this tool call in the response.
        index: usize,
        /// Unique identifier for this tool call.
        id: String,
        /// Name of the function being called.
        name: String,
    },

    /// Partial arguments for an in-progress tool call.
    ToolUseDelta {
        /// Index of the tool call being updated.
        index: usize,
        /// Partial JSON arguments.
        partial_json: String,
    },

    /// Tool call is complete.
    ToolUseComplete {
        /// Index of the completed tool call.
        index: usize,
    },

    /// Token usage information.
    Usage(Usage),

    /// Stream is complete.
    Done {
        /// Stop reason from the model.
        stop_reason: Option<StopReason>,
    },

    /// Error during streaming.
    Error {
        /// Error message.
        message: String,
    },
}

impl StreamChunk {
    /// Creates a text chunk.
    #[inline]
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Creates a tool use start chunk.
    #[must_use]
    pub fn tool_use_start(index: usize, id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolUseStart {
            index,
            id: id.into(),
            name: name.into(),
        }
    }

    /// Creates a tool use delta chunk.
    #[must_use]
    pub fn tool_use_delta(index: usize, partial_json: impl Into<String>) -> Self {
        Self::ToolUseDelta {
            index,
            partial_json: partial_json.into(),
        }
    }

    /// Creates a done chunk.
    #[must_use]
    pub const fn done(stop_reason: Option<StopReason>) -> Self {
        Self::Done { stop_reason }
    }

    /// Creates an error chunk.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Returns the text content if this is a text chunk.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Returns `true` if this is a text chunk.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Returns `true` if this is a done chunk.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        matches!(self, Self::Done { .. })
    }

    /// Returns `true` if this is an error chunk.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Creates a reasoning content chunk.
    #[inline]
    #[must_use]
    pub fn reasoning(content: impl Into<String>) -> Self {
        Self::ReasoningContent(content.into())
    }

    /// Creates an audio chunk.
    #[must_use]
    pub fn audio(data: impl Into<String>, transcript: Option<String>) -> Self {
        Self::Audio {
            data: data.into(),
            transcript,
        }
    }

    /// Returns the reasoning content if this is a reasoning chunk.
    #[must_use]
    pub fn as_reasoning(&self) -> Option<&str> {
        match self {
            Self::ReasoningContent(content) => Some(content),
            _ => None,
        }
    }

    /// Returns `true` if this is a reasoning content chunk.
    #[must_use]
    pub const fn is_reasoning(&self) -> bool {
        matches!(self, Self::ReasoningContent(_))
    }

    /// Returns `true` if this is an audio chunk.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Audio { .. })
    }
}

/// Reason why the model stopped generating.
///
/// # `OpenAI` API Alignment
/// Maps to `finish_reason` in `OpenAI`'s Chat Completions API response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    /// Natural stop (end of response).
    #[default]
    Stop,
    /// Maximum token limit reached.
    Length,
    /// Model decided to call tools.
    #[serde(alias = "function_call")]
    ToolCalls,
    /// Content was filtered by safety systems.
    ContentFilter,
    /// Model is still generating (streaming only, no `finish_reason` yet).
    Null,
}

impl StopReason {
    /// Returns the string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Null => "null",
        }
    }

    /// Parse from a string (case-insensitive).
    ///
    /// Handles various provider-specific finish reason strings:
    /// - `OpenAI`: "stop", "length", "`tool_calls`", "`content_filter`", "`function_call`"
    /// - Anthropic: "`end_turn`", "`max_tokens`", "`tool_use`"
    /// - Ollama: "stop", "length"
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "length" | "max_tokens" => Self::Length,
            "tool_calls" | "tool_use" | "function_call" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            "null" => Self::Null,
            // "stop", "end_turn", and any other value defaults to Stop
            _ => Self::Stop,
        }
    }

    /// Returns `true` if the model completed normally.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Stop | Self::ToolCalls)
    }

    /// Returns `true` if the model was cut off due to length.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        matches!(self, Self::Length)
    }

    /// Returns `true` if content was filtered.
    #[must_use]
    pub const fn is_filtered(&self) -> bool {
        matches!(self, Self::ContentFilter)
    }

    /// Returns `true` if the model called tools/functions.
    #[must_use]
    pub const fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCalls)
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Aggregator for building complete content from stream chunks.
#[derive(Debug, Clone, Default)]
#[allow(clippy::module_name_repetitions)]
pub struct StreamAggregator {
    /// Accumulated text content.
    text: String,
    /// Accumulated reasoning content (for o1/o3 models).
    reasoning_content: String,
    /// Tool calls being built.
    tool_calls: std::collections::BTreeMap<usize, ToolCallBuilder>,
    /// Total usage.
    usage: Option<Usage>,
    /// Final stop reason.
    stop_reason: Option<StopReason>,
}

/// Builder for assembling tool calls from stream chunks.
#[derive(Debug, Clone, Default)]
struct ToolCallBuilder {
    /// The tool call ID.
    id: String,
    /// The function name.
    name: String,
    /// Accumulated argument fragments.
    arguments: String,
}

impl StreamAggregator {
    /// Creates a new aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies a stream chunk to the aggregator.
    pub fn apply(&mut self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::Text(text) => {
                self.text.push_str(text);
            }
            StreamChunk::ToolUseStart { index, id, name } => {
                self.tool_calls.insert(
                    *index,
                    ToolCallBuilder {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: String::new(),
                    },
                );
            }
            StreamChunk::ToolUseDelta {
                index,
                partial_json,
            } => {
                if let Some(tc) = self.tool_calls.get_mut(index) {
                    tc.arguments.push_str(partial_json);
                }
            }
            StreamChunk::ReasoningContent(content) => {
                self.reasoning_content.push_str(content);
            }
            StreamChunk::Audio { .. }
            | StreamChunk::ToolUseComplete { .. }
            | StreamChunk::Error { .. } => {}
            StreamChunk::Usage(usage) => {
                self.usage = Some(*usage);
            }
            StreamChunk::Done { stop_reason } => {
                self.stop_reason = *stop_reason;
            }
        }
    }

    /// Returns the current accumulated text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the accumulated reasoning content.
    #[must_use]
    pub fn reasoning_content(&self) -> &str {
        &self.reasoning_content
    }

    /// Returns `true` if there is reasoning content.
    #[must_use]
    pub const fn has_reasoning_content(&self) -> bool {
        !self.reasoning_content.is_empty()
    }

    /// Returns the accumulated usage.
    #[must_use]
    pub const fn usage(&self) -> Option<Usage> {
        self.usage
    }

    /// Returns the stop reason.
    #[must_use]
    pub const fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    /// Returns `true` if any tool calls have been started.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Builds the final tool calls.
    #[must_use]
    pub fn build_tool_calls(&self) -> Vec<crate::message::ToolCall> {
        self.tool_calls
            .values()
            .map(|tc| crate::message::ToolCall::function(&tc.id, &tc.name, &tc.arguments))
            .collect()
    }

    /// Converts the accumulated stream data into a [`ChatResponse`].
    ///
    /// Constructs an assistant message with either text content, tool calls,
    /// or both, along with usage statistics and stop reason.
    #[must_use]
    pub fn into_chat_response(self) -> crate::chat::ChatResponse {
        use crate::chat::ChatResponse;
        use crate::message::{Content, Message, Role};

        let tool_calls = self.build_tool_calls();
        let has_text = !self.text.is_empty();
        let has_tools = !tool_calls.is_empty();

        let mut message = match (has_text, has_tools) {
            (_, true) => {
                let mut msg = Message::assistant_tool_calls(tool_calls);
                if has_text {
                    msg.content = Some(Content::text(self.text));
                }
                msg
            }
            _ => Message::new(Role::Assistant, Content::text(self.text)),
        };

        if !self.reasoning_content.is_empty() {
            message.reasoning_content = Some(self.reasoning_content);
        }

        let mut response = ChatResponse::new(message);
        if let Some(reason) = self.stop_reason {
            response = response.with_stop_reason(reason);
        }
        if let Some(usage) = self.usage {
            response = response.with_usage(usage);
        }
        response
    }
}
