//! Streaming response types for LLM operations.
//!
//! This module provides types for handling streaming responses from LLM providers,
//! enabling real-time display of generated content.

use serde::{Deserialize, Serialize};

use crate::usage::Usage;

/// A chunk of streaming response from an LLM.
///
/// # OpenAI API Alignment
/// This enum represents the various types of content that can be streamed
/// from an LLM provider, following OpenAI's streaming response format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
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
/// # OpenAI API Alignment
/// Maps to `finish_reason` in OpenAI's Chat Completions API response.
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
    ToolCalls,
    /// Content was filtered by safety systems.
    ContentFilter,
    /// Model is still generating (streaming only, no finish_reason yet).
    Null,
    /// Deprecated: Function call (use ToolCalls instead).
    #[serde(rename = "function_call")]
    FunctionCall,
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
            Self::FunctionCall => "function_call",
        }
    }

    /// Parse from a string (case-insensitive).
    ///
    /// Handles various provider-specific finish reason strings:
    /// - OpenAI: "stop", "length", "tool_calls", "content_filter", "function_call"
    /// - Anthropic: "end_turn", "max_tokens", "tool_use"
    /// - Ollama: "stop", "length"
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "length" | "max_tokens" => Self::Length,
            "tool_calls" | "tool_use" => Self::ToolCalls,
            "function_call" => Self::FunctionCall,
            "content_filter" => Self::ContentFilter,
            "null" => Self::Null,
            // "stop", "end_turn", and any other value defaults to Stop
            _ => Self::Stop,
        }
    }

    /// Returns `true` if the model completed normally.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Stop | Self::ToolCalls | Self::FunctionCall)
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
        matches!(self, Self::ToolCalls | Self::FunctionCall)
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Aggregator for building complete content from stream chunks.
#[derive(Debug, Clone, Default)]
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

#[derive(Debug, Clone, Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    mod stream_chunk {
        use super::*;

        #[test]
        fn text_creates_text_chunk() {
            let chunk = StreamChunk::text("Hello");
            assert!(chunk.is_text());
            assert_eq!(chunk.as_text(), Some("Hello"));
        }

        #[test]
        fn text_accepts_string() {
            let chunk = StreamChunk::text(String::from("World"));
            assert_eq!(chunk.as_text(), Some("World"));
        }

        #[test]
        fn reasoning_creates_reasoning_chunk() {
            let chunk = StreamChunk::reasoning("Let me think...");
            assert!(chunk.is_reasoning());
            assert_eq!(chunk.as_reasoning(), Some("Let me think..."));
        }

        #[test]
        fn audio_creates_audio_chunk() {
            let chunk = StreamChunk::audio("base64data", Some("transcript".to_owned()));
            assert!(chunk.is_audio());
            if let StreamChunk::Audio { data, transcript } = chunk {
                assert_eq!(data, "base64data");
                assert_eq!(transcript, Some("transcript".to_owned()));
            } else {
                panic!("Expected Audio chunk");
            }
        }

        #[test]
        fn audio_with_no_transcript() {
            let chunk = StreamChunk::audio("data", None);
            if let StreamChunk::Audio { transcript, .. } = chunk {
                assert!(transcript.is_none());
            }
        }

        #[test]
        fn tool_use_start_creates_chunk() {
            let chunk = StreamChunk::tool_use_start(0, "call_123", "get_weather");
            if let StreamChunk::ToolUseStart { index, id, name } = chunk {
                assert_eq!(index, 0);
                assert_eq!(id, "call_123");
                assert_eq!(name, "get_weather");
            } else {
                panic!("Expected ToolUseStart chunk");
            }
        }

        #[test]
        fn tool_use_delta_creates_chunk() {
            let chunk = StreamChunk::tool_use_delta(0, r#"{"city":"#);
            if let StreamChunk::ToolUseDelta {
                index,
                partial_json,
            } = chunk
            {
                assert_eq!(index, 0);
                assert_eq!(partial_json, r#"{"city":"#);
            } else {
                panic!("Expected ToolUseDelta chunk");
            }
        }

        #[test]
        fn done_creates_done_chunk() {
            let chunk = StreamChunk::done(Some(StopReason::Stop));
            assert!(chunk.is_done());
            if let StreamChunk::Done { stop_reason } = chunk {
                assert_eq!(stop_reason, Some(StopReason::Stop));
            }
        }

        #[test]
        fn done_with_none() {
            let chunk = StreamChunk::done(None);
            assert!(chunk.is_done());
        }

        #[test]
        fn error_creates_error_chunk() {
            let chunk = StreamChunk::error("Something went wrong");
            assert!(chunk.is_error());
            if let StreamChunk::Error { message } = chunk {
                assert_eq!(message, "Something went wrong");
            }
        }

        #[test]
        fn as_text_returns_none_for_non_text() {
            let chunk = StreamChunk::done(None);
            assert!(chunk.as_text().is_none());
        }

        #[test]
        fn as_reasoning_returns_none_for_non_reasoning() {
            let chunk = StreamChunk::text("Hello");
            assert!(chunk.as_reasoning().is_none());
        }

        #[test]
        fn is_methods_return_false_for_other_types() {
            let text = StreamChunk::text("Hi");
            assert!(!text.is_done());
            assert!(!text.is_error());
            assert!(!text.is_reasoning());
            assert!(!text.is_audio());

            let done = StreamChunk::done(None);
            assert!(!done.is_text());
            assert!(!done.is_error());
        }

        #[test]
        fn serde_done_with_stop_reason() {
            // Note: Text/ReasoningContent variants cannot be serialized with internally tagged enums
            // Test struct variants which work correctly
            let chunk = StreamChunk::done(Some(StopReason::Stop));
            let json = serde_json::to_string(&chunk).unwrap();
            assert!(json.contains("done"));
            let parsed: StreamChunk = serde_json::from_str(&json).unwrap();
            assert!(parsed.is_done());
        }

        #[test]
        fn serde_done_roundtrip() {
            let chunk = StreamChunk::done(Some(StopReason::ToolCalls));
            let json = serde_json::to_string(&chunk).unwrap();
            let parsed: StreamChunk = serde_json::from_str(&json).unwrap();
            if let StreamChunk::Done { stop_reason } = parsed {
                assert_eq!(stop_reason, Some(StopReason::ToolCalls));
            }
        }

        #[test]
        fn serde_uses_snake_case_tags() {
            let chunk = StreamChunk::tool_use_start(0, "id", "name");
            let json = serde_json::to_string(&chunk).unwrap();
            assert!(json.contains("tool_use_start"));
        }

        #[test]
        fn clone_trait() {
            let chunk = StreamChunk::text("Clone me");
            let cloned = chunk;
            assert_eq!(cloned.as_text(), Some("Clone me"));
        }
    }

    mod stop_reason {
        use super::*;

        #[test]
        fn default_is_stop() {
            assert_eq!(StopReason::default(), StopReason::Stop);
        }

        #[test]
        fn as_str_all_variants() {
            assert_eq!(StopReason::Stop.as_str(), "stop");
            assert_eq!(StopReason::Length.as_str(), "length");
            assert_eq!(StopReason::ToolCalls.as_str(), "tool_calls");
            assert_eq!(StopReason::ContentFilter.as_str(), "content_filter");
            assert_eq!(StopReason::Null.as_str(), "null");
            assert_eq!(StopReason::FunctionCall.as_str(), "function_call");
        }

        #[test]
        fn display_matches_as_str() {
            for reason in [
                StopReason::Stop,
                StopReason::Length,
                StopReason::ToolCalls,
                StopReason::ContentFilter,
                StopReason::Null,
                StopReason::FunctionCall,
            ] {
                assert_eq!(reason.to_string(), reason.as_str());
            }
        }

        #[test]
        fn parse_openai_stop() {
            assert_eq!(StopReason::parse("stop"), StopReason::Stop);
        }

        #[test]
        fn parse_openai_length() {
            assert_eq!(StopReason::parse("length"), StopReason::Length);
        }

        #[test]
        fn parse_anthropic_max_tokens() {
            assert_eq!(StopReason::parse("max_tokens"), StopReason::Length);
        }

        #[test]
        fn parse_openai_tool_calls() {
            assert_eq!(StopReason::parse("tool_calls"), StopReason::ToolCalls);
        }

        #[test]
        fn parse_anthropic_tool_use() {
            assert_eq!(StopReason::parse("tool_use"), StopReason::ToolCalls);
        }

        #[test]
        fn parse_function_call() {
            assert_eq!(StopReason::parse("function_call"), StopReason::FunctionCall);
        }

        #[test]
        fn parse_content_filter() {
            assert_eq!(
                StopReason::parse("content_filter"),
                StopReason::ContentFilter
            );
        }

        #[test]
        fn parse_null() {
            assert_eq!(StopReason::parse("null"), StopReason::Null);
        }

        #[test]
        fn parse_anthropic_end_turn() {
            assert_eq!(StopReason::parse("end_turn"), StopReason::Stop);
        }

        #[test]
        fn parse_unknown_defaults_to_stop() {
            assert_eq!(StopReason::parse("unknown_value"), StopReason::Stop);
        }

        #[test]
        fn parse_case_insensitive() {
            assert_eq!(StopReason::parse("TOOL_CALLS"), StopReason::ToolCalls);
            assert_eq!(StopReason::parse("Length"), StopReason::Length);
        }

        #[test]
        fn is_complete_returns_true() {
            assert!(StopReason::Stop.is_complete());
            assert!(StopReason::ToolCalls.is_complete());
            assert!(StopReason::FunctionCall.is_complete());
        }

        #[test]
        fn is_complete_returns_false() {
            assert!(!StopReason::Length.is_complete());
            assert!(!StopReason::ContentFilter.is_complete());
            assert!(!StopReason::Null.is_complete());
        }

        #[test]
        fn is_truncated() {
            assert!(StopReason::Length.is_truncated());
            assert!(!StopReason::Stop.is_truncated());
        }

        #[test]
        fn is_filtered() {
            assert!(StopReason::ContentFilter.is_filtered());
            assert!(!StopReason::Stop.is_filtered());
        }

        #[test]
        fn is_tool_call() {
            assert!(StopReason::ToolCalls.is_tool_call());
            assert!(StopReason::FunctionCall.is_tool_call());
            assert!(!StopReason::Stop.is_tool_call());
        }

        #[test]
        fn serde_roundtrip() {
            for reason in [
                StopReason::Stop,
                StopReason::Length,
                StopReason::ToolCalls,
                StopReason::ContentFilter,
            ] {
                let json = serde_json::to_string(&reason).unwrap();
                let parsed: StopReason = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed, reason);
            }
        }

        #[test]
        fn serde_function_call_rename() {
            let json = serde_json::to_string(&StopReason::FunctionCall).unwrap();
            assert_eq!(json, r#""function_call""#);
        }

        #[test]
        fn copy_trait() {
            let reason = StopReason::ToolCalls;
            let copy = reason;
            assert_eq!(reason, copy);
        }

        #[test]
        fn hash_trait() {
            use std::collections::HashSet;
            let mut set = HashSet::new();
            set.insert(StopReason::Stop);
            set.insert(StopReason::Length);
            assert_eq!(set.len(), 2);
        }
    }

    mod stream_aggregator {
        use super::*;

        #[test]
        fn new_creates_empty_aggregator() {
            let agg = StreamAggregator::new();
            assert!(agg.text().is_empty());
            assert!(agg.reasoning_content().is_empty());
            assert!(!agg.has_reasoning_content());
            assert!(agg.usage().is_none());
            assert!(agg.stop_reason().is_none());
            assert!(!agg.has_tool_calls());
        }

        #[test]
        fn default_creates_empty_aggregator() {
            let agg = StreamAggregator::default();
            assert!(agg.text().is_empty());
        }

        #[test]
        fn apply_text_accumulates() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::text("Hello"));
            agg.apply(&StreamChunk::text(" World"));
            assert_eq!(agg.text(), "Hello World");
        }

        #[test]
        fn apply_reasoning_accumulates() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::reasoning("First "));
            agg.apply(&StreamChunk::reasoning("Second"));
            assert_eq!(agg.reasoning_content(), "First Second");
            assert!(agg.has_reasoning_content());
        }

        #[test]
        fn apply_usage_sets_usage() {
            let mut agg = StreamAggregator::new();
            let usage = Usage::new(100, 50);
            agg.apply(&StreamChunk::Usage(usage));
            assert_eq!(agg.usage(), Some(usage));
        }

        #[test]
        fn apply_done_sets_stop_reason() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::done(Some(StopReason::Length)));
            assert_eq!(agg.stop_reason(), Some(StopReason::Length));
        }

        #[test]
        fn apply_tool_use_start_creates_tool_call() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::tool_use_start(0, "call_123", "get_weather"));
            assert!(agg.has_tool_calls());
        }

        #[test]
        fn apply_tool_use_delta_appends_arguments() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::tool_use_start(0, "call_123", "get_weather"));
            agg.apply(&StreamChunk::tool_use_delta(0, r#"{"city":"#));
            agg.apply(&StreamChunk::tool_use_delta(0, r#""Tokyo"}"#));

            let tool_calls = agg.build_tool_calls();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].function.arguments, r#"{"city":"Tokyo"}"#);
        }

        #[test]
        fn apply_tool_use_delta_ignores_unknown_index() {
            let mut agg = StreamAggregator::new();
            // Delta without corresponding start should be ignored
            agg.apply(&StreamChunk::tool_use_delta(999, "ignored"));
            assert!(!agg.has_tool_calls());
        }

        #[test]
        fn apply_multiple_tool_calls() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::tool_use_start(0, "call_1", "tool_a"));
            agg.apply(&StreamChunk::tool_use_start(1, "call_2", "tool_b"));
            agg.apply(&StreamChunk::tool_use_delta(0, r#"{"a":1}"#));
            agg.apply(&StreamChunk::tool_use_delta(1, r#"{"b":2}"#));

            let tool_calls = agg.build_tool_calls();
            assert_eq!(tool_calls.len(), 2);
        }

        #[test]
        fn apply_audio_is_noop() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::audio("data", None));
            // Audio chunks are currently ignored
            assert!(agg.text().is_empty());
        }

        #[test]
        fn apply_error_is_noop() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::error("Error"));
            // Error chunks are currently ignored
            assert!(agg.text().is_empty());
        }

        #[test]
        fn build_tool_calls_returns_correct_structure() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::tool_use_start(0, "call_abc", "search"));
            agg.apply(&StreamChunk::tool_use_delta(0, r#"{"query":"test"}"#));

            let tool_calls = agg.build_tool_calls();
            assert_eq!(tool_calls[0].id, "call_abc");
            assert_eq!(tool_calls[0].function.name, "search");
            assert_eq!(tool_calls[0].function.arguments, r#"{"query":"test"}"#);
        }

        #[test]
        fn clone_trait() {
            let mut agg = StreamAggregator::new();
            agg.apply(&StreamChunk::text("Hello"));
            let cloned = agg.clone();
            assert_eq!(cloned.text(), "Hello");
        }

        #[test]
        fn full_streaming_scenario() {
            let mut agg = StreamAggregator::new();

            // Simulate streaming response
            agg.apply(&StreamChunk::text("The weather in "));
            agg.apply(&StreamChunk::text("Tokyo is sunny."));
            agg.apply(&StreamChunk::Usage(Usage::new(50, 20)));
            agg.apply(&StreamChunk::done(Some(StopReason::Stop)));

            assert_eq!(agg.text(), "The weather in Tokyo is sunny.");
            assert_eq!(agg.usage().unwrap().input_tokens, 50);
            assert_eq!(agg.stop_reason(), Some(StopReason::Stop));
        }

        #[test]
        fn streaming_with_tool_calls_scenario() {
            let mut agg = StreamAggregator::new();

            // Simulate tool call streaming
            agg.apply(&StreamChunk::tool_use_start(0, "call_123", "get_weather"));
            agg.apply(&StreamChunk::tool_use_delta(0, r"{"));
            agg.apply(&StreamChunk::tool_use_delta(0, r#""city":"#));
            agg.apply(&StreamChunk::tool_use_delta(0, r#""Paris"}"#));
            agg.apply(&StreamChunk::done(Some(StopReason::ToolCalls)));

            assert!(agg.has_tool_calls());
            assert_eq!(agg.stop_reason(), Some(StopReason::ToolCalls));

            let calls = agg.build_tool_calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].function.name, "get_weather");
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn stream_chunk_with_usage_struct() {
            let usage = Usage::new(100, 50);
            let chunk = StreamChunk::Usage(usage);

            if let StreamChunk::Usage(u) = chunk {
                assert_eq!(u.input_tokens, 100);
                assert_eq!(u.output_tokens, 50);
            } else {
                panic!("Expected Usage chunk");
            }
        }

        #[test]
        fn serde_struct_variant_chunks() {
            // Note: Newtype variants (Text, ReasoningContent) cannot be serialized with internally tagged enums
            // Only test struct variants which work correctly with serde
            let chunks = vec![
                StreamChunk::audio("data", Some("transcript".to_owned())),
                StreamChunk::tool_use_start(0, "id", "name"),
                StreamChunk::tool_use_delta(0, "{}"),
                StreamChunk::done(Some(StopReason::Stop)),
                StreamChunk::error("Error"),
            ];

            for chunk in chunks {
                let json = serde_json::to_string(&chunk).unwrap();
                let _parsed: StreamChunk = serde_json::from_str(&json).unwrap();
            }
        }

        #[test]
        fn reasoning_model_scenario() {
            let mut agg = StreamAggregator::new();

            // o1/o3 model with reasoning content
            agg.apply(&StreamChunk::reasoning("Let me analyze this..."));
            agg.apply(&StreamChunk::reasoning(" The answer is..."));
            agg.apply(&StreamChunk::text("42"));
            agg.apply(&StreamChunk::Usage(
                Usage::new(100, 200).with_reasoning(150),
            ));
            agg.apply(&StreamChunk::done(Some(StopReason::Stop)));

            assert!(agg.has_reasoning_content());
            assert_eq!(
                agg.reasoning_content(),
                "Let me analyze this... The answer is..."
            );
            assert_eq!(agg.text(), "42");
            assert_eq!(agg.usage().unwrap().reasoning_tokens(), 150);
        }
    }
}
