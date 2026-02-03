//! `OpenAI` streaming response handling.

#![allow(clippy::unused_self)]

use crate::error::AgentError;
use crate::message::{
    ChatMessageStreamDelta, ChatMessageToolCallFunction, ChatMessageToolCallStreamDelta,
};
use crate::providers::common::TokenUsage;
use bytes::Bytes;
use futures::Stream;
use serde::Deserialize;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Streaming response parser for `OpenAI`'s SSE format.
pub struct StreamingResponse<S> {
    /// Inner byte stream.
    inner: S,
    /// Line buffer for partial data.
    buffer: String,
}

impl<S> StreamingResponse<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    /// Create a new streaming response parser.
    pub const fn new(stream: S) -> Self {
        Self {
            inner: stream,
            buffer: String::new(),
        }
    }

    /// Parse a single SSE data line into a stream delta.
    fn parse_sse_line(&self, line: &str) -> Option<Result<ChatMessageStreamDelta, AgentError>> {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with(':') {
            return None;
        }

        // Parse "data: " prefix
        if let Some(data) = trimmed.strip_prefix("data: ") {
            // Check for stream end
            if data.trim() == "[DONE]" {
                return None;
            }

            // Parse JSON
            match serde_json::from_str::<StreamChunk>(data) {
                Ok(chunk) => Some(Ok(Self::chunk_to_delta(chunk))),
                Err(e) => Some(Err(AgentError::model(format!(
                    "Failed to parse streaming response: {e}"
                )))),
            }
        } else {
            None
        }
    }

    /// Convert a parsed chunk to a stream delta.
    fn chunk_to_delta(chunk: StreamChunk) -> ChatMessageStreamDelta {
        let first_choice = chunk.choices.into_iter().next();

        let (content, tool_calls) = if let Some(choice) = first_choice {
            let content = choice.delta.content;
            let tool_calls = choice.delta.tool_calls.map(|tcs| {
                tcs.into_iter()
                    .map(|tc| ChatMessageToolCallStreamDelta {
                        index: tc.index,
                        id: tc.id,
                        r#type: tc.r#type,
                        function: tc.function.map(|f| ChatMessageToolCallFunction {
                            name: f.name.unwrap_or_default(),
                            arguments: f.arguments.map_or(serde_json::Value::Null, |a| {
                                serde_json::from_str(&a).unwrap_or(serde_json::Value::Null)
                            }),
                            description: None,
                        }),
                    })
                    .collect()
            });
            (content, tool_calls)
        } else {
            (None, None)
        };

        let token_usage = chunk.usage.map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        });

        ChatMessageStreamDelta {
            content,
            tool_calls,
            token_usage,
        }
    }
}

impl<S> Stream for StreamingResponse<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ChatMessageStreamDelta, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // First, try to parse any complete lines from the buffer
            if let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer[..newline_pos].to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();

                if let Some(result) = self.parse_sse_line(&line) {
                    return Poll::Ready(Some(result));
                }
                // Continue to process more lines
                continue;
            }

            // No complete line, need more data
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        self.buffer.push_str(text);
                    }
                    // Loop continues to try parsing
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(AgentError::from(e))));
                }
                Poll::Ready(None) => {
                    // Stream ended, process any remaining buffer
                    if !self.buffer.is_empty() {
                        let remaining = std::mem::take(&mut self.buffer);
                        for line in remaining.lines() {
                            if let Some(result) = self.parse_sse_line(line) {
                                return Poll::Ready(Some(result));
                            }
                        }
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// `OpenAI` streaming chunk structure.
#[derive(Debug, Deserialize)]
struct StreamChunk {
    /// Chunk ID.
    #[allow(dead_code)]
    id: Option<String>,
    /// Response choices.
    choices: Vec<StreamChoice>,
    /// Token usage (only in final chunk).
    usage: Option<StreamUsage>,
}

/// Streaming response choice.
#[derive(Debug, Deserialize)]
struct StreamChoice {
    /// Choice index.
    #[allow(dead_code)]
    index: usize,
    /// Delta content.
    delta: StreamDelta,
    /// Finish reason.
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

/// Delta content in streaming response.
#[derive(Debug, Deserialize)]
struct StreamDelta {
    /// Text content.
    content: Option<String>,
    /// Tool calls.
    tool_calls: Option<Vec<StreamToolCall>>,
}

/// Tool call in streaming response.
#[derive(Debug, Deserialize)]
struct StreamToolCall {
    /// Tool call index.
    index: Option<usize>,
    /// Tool call ID.
    id: Option<String>,
    /// Tool call type.
    r#type: Option<String>,
    /// Function details.
    function: Option<StreamFunction>,
}

/// Function details in streaming tool call.
#[derive(Debug, Deserialize)]
struct StreamFunction {
    /// Function name.
    name: Option<String>,
    /// Function arguments (JSON string).
    arguments: Option<String>,
}

/// Token usage in streaming response.
#[derive(Debug, Deserialize)]
struct StreamUsage {
    /// Input tokens.
    prompt_tokens: u32,
    /// Output tokens.
    completion_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_done() {
        let response = StreamingResponse {
            inner: futures::stream::empty::<Result<Bytes, reqwest::Error>>(),
            buffer: String::new(),
        };

        assert!(response.parse_sse_line("data: [DONE]").is_none());
    }

    #[test]
    fn test_parse_sse_empty() {
        let response = StreamingResponse {
            inner: futures::stream::empty::<Result<Bytes, reqwest::Error>>(),
            buffer: String::new(),
        };

        assert!(response.parse_sse_line("").is_none());
        assert!(response.parse_sse_line(": comment").is_none());
    }
}
