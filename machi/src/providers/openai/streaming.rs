//! `OpenAI` streaming response handling.

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
    inner: S,
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
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with(':') {
            return None;
        }

        // Parse "data: " prefix
        if let Some(data) = line.strip_prefix("data: ") {
            // Check for stream end
            if data.trim() == "[DONE]" {
                return None;
            }

            // Parse JSON
            match serde_json::from_str::<StreamChunk>(data) {
                Ok(chunk) => Some(Ok(self.chunk_to_delta(chunk))),
                Err(e) => Some(Err(AgentError::model(format!(
                    "Failed to parse streaming response: {e}"
                )))),
            }
        } else {
            None
        }
    }

    /// Convert a parsed chunk to a stream delta.
    fn chunk_to_delta(&self, chunk: StreamChunk) -> ChatMessageStreamDelta {
        let choice = chunk.choices.into_iter().next();

        let (content, tool_calls) = if let Some(choice) = choice {
            let content = choice.delta.content;
            let tool_calls = choice.delta.tool_calls.map(|tcs| {
                tcs.into_iter()
                    .map(|tc| ChatMessageToolCallStreamDelta {
                        index: tc.index,
                        id: tc.id,
                        r#type: tc.r#type,
                        function: tc.function.map(|f| ChatMessageToolCallFunction {
                            name: f.name.unwrap_or_default(),
                            arguments: f
                                .arguments
                                .map_or(serde_json::Value::Null, |a| {
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
                    // Continue to try parsing
                    continue;
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
    #[allow(dead_code)]
    id: Option<String>,
    choices: Vec<StreamChoice>,
    usage: Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[allow(dead_code)]
    index: usize,
    delta: StreamDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    r#type: Option<String>,
    function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamUsage {
    prompt_tokens: u32,
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
