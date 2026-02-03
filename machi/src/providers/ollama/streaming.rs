//! Ollama streaming response handling.

use crate::error::AgentError;
use crate::message::{
    ChatMessageStreamDelta, ChatMessageToolCallFunction, ChatMessageToolCallStreamDelta,
};
use crate::providers::common::TokenUsage;
use bytes::Bytes;
use futures::Stream;
use serde::Deserialize;
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Streaming response parser for Ollama's NDJSON format.
///
/// Ollama streams responses as newline-delimited JSON objects.
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

    /// Parse a single line of NDJSON into a stream delta.
    fn parse_line(&self, line: &str) -> Option<Result<ChatMessageStreamDelta, AgentError>> {
        let line = line.trim();

        if line.is_empty() {
            return None;
        }

        match serde_json::from_str::<StreamChunk>(line) {
            Ok(chunk) => Some(Ok(self.chunk_to_delta(chunk))),
            Err(e) => {
                tracing::debug!(
                    "Failed to parse Ollama streaming chunk: {} - line: {}",
                    e,
                    line
                );
                None
            }
        }
    }

    /// Convert a parsed chunk to a stream delta.
    fn chunk_to_delta(&self, chunk: StreamChunk) -> ChatMessageStreamDelta {
        let (content, tool_calls) = if let Some(msg) = chunk.message {
            let content = if msg.content.is_empty() {
                None
            } else {
                Some(msg.content)
            };

            // Parse tool calls from streaming response
            let tool_calls = msg.tool_calls.map(|tcs| {
                tcs.into_iter()
                    .enumerate()
                    .map(|(i, tc)| ChatMessageToolCallStreamDelta {
                        index: tc.function.as_ref().and_then(|f| f.index).or(Some(i)),
                        id: None,
                        r#type: Some("function".to_string()),
                        function: tc.function.map(|f| ChatMessageToolCallFunction {
                            name: f.name.unwrap_or_default(),
                            arguments: f.arguments.unwrap_or(Value::Null),
                            description: None,
                        }),
                    })
                    .collect()
            });

            (content, tool_calls)
        } else {
            (None, None)
        };

        let token_usage = if chunk.done {
            chunk.prompt_eval_count.map(|input| TokenUsage {
                input_tokens: input,
                output_tokens: chunk.eval_count.unwrap_or(0),
            })
        } else {
            None
        };

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
            // Try to parse any complete lines from the buffer
            if let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer[..newline_pos].to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();

                if let Some(result) = self.parse_line(&line) {
                    return Poll::Ready(Some(result));
                }
                continue;
            }

            // No complete line, need more data
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        self.buffer.push_str(text);
                    }
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
                            if let Some(result) = self.parse_line(line) {
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

/// Ollama streaming chunk structure.
#[derive(Debug, Deserialize)]
struct StreamChunk {
    /// The message content (partial).
    message: Option<StreamMessage>,
    /// Whether this is the final chunk.
    #[serde(default)]
    done: bool,
    /// Number of tokens in the prompt (only in final chunk).
    prompt_eval_count: Option<u32>,
    /// Number of tokens generated (only in final chunk).
    eval_count: Option<u32>,
}

/// Message content in a streaming chunk.
#[derive(Debug, Deserialize)]
struct StreamMessage {
    /// Message content.
    #[serde(default)]
    content: String,
    /// Tool calls from the model.
    tool_calls: Option<Vec<StreamToolCall>>,
}

/// Tool call in a streaming chunk.
#[derive(Debug, Deserialize)]
struct StreamToolCall {
    /// Function details.
    function: Option<StreamFunction>,
}

/// Function details in a streaming tool call.
#[derive(Debug, Deserialize)]
struct StreamFunction {
    /// Function index.
    index: Option<usize>,
    /// Function name.
    name: Option<String>,
    /// Function arguments.
    arguments: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chunk() {
        let response = StreamingResponse {
            inner: futures::stream::empty::<Result<Bytes, reqwest::Error>>(),
            buffer: String::new(),
        };

        let json = r#"{"message":{"content":"Hello"},"done":false}"#;
        let result = response.parse_line(json);

        assert!(result.is_some());
        if let Some(Ok(delta)) = result {
            assert_eq!(delta.content, Some("Hello".to_string()));
        }
    }

    #[test]
    fn test_parse_final_chunk() {
        let response = StreamingResponse {
            inner: futures::stream::empty::<Result<Bytes, reqwest::Error>>(),
            buffer: String::new(),
        };

        let json =
            r#"{"message":{"content":""},"done":true,"prompt_eval_count":10,"eval_count":50}"#;
        let result = response.parse_line(json);

        assert!(result.is_some());
        if let Some(Ok(delta)) = result {
            assert!(delta.token_usage.is_some());
            let usage = delta.token_usage.unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 50);
        }
    }

    #[test]
    fn test_parse_tool_call_chunk() {
        let response = StreamingResponse {
            inner: futures::stream::empty::<Result<Bytes, reqwest::Error>>(),
            buffer: String::new(),
        };

        let json = r#"{"message":{"content":"","tool_calls":[{"function":{"index":0,"name":"get_weather","arguments":{"city":"Tokyo"}}}]},"done":false}"#;
        let result = response.parse_line(json);

        assert!(result.is_some());
        if let Some(Ok(delta)) = result {
            assert!(delta.tool_calls.is_some());
            let tool_calls = delta.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].index, Some(0));
            assert!(tool_calls[0].function.is_some());
            let func = tool_calls[0].function.as_ref().unwrap();
            assert_eq!(func.name, "get_weather");
        }
    }
}
