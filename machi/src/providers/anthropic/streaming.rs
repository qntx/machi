//! Anthropic streaming response handling.

#![allow(
    clippy::option_if_let_else,
    clippy::needless_continue,
    clippy::enum_variant_names
)]

use crate::error::AgentError;
use crate::message::{
    ChatMessageStreamDelta, ChatMessageToolCallFunction, ChatMessageToolCallStreamDelta,
};
use crate::providers::common::TokenUsage;
use bytes::Bytes;
use futures::Stream;
use serde::Deserialize;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Streaming response parser for Anthropic's SSE format.
pub struct StreamingResponse<S> {
    inner: S,
    buffer: String,
    /// Accumulated tool use blocks by index
    tool_use_blocks: HashMap<usize, ToolUseAccumulator>,
    /// Current tool use index
    current_tool_index: usize,
}

/// Accumulator for tool use content blocks during streaming.
#[derive(Debug, Default)]
struct ToolUseAccumulator {
    id: String,
    name: String,
    input_json: String,
}

impl<S> StreamingResponse<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    /// Create a new streaming response parser.
    pub fn new(stream: S) -> Self {
        Self {
            inner: stream,
            buffer: String::new(),
            tool_use_blocks: HashMap::new(),
            current_tool_index: 0,
        }
    }

    /// Parse a single SSE data line into a stream delta.
    fn parse_sse_line(&mut self, line: &str) -> Option<Result<ChatMessageStreamDelta, AgentError>> {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with(':') {
            return None;
        }

        // Parse "data: " prefix
        if let Some(data) = trimmed.strip_prefix("data: ") {
            // Parse JSON event
            match serde_json::from_str::<StreamEvent>(data) {
                Ok(event) => self.handle_event(event),
                Err(e) => {
                    // Some events might not match our schema, skip them
                    tracing::debug!("Failed to parse streaming event: {} - data: {}", e, data);
                    None
                }
            }
        } else if trimmed.starts_with("event: ") {
            // Event type line, skip
            None
        } else {
            None
        }
    }

    /// Handle a parsed streaming event.
    fn handle_event(
        &mut self,
        event: StreamEvent,
    ) -> Option<Result<ChatMessageStreamDelta, AgentError>> {
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                match content_block {
                    ContentBlock::Text { .. } => {
                        // Text blocks don't need special handling at start
                        None
                    }
                    ContentBlock::ToolUse { id, name, .. } => {
                        // Start accumulating a tool use block
                        self.current_tool_index = index;
                        self.tool_use_blocks.insert(
                            index,
                            ToolUseAccumulator {
                                id,
                                name,
                                input_json: String::new(),
                            },
                        );
                        None
                    }
                    ContentBlock::Thinking { .. } => None,
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                match delta {
                    ContentDelta::TextDelta { text } => Some(Ok(ChatMessageStreamDelta {
                        content: Some(text),
                        tool_calls: None,
                        token_usage: None,
                    })),
                    ContentDelta::InputJsonDelta { partial_json } => {
                        // Accumulate JSON for tool use
                        if let Some(accumulator) = self.tool_use_blocks.get_mut(&index) {
                            accumulator.input_json.push_str(&partial_json);
                        }
                        None
                    }
                    ContentDelta::ThinkingDelta { .. } => None,
                }
            }
            StreamEvent::ContentBlockStop { index } => {
                // If this was a tool use block, emit it now
                if let Some(accumulator) = self.tool_use_blocks.remove(&index) {
                    let input = serde_json::from_str(&accumulator.input_json)
                        .unwrap_or(serde_json::Value::Null);

                    Some(Ok(ChatMessageStreamDelta {
                        content: None,
                        tool_calls: Some(vec![ChatMessageToolCallStreamDelta {
                            index: Some(index),
                            id: Some(accumulator.id),
                            r#type: Some("function".to_string()),
                            function: Some(ChatMessageToolCallFunction {
                                name: accumulator.name,
                                arguments: input,
                                description: None,
                            }),
                        }]),
                        token_usage: None,
                    }))
                } else {
                    None
                }
            }
            StreamEvent::MessageDelta { usage, .. } => {
                // Final message delta with usage
                usage.map(|usage| {
                    Ok(ChatMessageStreamDelta {
                        content: None,
                        tool_calls: None,
                        token_usage: Some(TokenUsage {
                            input_tokens: usage.input_tokens.unwrap_or(0),
                            output_tokens: usage.output_tokens,
                        }),
                    })
                })
            }
            StreamEvent::MessageStart { message } => {
                // Extract input tokens from message start
                message.usage.map(|usage| {
                    Ok(ChatMessageStreamDelta {
                        content: None,
                        tool_calls: None,
                        token_usage: Some(TokenUsage {
                            input_tokens: usage.input_tokens.unwrap_or(0),
                            output_tokens: 0,
                        }),
                    })
                })
            }
            StreamEvent::MessageStop | StreamEvent::Ping => None,
            StreamEvent::Error { error } => Some(Err(AgentError::model(format!(
                "Anthropic streaming error: {}",
                error.message
            )))),
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

/// Anthropic streaming event types.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        #[allow(dead_code)]
        delta: MessageDeltaContent,
        usage: Option<UsageDelta>,
    },
    MessageStop,
    Ping,
    Error {
        error: StreamError,
    },
}

#[derive(Debug, Deserialize)]
struct MessageStart {
    #[allow(dead_code)]
    id: Option<String>,
    usage: Option<UsageStart>,
}

#[derive(Debug, Deserialize)]
struct UsageStart {
    input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text {
        #[allow(dead_code)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[allow(dead_code)]
        input: serde_json::Value,
    },
    Thinking {
        #[allow(dead_code)]
        thinking: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        #[allow(dead_code)]
        thinking: String,
    },
}

#[derive(Debug, Deserialize)]
struct MessageDeltaContent {
    #[allow(dead_code)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageDelta {
    input_tokens: Option<u32>,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct StreamError {
    #[allow(dead_code)]
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_delta() {
        let mut response = StreamingResponse {
            inner: futures::stream::empty::<Result<Bytes, reqwest::Error>>(),
            buffer: String::new(),
            tool_use_blocks: HashMap::new(),
            current_tool_index: 0,
        };

        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: "Hello".to_string(),
            },
        };

        let result = response.handle_event(event);
        assert!(result.is_some());

        if let Some(Ok(delta)) = result {
            assert_eq!(delta.content, Some("Hello".to_string()));
        }
    }
}
