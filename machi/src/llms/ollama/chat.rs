//! Ollama ChatProvider implementation.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::Deserialize;

use crate::chat::ChatProvider;
use crate::chat::{ChatRequest, ChatResponse};
use crate::error::{LlmError, Result};
use crate::message::{Content, Message, Role, ToolCall};
use crate::stream::{StopReason, StreamChunk};
use crate::usage::Usage;

use super::client::{Ollama, OllamaToolCall};
use super::stream::parse_stream_line;

/// Ollama chat completion response.
#[derive(Debug, Clone, Deserialize)]
struct OllamaChatResponse {
    pub model: String,
    pub message: OllamaResponseMessage,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
}

/// Ollama response message.
#[derive(Debug, Clone, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default)]
    pub thinking: Option<String>,
}

impl Ollama {
    /// Parse the response into ChatResponse.
    fn parse_response(response: OllamaChatResponse) -> ChatResponse {
        let stop_reason = match response.done_reason.as_deref() {
            Some("length") => StopReason::Length,
            // "stop", None, and any other value defaults to Stop
            _ => StopReason::Stop,
        };

        let tool_calls = response.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| {
                    let args = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                    ToolCall::function(
                        format!("call_{}", uuid::Uuid::new_v4()),
                        tc.function.name,
                        args,
                    )
                })
                .collect()
        });

        let content = if response.message.content.is_empty() {
            None
        } else {
            Some(Content::Text(response.message.content))
        };

        // Extract thinking content from reasoning models
        let reasoning_content = response.message.thinking.filter(|t| !t.is_empty());

        let message = Message {
            role: Role::Assistant,
            content,
            refusal: None,
            annotations: Vec::new(),
            tool_calls,
            tool_call_id: None,
            name: None,
            reasoning_content,
            thinking_blocks: None,
        };

        let usage = match (response.prompt_eval_count, response.eval_count) {
            (Some(input), Some(output)) => Some(Usage::new(input, output)),
            _ => None,
        };

        ChatResponse {
            message,
            stop_reason,
            usage,
            model: Some(response.model),
            id: None,
            service_tier: None,
            raw: None,
        }
    }
}

#[async_trait]
impl ChatProvider for Ollama {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let url = self.chat_url();
        let mut body = self.build_body(request);
        body.stream = false;

        let response = self.client().post(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let response_text = response.text().await?;
        let parsed: OllamaChatResponse = serde_json::from_str(&response_text).map_err(|e| {
            LlmError::response_format(
                "valid Ollama response",
                format!("parse error: {e}, response: {response_text}"),
            )
        })?;

        Ok(Self::parse_response(parsed))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = self.chat_url();
        let mut body = self.build_body(request);
        body.stream = true;

        let response = self.client().post(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let stream = response.bytes_stream();
        let parsed_stream = stream.flat_map(move |chunk_result| {
            let chunks: Vec<Result<StreamChunk>> = match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    text.lines().filter_map(parse_stream_line).collect()
                }
                Err(e) => vec![Err(LlmError::stream(e.to_string()).into())],
            };
            futures::stream::iter(chunks)
        });

        Ok(Box::pin(parsed_stream))
    }

    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    fn default_model(&self) -> &str {
        self.model()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }
}
