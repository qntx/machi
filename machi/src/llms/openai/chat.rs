//! OpenAI ChatProvider implementation.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::chat::ChatProvider;
use crate::chat::{ChatRequest, ChatResponse};
use crate::error::{LlmError, Result};
use crate::message::{Content, Role, ToolCall as MsgToolCall};
use crate::stream::{StopReason, StreamChunk};

use super::client::OpenAI;
use super::stream::parse_sse_events;
use super::types::{OpenAIChatResponse, StreamOptions};

impl OpenAI {
    /// Parse the response into ChatResponse.
    pub(crate) fn parse_response(response: OpenAIChatResponse) -> Result<ChatResponse> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::response_format("at least one choice", "empty choices"))?;

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("length") => StopReason::Length,
            Some("tool_calls") => StopReason::ToolCalls,
            Some("content_filter") => StopReason::ContentFilter,
            // "stop", None, and any other value defaults to Stop
            _ => StopReason::Stop,
        };

        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| MsgToolCall::function(tc.id, tc.function.name, tc.function.arguments))
                .collect()
        });

        let content = choice.message.content.map(Content::Text);

        let message = crate::message::Message {
            role: Role::Assistant,
            content,
            refusal: None,
            annotations: Vec::new(),
            tool_calls,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking_blocks: None,
        };

        Ok(ChatResponse {
            message,
            stop_reason,
            usage: response.usage,
            model: Some(response.model),
            id: Some(response.id),
            service_tier: None,
            raw: None,
        })
    }
}

#[async_trait]
impl ChatProvider for OpenAI {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let url = self.chat_url();
        let body = self.build_body(request);

        let response = self.build_request(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let response_text = response.text().await?;
        let parsed: OpenAIChatResponse = serde_json::from_str(&response_text).map_err(|e| {
            LlmError::response_format(
                "valid OpenAI response",
                format!("parse error: {e}, response: {response_text}"),
            )
        })?;

        Self::parse_response(parsed)
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = self.chat_url();
        let mut body = self.build_body(request);
        body.stream = true;
        body.stream_options = Some(StreamOptions {
            include_usage: true,
        });

        let response = self.build_request(&url).json(&body).send().await?;

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
                    parse_sse_events(&text)
                }
                Err(e) => vec![Err(LlmError::stream(e.to_string()).into())],
            };
            futures::stream::iter(chunks)
        });

        Ok(Box::pin(parsed_stream))
    }

    fn provider_name(&self) -> &'static str {
        "openai"
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
