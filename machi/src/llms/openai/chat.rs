//! `OpenAI` `ChatProvider` implementation.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::Deserialize;
use tracing::{Instrument, debug, error, info, info_span};

use super::client::OpenAI;
use super::stream::parse_sse_events;
use crate::chat::ChatProvider;
use crate::chat::{ChatRequest, ChatResponse};
use crate::error::Result;
use crate::llms::LlmError;
use crate::message::{Content, Role, ToolCall};
use crate::stream::{StopReason, StreamChunk};
use crate::usage::Usage;

/// `OpenAI` chat completion response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Service tier used for processing.
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// `OpenAI` response choice.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIChoice {
    pub message: OpenAIResponseMessage,
    pub finish_reason: Option<String>,
}

/// `OpenAI` response message.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIResponseMessage {
    pub content: Option<String>,
    /// Refusal message if the model declined to respond.
    #[serde(default)]
    pub refusal: Option<String>,
    /// Tool calls deserialized directly into the core [`ToolCall`] type.
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl OpenAI {
    /// Parse the response into `ChatResponse`.
    fn parse_response(response: OpenAIChatResponse) -> Result<ChatResponse> {
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

        let content = choice.message.content.map(Content::Text);

        let message = crate::message::Message {
            role: Role::Assistant,
            content,
            refusal: choice.message.refusal,
            annotations: Vec::new(),
            tool_calls: choice.message.tool_calls,
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
            service_tier: response.service_tier,
            raw: None,
        })
    }
}

#[async_trait]
impl ChatProvider for OpenAI {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let span = info_span!(
            "gen_ai.chat",
            gen_ai.system = "openai",
            gen_ai.request.model = %request.model,
            gen_ai.request.temperature = request.temperature.unwrap_or(-1.0),
            gen_ai.request.max_tokens = request.max_completion_tokens.unwrap_or(0),
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.response.finish_reason = tracing::field::Empty,
            error = tracing::field::Empty,
        );

        async {
            let url = self.chat_url();
            let body = self.build_chat_body(request, false)?;

            debug!(model = %request.model, messages = request.messages.len(), "Sending OpenAI chat request");

            let response = self.build_request(&url).json(&body).send().await?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                let err = Self::parse_error(status.as_u16(), &error_text);
                error!(error = %err, status = status.as_u16(), "OpenAI API error");
                tracing::Span::current().record("error", tracing::field::display(&err));
                return Err(err.into());
            }

            let response_text = response.text().await?;
            let parsed: OpenAIChatResponse = serde_json::from_str(&response_text).map_err(|e| {
                let err = LlmError::response_format(
                    "valid OpenAI response",
                    format!("parse error: {e}, response: {response_text}"),
                );
                error!(error = %err, "OpenAI response parse error");
                tracing::Span::current().record("error", tracing::field::display(&err));
                err
            })?;

            let result = Self::parse_response(parsed)?;

            // Record usage, model, and finish_reason in the span.
            let current = tracing::Span::current();
            if let Some(ref usage) = result.usage {
                current.record("gen_ai.usage.input_tokens", usage.input_tokens);
                current.record("gen_ai.usage.output_tokens", usage.output_tokens);
            }
            if let Some(ref model) = result.model {
                current.record("gen_ai.response.model", model.as_str());
            }
            current.record("gen_ai.response.finish_reason", result.stop_reason.as_str());

            info!(
                model = result.model.as_deref().unwrap_or(&request.model),
                finish_reason = result.stop_reason.as_str(),
                "OpenAI chat completed",
            );

            Ok(result)
        }
        .instrument(span)
        .await
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        debug!(
            gen_ai.system = "openai",
            model = %request.model,
            messages = request.messages.len(),
            "Starting OpenAI chat stream",
        );

        let url = self.chat_url();
        let body = self.build_chat_body(request, true)?;

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
