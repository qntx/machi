//! `OpenAI` Chat Completions API implementation.

#![allow(
    clippy::cast_possible_truncation,
    clippy::or_fun_call,
    clippy::default_trait_access,
    clippy::option_if_let_else,
    clippy::unused_self,
    clippy::unwrap_used,
    clippy::missing_fields_in_debug,
    clippy::match_same_arms
)]

use super::client::OpenAIClient;
use super::streaming::StreamingResponse;
use crate::error::AgentError;
use crate::message::{ChatMessage, ChatMessageToolCall, MessageRole};
use crate::providers::common::{
    GenerateOptions, Model, ModelResponse, ModelStream, TokenUsage,
    model_requires_max_completion_tokens, model_supports_stop_parameter, saturating_u32,
};
use crate::tool::ToolDefinition;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tracing::{debug, instrument};

/// `OpenAI` Chat Completion model.
///
/// Implements the [`Model`] trait for `OpenAI`'s Chat Completions API.
#[derive(Clone)]
pub struct CompletionModel {
    client: OpenAIClient,
    model_id: String,
    /// Default max tokens for generation.
    pub default_max_tokens: Option<u32>,
}

impl std::fmt::Debug for CompletionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionModel")
            .field("model_id", &self.model_id)
            .field("default_max_tokens", &self.default_max_tokens)
            .finish()
    }
}

impl CompletionModel {
    /// Create a new completion model.
    pub(crate) fn new(client: OpenAIClient, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
            default_max_tokens: None,
        }
    }

    /// Set default max tokens.
    #[must_use]
    pub const fn with_default_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = Some(max_tokens);
        self
    }

    /// Convert MessageContent to OpenAI API format.
    fn convert_content_to_openai(content: &crate::message::MessageContent) -> Value {
        use crate::message::MessageContent;
        match content {
            MessageContent::Text { text } => serde_json::json!({
                "type": "text",
                "text": text
            }),
            MessageContent::Image { image, .. } => serde_json::json!({
                "type": "image_url",
                "image_url": { "url": format!("data:image/jpeg;base64,{}", image) }
            }),
            MessageContent::ImageUrl { image_url } => serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": image_url.url,
                    "detail": image_url.detail.as_deref().unwrap_or("auto")
                }
            }),
            MessageContent::Audio { audio, format, .. } => {
                let fmt = format.as_deref().unwrap_or("wav");
                serde_json::json!({
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio,
                        "format": fmt
                    }
                })
            }
        }
    }

    /// Build the request body for the API.
    fn build_request_body(&self, messages: &[ChatMessage], options: &GenerateOptions) -> Value {
        let mut body = serde_json::json!({
            "model": self.model_id,
            "messages": self.convert_messages(messages),
        });

        // Temperature
        if let Some(temp) = options.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        // Max tokens - use max_completion_tokens for o-series and gpt-5 models
        let max_tokens = options.max_tokens.or(self.default_max_tokens);
        if let Some(max) = max_tokens {
            if model_requires_max_completion_tokens(&self.model_id) {
                body["max_completion_tokens"] = serde_json::json!(max);
            } else {
                body["max_tokens"] = serde_json::json!(max);
            }
        }

        // Top-p
        if let Some(top_p) = options.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }

        // Stop sequences (only if model supports it)
        if let Some(stop) = &options.stop_sequences
            && !stop.is_empty()
            && self.supports_stop_parameter()
        {
            body["stop"] = serde_json::json!(stop);
        }

        // Tools
        if let Some(tools) = &options.tools
            && !tools.is_empty()
        {
            let tool_defs: Vec<Value> =
                tools.iter().map(ToolDefinition::to_openai_format).collect();
            body["tools"] = serde_json::json!(tool_defs);
        }

        // Response format
        if let Some(format) = &options.response_format {
            body["response_format"] = format.clone();
        }

        body
    }

    /// Convert `ChatMessage` to `OpenAI` API format.
    fn convert_messages(&self, messages: &[ChatMessage]) -> Vec<Value> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::ToolCall => "assistant",
                    MessageRole::ToolResponse => "tool",
                };

                let mut obj = serde_json::json!({ "role": role });

                // Content - check for multimodal content first
                if let Some(contents) = &msg.content {
                    let has_media = contents.iter().any(|c| c.is_image() || c.is_audio());
                    if has_media {
                        // Multimodal: convert to array format
                        let content_array: Vec<Value> = contents
                            .iter()
                            .map(Self::convert_content_to_openai)
                            .collect();
                        obj["content"] = serde_json::json!(content_array);
                    } else if let Some(text) = msg.text_content() {
                        obj["content"] = serde_json::json!(text);
                    }
                } else if let Some(text) = msg.text_content() {
                    obj["content"] = serde_json::json!(text);
                }

                // Tool calls
                if let Some(tool_calls) = &msg.tool_calls {
                    let tc_json: Vec<Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.arguments_string()
                                }
                            })
                        })
                        .collect();
                    obj["tool_calls"] = serde_json::json!(tc_json);
                }

                // Tool call ID (for tool responses)
                if let Some(tool_call_id) = &msg.tool_call_id {
                    obj["tool_call_id"] = serde_json::json!(tool_call_id);
                }

                obj
            })
            .collect()
    }

    /// Parse the API response into a `ModelResponse`.
    fn parse_response(&self, json: Value) -> Result<ModelResponse, AgentError> {
        let choice = json["choices"]
            .get(0)
            .ok_or_else(|| AgentError::model("No choices in response"))?;

        let message_json = &choice["message"];
        let content = message_json["content"].as_str().map(String::from);

        // Parse tool calls
        let tool_calls = if message_json["tool_calls"].is_array() {
            let tc_array = message_json["tool_calls"].as_array().unwrap();
            let calls: Result<Vec<ChatMessageToolCall>, _> = tc_array
                .iter()
                .map(|tc| {
                    let id = tc["id"].as_str().unwrap_or_default().to_string();
                    let name = tc["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    let arguments = if let Some(args_str) = tc["function"]["arguments"].as_str() {
                        serde_json::from_str(args_str).unwrap_or(Value::Object(Default::default()))
                    } else {
                        tc["function"]["arguments"].clone()
                    };
                    Ok::<_, AgentError>(ChatMessageToolCall::new(id, name, arguments))
                })
                .collect();
            Some(calls?)
        } else {
            None
        };

        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: content.map(|c| vec![crate::message::MessageContent::text(c)]),
            tool_calls,
            tool_call_id: None,
        };

        // Parse token usage
        let token_usage = json.get("usage").map(|usage| TokenUsage {
            input_tokens: saturating_u32(usage["prompt_tokens"].as_u64().unwrap_or(0)),
            output_tokens: saturating_u32(usage["completion_tokens"].as_u64().unwrap_or(0)),
        });

        Ok(ModelResponse {
            message,
            token_usage,
            raw: Some(json),
        })
    }
}

#[async_trait]
impl Model for CompletionModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn supports_stop_parameter(&self) -> bool {
        model_supports_stop_parameter(&self.model_id)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tool_calling(&self) -> bool {
        true
    }

    #[instrument(skip(self, messages, options), fields(model = %self.model_id))]
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError> {
        let body = self.build_request_body(&messages, &options);

        debug!("Sending request to OpenAI API");

        let response = self
            .client
            .http_client
            .post(format!("{}/chat/completions", self.client.base_url))
            .headers(self.client.auth_headers())
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!(
                "OpenAI API error ({status}): {error_text}"
            )));
        }

        let json: Value = response.json().await?;
        self.parse_response(json)
    }

    #[instrument(skip(self, messages, options), fields(model = %self.model_id))]
    async fn generate_stream(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelStream, AgentError> {
        let mut body = self.build_request_body(&messages, &options);
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({"include_usage": true});

        debug!("Sending streaming request to OpenAI API");

        let response = self
            .client
            .http_client
            .post(format!("{}/chat/completions", self.client.base_url))
            .headers(self.client.auth_headers())
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!(
                "OpenAI API error ({status}): {error_text}"
            )));
        }

        let stream = StreamingResponse::new(response.bytes_stream());
        Ok(Box::pin(stream))
    }
}

/// `OpenAI` API error response.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub struct ApiErrorResponse {
    /// Detailed error information.
    pub error: ApiError,
}

/// `OpenAI` API error details.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub struct ApiError {
    /// Human-readable error message.
    pub message: String,
    /// Error type identifier.
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    /// Error code.
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id() {
        let client = OpenAIClient::new("test-key");
        let model = client.completion_model("gpt-4o");
        assert_eq!(model.model_id(), "gpt-4o");
    }

    #[test]
    fn test_supports_stop() {
        let client = OpenAIClient::new("test-key");

        let gpt4 = client.completion_model("gpt-4o");
        assert!(gpt4.supports_stop_parameter());

        let o3 = client.completion_model("o3");
        assert!(!o3.supports_stop_parameter());

        let o3_mini = client.completion_model("o3-mini");
        assert!(o3_mini.supports_stop_parameter());
    }
}
