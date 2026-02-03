//! Anthropic Messages API implementation.

#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_fields_in_debug,
    clippy::match_same_arms,
    clippy::unused_self,
    clippy::unwrap_used,
    clippy::unnecessary_wraps
)]

use super::client::AnthropicClient;
use super::streaming::StreamingResponse;
use crate::error::AgentError;
use crate::message::{ChatMessage, ChatMessageToolCall, MessageRole};
use crate::providers::common::{GenerateOptions, Model, ModelResponse, ModelStream, TokenUsage};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, instrument};

/// Anthropic Chat Completion model.
///
/// Implements the [`Model`] trait for Anthropic's Messages API.
#[derive(Clone)]
pub struct CompletionModel {
    client: AnthropicClient,
    model_id: String,
    /// Default max tokens for generation.
    pub default_max_tokens: u32,
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
    pub(crate) fn new(client: AnthropicClient, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
            default_max_tokens: 4096,
        }
    }

    /// Set default max tokens.
    #[must_use]
    pub const fn with_default_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Build the request body for the API.
    fn build_request_body(&self, messages: &[ChatMessage], options: &GenerateOptions) -> Value {
        // Extract system message and convert other messages
        let mut system_content = String::new();
        let mut api_messages: Vec<Value> = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    if let Some(text) = msg.text_content() {
                        if !system_content.is_empty() {
                            system_content.push('\n');
                        }
                        system_content.push_str(&text);
                    }
                }
                MessageRole::User => {
                    if let Some(text) = msg.text_content() {
                        api_messages.push(serde_json::json!({
                            "role": "user",
                            "content": text
                        }));
                    }
                }
                MessageRole::Assistant => {
                    if let Some(text) = msg.text_content() {
                        api_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": text
                        }));
                    }
                }
                MessageRole::ToolCall => {
                    // Anthropic uses tool_use content blocks
                    if let Some(tool_calls) = &msg.tool_calls {
                        let content: Vec<Value> = tool_calls
                            .iter()
                            .map(|tc| {
                                serde_json::json!({
                                    "type": "tool_use",
                                    "id": tc.id,
                                    "name": tc.function.name,
                                    "input": tc.function.arguments
                                })
                            })
                            .collect();
                        api_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": content
                        }));
                    }
                }
                MessageRole::ToolResponse => {
                    if let Some(text) = msg.text_content() {
                        api_messages.push(serde_json::json!({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id.as_deref().unwrap_or(""),
                                "content": text
                            }]
                        }));
                    }
                }
            }
        }

        let max_tokens = options.max_tokens.unwrap_or(self.default_max_tokens);

        let mut body = serde_json::json!({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": api_messages
        });

        // System prompt
        if !system_content.is_empty() {
            body["system"] = serde_json::json!(system_content);
        }

        // Temperature
        if let Some(temp) = options.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        // Top-p
        if let Some(top_p) = options.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }

        // Stop sequences
        if let Some(stop) = &options.stop_sequences
            && !stop.is_empty()
        {
            body["stop_sequences"] = serde_json::json!(stop);
        }

        // Tools
        if let Some(tools) = &options.tools
            && !tools.is_empty()
        {
            let tool_defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tool_defs);
        }

        body
    }

    /// Parse the API response into a `ModelResponse`.
    fn parse_response(&self, json: Value) -> Result<ModelResponse, AgentError> {
        let content = &json["content"];
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(blocks) = content.as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            if !text_content.is_empty() {
                                text_content.push('\n');
                            }
                            text_content.push_str(text);
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or_default().to_string();
                        let name = block["name"].as_str().unwrap_or_default().to_string();
                        let input = block["input"].clone();
                        tool_calls.push(ChatMessageToolCall::new(id, name, input));
                    }
                    _ => {}
                }
            }
        }

        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: if text_content.is_empty() {
                None
            } else {
                Some(vec![crate::message::MessageContent::text(text_content)])
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        };

        let token_usage = json.get("usage").map(|usage| TokenUsage {
            input_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as u32,
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
        true
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

        debug!("Sending request to Anthropic API");

        let response = self
            .client
            .http_client
            .post(format!("{}/v1/messages", self.client.base_url))
            .headers(self.client.auth_headers())
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!(
                "Anthropic API error ({status}): {error_text}"
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

        debug!("Sending streaming request to Anthropic API");

        let response = self
            .client
            .http_client
            .post(format!("{}/v1/messages", self.client.base_url))
            .headers(self.client.auth_headers())
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!(
                "Anthropic API error ({status}): {error_text}"
            )));
        }

        let stream = StreamingResponse::new(response.bytes_stream());
        Ok(Box::pin(stream))
    }
}

/// Anthropic API error response.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    /// Error type identifier.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Detailed error information.
    pub error: ApiError,
}

/// Anthropic API error details.
#[derive(Debug, Deserialize)]
pub struct ApiError {
    /// Error type identifier.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Human-readable error message.
    pub message: String,
}

/// Content block types in Anthropic responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content block.
    Text {
        /// The text content.
        text: String,
    },
    /// Tool use content block.
    ToolUse {
        /// Unique identifier for this tool use.
        id: String,
        /// Name of the tool being used.
        name: String,
        /// Input arguments for the tool.
        input: Value,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id() {
        let client = AnthropicClient::new("test-key");
        let model = client.completion_model("claude-3-5-sonnet-latest");
        assert_eq!(model.model_id(), "claude-3-5-sonnet-latest");
    }

    #[test]
    fn test_default_max_tokens() {
        let client = AnthropicClient::new("test-key");
        let model = client.completion_model("claude-3-5-sonnet-latest");
        assert_eq!(model.default_max_tokens, 4096);

        let model_updated = model.with_default_max_tokens(8192);
        assert_eq!(model_updated.default_max_tokens, 8192);
    }
}
