//! Ollama Chat Completions API implementation.

#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_fields_in_debug,
    clippy::match_same_arms,
    clippy::unused_self,
    clippy::unnecessary_wraps,
    clippy::unwrap_used,
    clippy::unnecessary_filter_map
)]

use super::client::OllamaClient;
use super::streaming::StreamingResponse;
use crate::error::AgentError;
use crate::message::{ChatMessage, ChatMessageToolCall, MessageRole};
use crate::providers::common::{GenerateOptions, Model, ModelResponse, ModelStream, TokenUsage};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tracing::{debug, instrument};

/// Ollama Chat Completion model.
///
/// Implements the [`Model`] trait for Ollama's Chat API.
#[derive(Clone)]
pub struct CompletionModel {
    client: OllamaClient,
    model_id: String,
    /// Default number of tokens to predict.
    pub num_predict: Option<u32>,
    /// Keep model loaded in memory.
    pub keep_alive: Option<String>,
}

impl std::fmt::Debug for CompletionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionModel")
            .field("model_id", &self.model_id)
            .field("num_predict", &self.num_predict)
            .finish()
    }
}

impl CompletionModel {
    /// Create a new completion model.
    pub(crate) fn new(client: OllamaClient, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
            num_predict: None,
            keep_alive: None,
        }
    }

    /// Set the number of tokens to predict.
    #[must_use]
    pub const fn with_num_predict(mut self, num_predict: u32) -> Self {
        self.num_predict = Some(num_predict);
        self
    }

    /// Set `keep_alive` duration (e.g., "5m", "1h", "-1" for indefinite).
    #[must_use]
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Build the request body for the API.
    fn build_request_body(&self, messages: &[ChatMessage], options: &GenerateOptions) -> Value {
        let api_messages: Vec<Value> = messages
            .iter()
            .filter_map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::ToolCall => "assistant",
                    MessageRole::ToolResponse => "tool",
                };

                let mut obj = serde_json::json!({ "role": role });

                // Content (optional for tool call messages)
                if let Some(content) = msg.text_content() {
                    obj["content"] = serde_json::json!(content);
                }

                // Tool calls - Ollama format requires type and index
                if let Some(tool_calls) = &msg.tool_calls {
                    let tc_json: Vec<Value> = tool_calls
                        .iter()
                        .enumerate()
                        .map(|(i, tc)| {
                            serde_json::json!({
                                "type": "function",
                                "function": {
                                    "index": i,
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            })
                        })
                        .collect();
                    obj["tool_calls"] = serde_json::json!(tc_json);
                }

                // Tool response requires tool_name field
                if msg.role == MessageRole::ToolResponse
                    && let Some(tool_call_id) = &msg.tool_call_id
                {
                    obj["tool_name"] = serde_json::json!(tool_call_id);
                }

                Some(obj)
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model_id,
            "messages": api_messages,
            "stream": false
        });

        // Options
        let mut opts = serde_json::Map::new();

        if let Some(temp) = options.temperature {
            opts.insert("temperature".to_string(), serde_json::json!(temp));
        }

        if let Some(top_p) = options.top_p {
            opts.insert("top_p".to_string(), serde_json::json!(top_p));
        }

        if let Some(max_tokens) = options.max_tokens.or(self.num_predict) {
            opts.insert("num_predict".to_string(), serde_json::json!(max_tokens));
        }

        if let Some(stop) = &options.stop_sequences
            && !stop.is_empty()
        {
            opts.insert("stop".to_string(), serde_json::json!(stop));
        }

        if !opts.is_empty() {
            body["options"] = Value::Object(opts);
        }

        // Keep alive
        if let Some(keep_alive) = &self.keep_alive {
            body["keep_alive"] = serde_json::json!(keep_alive);
        }

        // Tools
        if let Some(tools) = &options.tools
            && !tools.is_empty()
        {
            let tool_defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tool_defs);
        }

        body
    }

    /// Parse the API response into a `ModelResponse`.
    fn parse_response(&self, json: Value) -> Result<ModelResponse, AgentError> {
        let message_json = &json["message"];
        let content = message_json["content"].as_str().map(String::from);

        // Parse tool calls
        let tool_calls = if message_json["tool_calls"].is_array() {
            let tc_array = message_json["tool_calls"].as_array().unwrap();
            let calls: Vec<ChatMessageToolCall> = tc_array
                .iter()
                .enumerate()
                .filter_map(|(i, tc)| {
                    let name = tc["function"]["name"].as_str()?.to_string();
                    let arguments = tc["function"]["arguments"].clone();
                    Some(ChatMessageToolCall::new(
                        format!("call_{i}"),
                        name,
                        arguments,
                    ))
                })
                .collect();
            if calls.is_empty() { None } else { Some(calls) }
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
        let token_usage = if json.get("prompt_eval_count").is_some() {
            Some(TokenUsage {
                input_tokens: json["prompt_eval_count"].as_u64().unwrap_or(0) as u32,
                output_tokens: json["eval_count"].as_u64().unwrap_or(0) as u32,
            })
        } else {
            None
        };

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
        // Tool calling support varies by model
        // Models like llama3.1+, qwen2.5, mistral-nemo support it
        true
    }

    #[instrument(skip(self, messages, options), fields(model = %self.model_id))]
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError> {
        let body = self.build_request_body(&messages, &options);

        debug!("Sending request to Ollama API");

        let response = self
            .client
            .http_client
            .post(format!("{}/api/chat", self.client.base_url))
            .headers(self.client.headers())
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!(
                "Ollama API error ({status}): {error_text}"
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

        debug!("Sending streaming request to Ollama API");

        let response = self
            .client
            .http_client
            .post(format!("{}/api/chat", self.client.base_url))
            .headers(self.client.headers())
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!(
                "Ollama API error ({status}): {error_text}"
            )));
        }

        let stream = StreamingResponse::new(response.bytes_stream());
        Ok(Box::pin(stream))
    }
}

/// Ollama API error response.
#[derive(Debug, Deserialize)]
pub struct ApiError {
    /// Error message.
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id() {
        let client = OllamaClient::new();
        let model = client.completion_model("llama3.3");
        assert_eq!(model.model_id(), "llama3.3");
    }

    #[test]
    fn test_with_options() {
        let client = OllamaClient::new();
        let model = client
            .completion_model("llama3.3")
            .with_num_predict(2048)
            .with_keep_alive("10m");

        assert_eq!(model.num_predict, Some(2048));
        assert_eq!(model.keep_alive, Some("10m".to_string()));
    }
}
