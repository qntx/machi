//! Model trait and LLM integrations.
//!
//! This module defines the interface for language model implementations
//! and provides common model types.

use crate::error::AgentError;
use crate::memory::TokenUsage;
use crate::message::{ChatMessage, ChatMessageStreamDelta};
use crate::tool::ToolDefinition;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;

/// Response from a model generation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The generated message.
    pub message: ChatMessage,
    /// Token usage information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
    /// Raw response from the API.
    #[serde(skip)]
    pub raw: Option<Value>,
}

impl ModelResponse {
    /// Create a new model response.
    #[must_use]
    pub fn new(message: ChatMessage) -> Self {
        Self {
            message,
            token_usage: None,
            raw: None,
        }
    }

    /// Set token usage.
    #[must_use]
    pub fn with_token_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = Some(usage);
        self
    }

    /// Set raw response.
    #[must_use]
    pub fn with_raw(mut self, raw: Value) -> Self {
        self.raw = Some(raw);
        self
    }

    /// Get the text content of the response.
    #[must_use]
    pub fn text(&self) -> Option<String> {
        self.message.text_content()
    }
}

/// Stream of model response deltas.
pub type ModelStream =
    Pin<Box<dyn Stream<Item = Result<ChatMessageStreamDelta, AgentError>> + Send>>;

/// Options for model generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Stop sequences to end generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Available tools for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Temperature for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-p sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Response format (e.g., JSON mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
}

impl GenerateOptions {
    /// Create new generate options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set stop sequences.
    #[must_use]
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Set tools.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens.
    #[must_use]
    pub const fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }
}

/// The core trait for language model implementations.
///
/// Models are responsible for generating responses based on conversation history
/// and optionally making tool calls.
#[async_trait]
pub trait Model: Send + Sync {
    /// Get the model identifier.
    fn model_id(&self) -> &str;

    /// Generate a response for the given messages.
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError>;

    /// Generate a streaming response.
    ///
    /// Default implementation falls back to non-streaming generate.
    async fn generate_stream(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelStream, AgentError> {
        let response = self.generate(messages, options).await?;
        let delta = ChatMessageStreamDelta {
            content: response.message.text_content(),
            tool_calls: None,
        };
        Ok(Box::pin(futures::stream::once(async move { Ok(delta) })))
    }

    /// Check if the model supports the stop parameter.
    fn supports_stop_parameter(&self) -> bool {
        true
    }

    /// Check if the model supports streaming.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if the model supports tool calling.
    fn supports_tool_calling(&self) -> bool {
        true
    }
}

/// A simple mock model for testing.
#[derive(Debug)]
pub struct MockModel {
    model_id: String,
    responses: Vec<String>,
    response_index: std::sync::atomic::AtomicUsize,
}

impl MockModel {
    /// Create a new mock model with predefined responses.
    #[must_use]
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            model_id: "mock-model".to_string(),
            responses,
            response_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Model for MockModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn generate(
        &self,
        _messages: Vec<ChatMessage>,
        _options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError> {
        let index = self
            .response_index
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let response = self
            .responses
            .get(index % self.responses.len())
            .cloned()
            .unwrap_or_else(|| "No response".to_string());

        Ok(ModelResponse::new(ChatMessage::assistant(response)))
    }
}

/// OpenAI-compatible model implementation.
#[derive(Debug, Clone)]
pub struct OpenAIModel {
    /// Model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
    pub model_id: String,
    /// API key for authentication.
    api_key: String,
    /// Base URL for the API.
    pub base_url: String,
    /// HTTP client.
    client: reqwest::Client,
}

impl OpenAIModel {
    /// Create a new OpenAI model.
    ///
    /// # Panics
    ///
    /// Panics if OPENAI_API_KEY environment variable is not set.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key =
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");
        Self::with_api_key(model_id, api_key)
    }

    /// Create a new OpenAI model with explicit API key.
    #[must_use]
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Set custom base URL (for Azure, local models, etc.).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Build the request body for the API.
    fn build_request_body(&self, messages: &[ChatMessage], options: &GenerateOptions) -> Value {
        let mut body = serde_json::json!({
            "model": self.model_id,
            "messages": self.convert_messages(messages),
        });

        if let Some(temp) = options.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(max) = options.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(top_p) = options.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(stop) = &options.stop_sequences {
            if !stop.is_empty() && self.supports_stop_parameter() {
                body["stop"] = serde_json::json!(stop);
            }
        }
        if let Some(tools) = &options.tools {
            if !tools.is_empty() {
                let tool_defs: Vec<_> =
                    tools.iter().map(ToolDefinition::to_openai_format).collect();
                body["tools"] = serde_json::json!(tool_defs);
            }
        }
        if let Some(format) = &options.response_format {
            body["response_format"] = format.clone();
        }

        body
    }

    /// Convert ChatMessage to OpenAI API format.
    fn convert_messages(&self, messages: &[ChatMessage]) -> Vec<Value> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    crate::message::MessageRole::System => "system",
                    crate::message::MessageRole::User => "user",
                    crate::message::MessageRole::Assistant => "assistant",
                    crate::message::MessageRole::ToolCall => "assistant",
                    crate::message::MessageRole::ToolResponse => "tool",
                };

                let mut obj = serde_json::json!({ "role": role });

                if let Some(content) = msg.text_content() {
                    obj["content"] = serde_json::json!(content);
                }

                if let Some(tool_calls) = &msg.tool_calls {
                    obj["tool_calls"] = serde_json::to_value(tool_calls).unwrap_or_default();
                }

                if let Some(tool_call_id) = &msg.tool_call_id {
                    obj["tool_call_id"] = serde_json::json!(tool_call_id);
                }

                obj
            })
            .collect()
    }
}

#[async_trait]
impl Model for OpenAIModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn supports_stop_parameter(&self) -> bool {
        // Some models (like o3, o4, gpt-5) don't support stop
        let model = self.model_id.as_str();
        !model.starts_with("o3") && !model.starts_with("o4") && !model.starts_with("gpt-5")
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError> {
        let body = self.build_request_body(&messages, &options);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!("API error: {error_text}")));
        }

        let json: Value = response.json().await?;

        // Parse the response
        let choice = json["choices"]
            .get(0)
            .ok_or_else(|| AgentError::model("No choices in response"))?;

        let message_json = &choice["message"];
        let content = message_json["content"].as_str().map(String::from);
        let tool_calls = if message_json["tool_calls"].is_array() {
            Some(serde_json::from_value(message_json["tool_calls"].clone())?)
        } else {
            None
        };

        let message = ChatMessage {
            role: crate::message::MessageRole::Assistant,
            content: content.map(|c| vec![crate::message::MessageContent::text(c)]),
            tool_calls,
            tool_call_id: None,
        };

        let token_usage = if let Some(usage) = json.get("usage") {
            Some(TokenUsage {
                input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as u32,
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

/// Anthropic Claude model implementation.
#[derive(Debug, Clone)]
pub struct AnthropicModel {
    /// Model identifier (e.g., "claude-3-5-sonnet-20241022").
    pub model_id: String,
    /// API key for authentication.
    api_key: String,
    /// Base URL for the API.
    pub base_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
}

impl AnthropicModel {
    /// Create a new Anthropic model.
    ///
    /// # Panics
    ///
    /// Panics if ANTHROPIC_API_KEY environment variable is not set.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY environment variable not set");
        Self::with_api_key(model_id, api_key)
    }

    /// Create a new Anthropic model with explicit API key.
    #[must_use]
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            client: reqwest::Client::new(),
            max_tokens: 4096,
        }
    }

    /// Set custom base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set max tokens.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Build the request body for the API.
    fn build_request_body(&self, messages: &[ChatMessage], options: &GenerateOptions) -> Value {
        // Extract system message and convert other messages
        let mut system_content = String::new();
        let mut api_messages = Vec::new();

        for msg in messages {
            match msg.role {
                crate::message::MessageRole::System => {
                    if let Some(text) = msg.text_content() {
                        system_content.push_str(&text);
                    }
                }
                crate::message::MessageRole::User => {
                    if let Some(text) = msg.text_content() {
                        api_messages.push(serde_json::json!({
                            "role": "user",
                            "content": text
                        }));
                    }
                }
                crate::message::MessageRole::Assistant => {
                    if let Some(text) = msg.text_content() {
                        api_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": text
                        }));
                    }
                }
                crate::message::MessageRole::ToolCall => {
                    // Anthropic uses tool_use content blocks
                    if let Some(tool_calls) = &msg.tool_calls {
                        let mut content = Vec::new();
                        for tc in tool_calls {
                            content.push(serde_json::json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": tc.function.arguments
                            }));
                        }
                        api_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": content
                        }));
                    }
                }
                crate::message::MessageRole::ToolResponse => {
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

        let max_tokens = options.max_tokens.unwrap_or(self.max_tokens);

        let mut body = serde_json::json!({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": api_messages
        });

        if !system_content.is_empty() {
            body["system"] = serde_json::json!(system_content);
        }

        if let Some(temp) = options.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = options.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(stop) = &options.stop_sequences {
            if !stop.is_empty() {
                body["stop_sequences"] = serde_json::json!(stop);
            }
        }
        if let Some(tools) = &options.tools {
            if !tools.is_empty() {
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
        }

        body
    }
}

#[async_trait]
impl Model for AnthropicModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError> {
        let body = self.build_request_body(&messages, &options);

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::model(format!(
                "Anthropic API error: {error_text}"
            )));
        }

        let json: Value = response.json().await?;

        // Parse the response
        let content = &json["content"];
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(blocks) = content.as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            text_content.push_str(text);
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or_default().to_string();
                        let name = block["name"].as_str().unwrap_or_default().to_string();
                        let input = block["input"].clone();
                        tool_calls.push(crate::message::ChatMessageToolCall::new(id, name, input));
                    }
                    _ => {}
                }
            }
        }

        let message = ChatMessage {
            role: crate::message::MessageRole::Assistant,
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

        let token_usage = if json.get("usage").is_some() {
            Some(TokenUsage {
                input_tokens: json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
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
