//! Chat types, traits, and utilities for LLM operations.
//!
//! This module provides:
//! - [`ChatRequest`] — request parameters for chat completions
//! - [`ChatResponse`] — response from chat completions
//! - [`ChatProvider`] — core trait for LLM providers
//!
//! # Examples
//!
//! ```rust
//! use machi::chat::{ChatRequest, ChatResponse};
//!
//! let request = ChatRequest::new("gpt-4o")
//!     .system("You are helpful.")
//!     .user("Hello!")
//!     .max_tokens(100)
//!     .temperature(0.7);
//!
//! let response = ChatResponse::from_text("Hello! How can I help?");
//! assert_eq!(response.text().unwrap(), "Hello! How can I help?");
//! ```

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::message::Message;
use crate::stream::{StopReason, StreamChunk};
use crate::tool::ToolDefinition;
use crate::usage::Usage;

/// Reasoning effort level for o-series models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No reasoning (gpt-5.1+ only).
    None,
    /// Minimal reasoning effort.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort (default for most models).
    #[default]
    Medium,
    /// High reasoning effort.
    High,
    /// Extra high reasoning effort (gpt-5.1-codex-max+).
    #[serde(rename = "xhigh")]
    XHigh,
}

impl ReasoningEffort {
    /// Returns the string representation for the API.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

/// A chat completion request to an LLM.
///
/// # `OpenAI` API Alignment
/// This struct aligns with `OpenAI`'s Chat Completions API parameters.
/// Some fields are provider-specific and may be ignored by other backends.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
    #[serde(default)]
    pub model: String,

    /// Conversation messages.
    #[serde(default)]
    pub messages: Vec<Message>,

    /// Maximum tokens to generate (deprecated, use `max_completion_tokens`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Maximum completion tokens (preferred over `max_tokens` for newer models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Sampling temperature (0.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number of completions to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Tools available for the model to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Controls how the model uses tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,

    /// Whether to enable parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,

    /// Response format specification (for JSON mode / structured outputs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Random seed for reproducibility (deprecated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// User identifier for tracking and abuse detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Frequency penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Whether to return log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Number of top log probabilities to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Service tier for processing (e.g., "default", "flex").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Whether to store the completion for later retrieval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Metadata key-value pairs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,

    /// Reasoning effort for o-series models (none, minimal, low, medium, high, xhigh).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ChatRequest {
    /// Creates a new request with the specified model.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Creates a request with messages.
    #[must_use]
    pub fn with_messages(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            ..Default::default()
        }
    }

    /// Adds a system message.
    #[must_use]
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::system(content));
        self
    }

    /// Adds a user message.
    #[must_use]
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::user(content));
        self
    }

    /// Adds an assistant message.
    #[must_use]
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(content));
        self
    }

    /// Adds a message.
    #[must_use]
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Sets all messages.
    #[must_use]
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Sets max tokens (legacy, prefer `max_completion_tokens`).
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets max completion tokens (preferred for newer models).
    #[must_use]
    pub const fn max_completion_tokens(mut self, tokens: u32) -> Self {
        self.max_completion_tokens = Some(tokens);
        self
    }

    /// Sets temperature.
    #[must_use]
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets `top_p`.
    #[must_use]
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets number of completions to generate.
    #[must_use]
    pub const fn n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }

    /// Sets stop sequences.
    #[must_use]
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Sets tools.
    #[must_use]
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets tool choice.
    #[must_use]
    pub fn tool_choice(mut self, choice: impl Into<ToolChoice>) -> Self {
        self.tool_choice = Some(choice.into().to_value());
        self
    }

    /// Enables or disables parallel tool calls.
    #[must_use]
    pub const fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Enables streaming.
    #[must_use]
    pub const fn stream(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Sets response format.
    #[must_use]
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Sets structured output by inferring the JSON Schema from a Rust type.
    ///
    /// This is the most ergonomic way to request structured JSON output from
    /// the LLM. The type must derive [`schemars::JsonSchema`].
    ///
    /// The response can be deserialized with [`ChatResponse::parse`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    /// use machi::chat::ChatRequest;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Country {
    ///     name: String,
    ///     capital: String,
    ///     population: u64,
    /// }
    ///
    /// let request = ChatRequest::new("gpt-4o")
    ///     .user("Tell me about France.")
    ///     .output_type::<Country>();
    ///
    /// assert!(request.response_format.is_some());
    /// ```
    #[cfg(feature = "schema")]
    #[must_use]
    pub fn output_type<T: schemars::JsonSchema>(self) -> Self {
        self.response_format(ResponseFormat::from_type::<T>())
    }

    /// Sets seed for reproducibility.
    #[must_use]
    pub const fn seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets user identifier.
    #[must_use]
    pub fn user_id(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Sets frequency penalty.
    #[must_use]
    pub const fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Sets presence penalty.
    #[must_use]
    pub const fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Enables log probabilities.
    #[must_use]
    pub const fn logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    /// Sets service tier.
    #[must_use]
    pub fn service_tier(mut self, tier: impl Into<String>) -> Self {
        self.service_tier = Some(tier.into());
        self
    }

    /// Sets reasoning effort for o-series models.
    #[must_use]
    pub const fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
}

/// Controls how the model uses tools.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Model decides whether to use tools.
    #[default]
    Auto,
    /// Model must use at least one tool.
    Required,
    /// Model cannot use any tools.
    None,
    /// Model must use the specified function.
    Function(String),
}

impl ToolChoice {
    /// Converts to JSON value for serialization.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".to_owned()),
            Self::Required => Value::String("required".to_owned()),
            Self::None => Value::String("none".to_owned()),
            Self::Function(name) => serde_json::json!({
                "type": "function",
                "function": {"name": name}
            }),
        }
    }
}

impl From<&str> for ToolChoice {
    fn from(s: &str) -> Self {
        match s {
            "auto" => Self::Auto,
            "required" => Self::Required,
            "none" => Self::None,
            name => Self::Function(name.to_owned()),
        }
    }
}

/// Response format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text response.
    Text,
    /// JSON object response.
    JsonObject,
    /// JSON response with schema (structured outputs).
    JsonSchema {
        /// Schema definition.
        json_schema: JsonSchemaSpec,
    },
}

impl ResponseFormat {
    /// Creates a JSON object format.
    #[must_use]
    pub const fn json() -> Self {
        Self::JsonObject
    }

    /// Creates a JSON schema format.
    #[must_use]
    pub fn json_schema(name: impl Into<String>, schema: Value) -> Self {
        Self::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: name.into(),
                schema,
                strict: Some(true),
            },
        }
    }

    /// Creates a JSON schema format by auto-generating the schema from a Rust type.
    ///
    /// The type must derive [`schemars::JsonSchema`]. The schema name is
    /// derived from the type name automatically.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use schemars::JsonSchema;
    /// use machi::chat::ResponseFormat;
    ///
    /// #[derive(JsonSchema)]
    /// struct Country { name: String, capital: String }
    ///
    /// let format = ResponseFormat::from_type::<Country>();
    /// ```
    #[cfg(feature = "schema")]
    #[must_use]
    pub fn from_type<T: schemars::JsonSchema>() -> Self {
        let (name, schema_value) = generate_json_schema::<T>();
        Self::json_schema(name, schema_value)
    }
}

/// Generate a JSON Schema from a Rust type that implements [`schemars::JsonSchema`].
///
/// Returns `(name, schema)` where `name` is derived from the type name and
/// `schema` is the JSON Schema definition with the `$schema` meta field removed
/// (LLM APIs don't need it).
///
/// This is the single source of truth for schema generation, used by both
/// [`ResponseFormat::from_type`] and [`OutputSchema::from_type`](crate::agent::OutputSchema::from_type).
#[cfg(feature = "schema")]
#[must_use]
pub fn generate_json_schema<T: schemars::JsonSchema>() -> (String, Value) {
    let root = schemars::schema_for!(T);
    let mut schema_value = serde_json::to_value(&root).unwrap_or_default();

    // Remove the $schema meta field — LLM APIs don't need it.
    if let Value::Object(ref mut map) = schema_value {
        map.remove("$schema");
    }

    let name = <T as schemars::JsonSchema>::schema_name();
    (name.into_owned(), schema_value)
}

/// JSON schema specification for structured outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaSpec {
    /// Schema name.
    pub name: String,
    /// JSON Schema definition.
    pub schema: Value,
    /// Whether to enforce strict validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A chat completion response from an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The generated message.
    pub message: Message,

    /// Why the model stopped generating.
    pub stop_reason: StopReason,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// Model identifier used for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Unique completion ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Service tier used for processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Raw response from the provider (for debugging).
    #[serde(skip)]
    pub raw: Option<Value>,
}

impl ChatResponse {
    /// Creates a new response with a message.
    #[must_use]
    pub const fn new(message: Message) -> Self {
        Self {
            message,
            stop_reason: StopReason::Stop,
            usage: None,
            model: None,
            id: None,
            service_tier: None,
            raw: None,
        }
    }

    /// Creates a response from text content.
    #[must_use]
    pub fn from_text(content: impl Into<String>) -> Self {
        Self::new(Message::assistant(content))
    }

    /// Sets the stop reason.
    #[must_use]
    pub const fn with_stop_reason(mut self, reason: StopReason) -> Self {
        self.stop_reason = reason;
        self
    }

    /// Sets usage statistics.
    #[must_use]
    pub const fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Sets the model identifier.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the completion ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the raw response.
    #[must_use]
    pub fn with_raw(mut self, raw: Value) -> Self {
        self.raw = Some(raw);
        self
    }

    /// Returns the text content of the response.
    #[must_use]
    pub fn text(&self) -> Option<String> {
        self.message.text()
    }

    /// Deserialize the response text into a concrete Rust type.
    ///
    /// This is the companion to [`ChatRequest::output_type`] and
    /// [`ChatRequest::response_format`]. When the LLM produces structured
    /// JSON output, this method parses the text content directly into `T`.
    ///
    /// # Errors
    ///
    /// Returns [`serde_json::Error`] if the response has no text content
    /// or if the text cannot be deserialized into `T`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use machi::chat::ChatResponse;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct Info { name: String }
    ///
    /// let response = ChatResponse::from_text(r#"{"name":"Rust"}"#);
    /// let info: Info = response.parse().unwrap();
    /// assert_eq!(info.name, "Rust");
    /// ```
    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> serde_json::Result<T> {
        let text = self.text().unwrap_or_default();
        serde_json::from_str(&text)
    }

    /// Returns `true` if the response contains tool calls.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.message.has_tool_calls()
    }

    /// Returns the tool calls if present.
    #[must_use]
    pub fn tool_calls(&self) -> Option<&[crate::message::ToolCall]> {
        self.message.tool_calls.as_deref()
    }

    /// Returns `true` if the model completed normally.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.stop_reason.is_complete()
    }

    /// Returns `true` if the response was truncated due to length.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.stop_reason.is_truncated()
    }
}

impl Default for ChatResponse {
    fn default() -> Self {
        Self::new(Message::default())
    }
}

/// Trait for providers that support chat completions.
///
/// This is the primary trait that all LLM backends must implement to support
/// chat-based interactions with language models.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Send a chat completion request and receive a complete response.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat request containing messages, tools, and parameters
    ///
    /// # Returns
    ///
    /// A `ChatResponse` containing the model's response, or an error.
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse>;

    /// Send a chat completion request and receive a streaming response.
    ///
    /// This method returns a stream of `StreamChunk` values that can be processed
    /// as they arrive, enabling real-time display of generated content.
    ///
    /// # Arguments
    ///
    /// * `request` - The chat request (the `stream` field will be set automatically)
    ///
    /// # Returns
    ///
    /// A stream of `StreamChunk` values, or an error if the request fails.
    ///
    /// # Default Implementation
    ///
    /// By default, this method returns an error indicating streaming is not supported.
    /// Providers should override this if they support streaming.
    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let _ = request;
        Err(crate::error::LlmError::not_supported("streaming").into())
    }

    /// Get the name of this provider.
    ///
    /// Used for error messages and logging.
    fn provider_name(&self) -> &'static str;

    /// Get the default model for this provider.
    fn default_model(&self) -> &str;

    /// Check if this provider supports streaming.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if this provider supports tool/function calling.
    fn supports_tools(&self) -> bool {
        true
    }

    /// Check if this provider supports vision (image inputs).
    fn supports_vision(&self) -> bool {
        false
    }

    /// Check if this provider supports JSON mode / structured outputs.
    fn supports_json_mode(&self) -> bool {
        false
    }
}

/// Extension trait for `ChatProvider` with convenience methods.
#[async_trait]
pub trait ChatProviderExt: ChatProvider {
    /// Send a simple text message and get a text response.
    ///
    /// This is a convenience method for simple one-shot interactions.
    async fn complete(&self, prompt: &str) -> Result<String> {
        let request = ChatRequest::new(self.default_model()).user(prompt);
        let response = self.chat(&request).await?;
        Ok(response.text().unwrap_or_default())
    }

    /// Send a message with a system prompt.
    async fn complete_with_system(&self, system: &str, prompt: &str) -> Result<String> {
        let request = ChatRequest::new(self.default_model())
            .system(system)
            .user(prompt);
        let response = self.chat(&request).await?;
        Ok(response.text().unwrap_or_default())
    }

    /// Send a message with a custom model.
    async fn complete_with_model(&self, model: &str, prompt: &str) -> Result<String> {
        let request = ChatRequest::new(model).user(prompt);
        let response = self.chat(&request).await?;
        Ok(response.text().unwrap_or_default())
    }
}

// Blanket implementation for all ChatProviders
impl<T: ChatProvider> ChatProviderExt for T {}

/// Type alias for an Arc-wrapped `ChatProvider`.
pub type SharedChatProvider = std::sync::Arc<dyn ChatProvider>;
