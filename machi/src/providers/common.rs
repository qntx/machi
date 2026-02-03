//! Common types and traits for all providers.
//!
//! This module defines the core abstractions that all provider implementations
//! must satisfy, ensuring a consistent interface across different LLM APIs.

use crate::error::AgentError;
use crate::message::{ChatMessage, ChatMessageStreamDelta, ChatMessageToolCall};
use crate::tool::ToolDefinition;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// Token usage information from a model response.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    /// Number of tokens in the input/prompt.
    pub input_tokens: u32,
    /// Number of tokens in the output/completion.
    pub output_tokens: u32,
}

impl TokenUsage {
    /// Create new token usage with specified counts.
    #[must_use]
    pub const fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    /// Get total token count.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
        }
    }
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
    }
}

/// Response from a model generation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The generated message.
    pub message: ChatMessage,
    /// Token usage information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
    /// Raw response from the API (provider-specific).
    #[serde(skip)]
    pub raw: Option<serde_json::Value>,
}

impl ModelResponse {
    /// Create a new model response.
    #[must_use]
    pub const fn new(message: ChatMessage) -> Self {
        Self {
            message,
            token_usage: None,
            raw: None,
        }
    }

    /// Set token usage.
    #[must_use]
    pub const fn with_token_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = Some(usage);
        self
    }

    /// Set raw response.
    #[must_use]
    pub fn with_raw(mut self, raw: serde_json::Value) -> Self {
        self.raw = Some(raw);
        self
    }

    /// Get the text content of the response.
    #[must_use]
    pub fn text(&self) -> Option<String> {
        self.message.text_content()
    }

    /// Get tool calls from the response.
    #[must_use]
    pub const fn tool_calls(&self) -> Option<&Vec<ChatMessageToolCall>> {
        self.message.tool_calls.as_ref()
    }
}

/// Stream of model response deltas for streaming generation.
pub type ModelStream =
    Pin<Box<dyn Stream<Item = Result<ChatMessageStreamDelta, AgentError>> + Send>>;

/// Options for model generation requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Stop sequences to end generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Available tools for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Temperature for sampling (0.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-p (nucleus) sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Response format specification (e.g., JSON mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
}

impl GenerateOptions {
    /// Create new default generate options.
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

    /// Set available tools for function calling.
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

    /// Set top-p sampling.
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set response format.
    #[must_use]
    pub fn with_response_format(mut self, format: serde_json::Value) -> Self {
        self.response_format = Some(format);
        self
    }
}

/// The core trait for language model implementations.
///
/// This trait defines the interface that all LLM providers must implement.
/// It supports both synchronous and streaming generation, as well as
/// tool/function calling capabilities.
///
/// # Example
///
/// ```rust,ignore
/// use machi::providers::{Model, GenerateOptions};
///
/// async fn generate_response(model: &impl Model) {
///     let messages = vec![ChatMessage::user("Hello!")];
///     let response = model.generate(messages, GenerateOptions::new()).await?;
///     println!("{}", response.text().unwrap_or_default());
/// }
/// ```
#[async_trait]
pub trait Model: Send + Sync {
    /// Get the model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-latest").
    fn model_id(&self) -> &str;

    /// Generate a response for the given messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history
    /// * `options` - Generation options (temperature, tools, etc.)
    ///
    /// # Errors
    ///
    /// Returns an error if the API call fails or the response cannot be parsed.
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelResponse, AgentError>;

    /// Generate a streaming response.
    ///
    /// Default implementation falls back to non-streaming generate.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history
    /// * `options` - Generation options
    ///
    /// # Errors
    ///
    /// Returns an error if the API call fails.
    async fn generate_stream(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
    ) -> Result<ModelStream, AgentError> {
        let response = self.generate(messages, options).await?;
        let delta = ChatMessageStreamDelta {
            content: response.message.text_content(),
            tool_calls: None,
            token_usage: None,
        };
        Ok(Box::pin(futures::stream::once(async move { Ok(delta) })))
    }

    /// Check if the model supports the stop parameter.
    ///
    /// Some models (like `OpenAI`'s o3, o4, gpt-5 series) don't support stop sequences.
    fn supports_stop_parameter(&self) -> bool {
        true
    }

    /// Check if the model supports streaming responses.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if the model supports tool/function calling.
    fn supports_tool_calling(&self) -> bool {
        true
    }
}

/// Trait for providers that can be created from environment variables.
pub trait FromEnv: Sized {
    /// Create a new client from environment variables.
    ///
    /// # Panics
    ///
    /// Panics if required environment variables are not set.
    fn from_env() -> Self;
}

/// Configuration for retrying failed requests.
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_attempts: u32,
    /// Initial delay between retries in milliseconds.
    pub initial_delay_ms: u64,
    /// Exponential backoff multiplier.
    pub backoff_multiplier: f64,
    /// Whether to add jitter to retry delays.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Check if a model ID supports the stop parameter.
///
/// `OpenAI`'s o3, o4, and gpt-5 series don't support stop sequences.
#[must_use]
pub fn model_supports_stop_parameter(model_id: &str) -> bool {
    let model_name = model_id.split('/').next_back().unwrap_or(model_id);

    // o3-mini is an exception that does support stop
    if model_name == "o3-mini" {
        return true;
    }

    // o3*, o4*, gpt-5* don't support stop
    !(model_name.starts_with("o3")
        || model_name.starts_with("o4")
        || model_name.starts_with("gpt-5"))
}

/// Check if a model requires `max_completion_tokens` instead of `max_tokens`.
///
/// `OpenAI`'s o-series and gpt-5 series require the new parameter name.
/// The `max_tokens` parameter is deprecated for these models.
#[must_use]
pub fn model_requires_max_completion_tokens(model_id: &str) -> bool {
    let model_name = model_id.split('/').next_back().unwrap_or(model_id);

    // o1*, o3*, o4*, gpt-5* require max_completion_tokens
    model_name.starts_with("o1")
        || model_name.starts_with("o3")
        || model_name.starts_with("o4")
        || model_name.starts_with("gpt-5")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage() {
        let usage1 = TokenUsage::new(100, 50);
        let usage2 = TokenUsage::new(200, 100);

        assert_eq!(usage1.total(), 150);
        assert_eq!((usage1 + usage2).total(), 450);
    }

    #[test]
    fn test_model_supports_stop() {
        assert!(model_supports_stop_parameter("gpt-4"));
        assert!(model_supports_stop_parameter("gpt-4o"));
        assert!(model_supports_stop_parameter("o3-mini"));
        assert!(!model_supports_stop_parameter("o3"));
        assert!(!model_supports_stop_parameter("o4-mini"));
        assert!(!model_supports_stop_parameter("gpt-5"));
    }

    #[test]
    fn test_model_requires_max_completion_tokens() {
        // Legacy models use max_tokens
        assert!(!model_requires_max_completion_tokens("gpt-4"));
        assert!(!model_requires_max_completion_tokens("gpt-4o"));
        assert!(!model_requires_max_completion_tokens("gpt-4-turbo"));
        assert!(!model_requires_max_completion_tokens("gpt-3.5-turbo"));
        assert!(!model_requires_max_completion_tokens("gpt-4.1"));
        assert!(!model_requires_max_completion_tokens("gpt-4.1-mini"));

        // o-series and gpt-5 models use max_completion_tokens
        assert!(model_requires_max_completion_tokens("o1"));
        assert!(model_requires_max_completion_tokens("o1-mini"));
        assert!(model_requires_max_completion_tokens("o3"));
        assert!(model_requires_max_completion_tokens("o3-mini"));
        assert!(model_requires_max_completion_tokens("o3-pro"));
        assert!(model_requires_max_completion_tokens("o4-mini"));
        assert!(model_requires_max_completion_tokens("gpt-5"));
        assert!(model_requires_max_completion_tokens("gpt-5-mini"));
        assert!(model_requires_max_completion_tokens("gpt-5.1"));
    }
}
