//! `OpenAI` API client implementation.
//!
//! Core types (`ChatRequest`, `Message`, `ToolDefinition`, `ResponseFormat`)
//! serialize directly to `OpenAI`'s expected JSON format — no intermediate
//! wire types are needed for the request path.

use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;

use super::config::OpenAIConfig;
use crate::chat::ChatRequest;
use crate::error::Result;
use crate::llms::LlmError;

/// `OpenAI` error response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

/// `OpenAI` error details.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// `OpenAI` API client.
#[derive(Debug, Clone)]
pub struct OpenAI {
    pub(crate) config: Arc<OpenAIConfig>,
    pub(crate) client: Client,
}

impl OpenAI {
    /// Create a new `OpenAI` client with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is empty or the HTTP client fails to build.
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(LlmError::auth("openai", "API key is required").into());
        }

        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout_secs {
            builder = builder.timeout(Duration::from_secs(timeout));
        }

        let client = builder
            .build()
            .map_err(|e| LlmError::internal(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            config: Arc::new(config),
            client,
        })
    }

    /// Create a client from environment variables.
    ///
    /// # Errors
    ///
    /// Returns an error if required environment variables are missing or the client fails to build.
    pub fn from_env() -> Result<Self> {
        let config = OpenAIConfig::from_env()?;
        Self::new(config)
    }

    /// Get the API key.
    #[must_use]
    pub fn api_key(&self) -> &str {
        &self.config.api_key
    }

    /// Get the base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Get the default model.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Build the chat completions URL.
    pub(crate) fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.config.base_url)
    }

    /// Build the audio speech URL.
    pub(crate) fn speech_url(&self) -> String {
        format!("{}/audio/speech", self.config.base_url)
    }

    /// Build the audio transcriptions URL.
    pub(crate) fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.config.base_url)
    }

    /// Build the embeddings URL.
    pub(crate) fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.config.base_url)
    }

    /// Build request headers for JSON requests.
    pub(crate) fn build_request(&self, url: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        req
    }

    /// Build request headers for multipart requests.
    pub(crate) fn build_multipart_request(&self, url: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key));

        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        req
    }

    /// Serialize a [`ChatRequest`] to a JSON [`Value`] for the `OpenAI` API.
    ///
    /// Core types already carry the correct serde attributes, so this is a
    /// thin wrapper that fills in the default model and adds `stream_options`
    /// when streaming.
    pub(crate) fn build_chat_body(
        &self,
        request: &ChatRequest,
        streaming: bool,
    ) -> Result<Value> {
        let mut body = serde_json::to_value(request)
            .map_err(|e| LlmError::internal(format!("Failed to serialize request: {e}")))?;

        // Fill default model from config when the request omits it.
        if request.model.is_empty() {
            body["model"] = Value::String(self.config.model.clone());
        }

        // Enable streaming with usage reporting.
        if streaming {
            body["stream"] = Value::Bool(true);
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }

        Ok(body)
    }

    /// Parse an error response from `OpenAI`.
    pub(crate) fn parse_error(status: u16, body: &str) -> LlmError {
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(body) {
            let error = error_response.error;
            let code = error.code.unwrap_or_else(|| error.error_type.clone());

            return match status {
                401 => LlmError::auth("openai", error.message),
                429 => LlmError::rate_limited("openai"),
                400 if error.message.contains("context_length") => {
                    // Attempt to extract token counts from the error message.
                    let (used, max) = parse_context_length_tokens(&error.message);
                    LlmError::context_exceeded(used, max)
                }
                _ => LlmError::provider_code("openai", code, error.message),
            };
        }

        LlmError::http_status(status, body.to_owned())
    }
}

/// Extract token counts from an `OpenAI` context-length error message.
///
/// `OpenAI` error messages typically look like:
/// "This model's maximum context length is 8192 tokens. However, your messages resulted in 9500 tokens."
/// Returns `(used, max)`, defaulting to `(0, 0)` if parsing fails.
fn parse_context_length_tokens(message: &str) -> (usize, usize) {
    let mut max = 0usize;
    let mut used = 0usize;

    // Look for "maximum context length is <N> tokens"
    if let Some(pos) = message.find("maximum context length is ") {
        let after = &message[pos + "maximum context length is ".len()..];
        if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
            max = after[..end].parse().unwrap_or(0);
        }
    }

    // Look for "resulted in <N> tokens" or "your messages resulted in <N> tokens"
    if let Some(pos) = message.find("resulted in ") {
        let after = &message[pos + "resulted in ".len()..];
        if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
            used = after[..end].parse().unwrap_or(0);
        }
    }

    (used, max)
}
