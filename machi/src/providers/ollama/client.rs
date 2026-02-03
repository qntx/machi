//! Ollama API client implementation.

#![allow(
    clippy::missing_fields_in_debug,
    clippy::missing_panics_doc,
    clippy::unused_self
)]

use super::completion::CompletionModel;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::Arc;

/// Default Ollama API base URL (local server).
pub const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

/// Ollama API client for creating completion models.
///
/// Ollama runs locally and doesn't require an API key by default.
///
/// # Example
///
/// ```rust,ignore
/// use machi::providers::ollama::OllamaClient;
///
/// // Connect to default local server
/// let client = OllamaClient::new();
///
/// // Connect to custom host
/// let client = OllamaClient::builder()
///     .base_url("http://192.168.1.100:11434")
///     .build();
///
/// let model = client.completion_model("llama3.3");
/// ```
#[derive(Clone)]
pub struct OllamaClient {
    pub(crate) http_client: reqwest::Client,
    pub(crate) base_url: Arc<str>,
}

impl std::fmt::Debug for OllamaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaClient")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaClient {
    /// Create a new Ollama client with default settings.
    ///
    /// Connects to `http://localhost:11434` by default.
    #[must_use]
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Create a new client builder.
    #[must_use]
    pub fn builder() -> OllamaClientBuilder {
        OllamaClientBuilder::default()
    }

    /// Create a completion model with the specified model ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier (e.g., "llama3.3", "qwen2.5", "mistral")
    #[must_use]
    pub fn completion_model(&self, model_id: impl Into<String>) -> CompletionModel {
        CompletionModel::new(self.clone(), model_id)
    }

    /// Get the base URL for API requests.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Build the headers for API requests.
    pub(crate) fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers
    }

    /// Check if the Ollama server is running and accessible.
    ///
    /// # Errors
    ///
    /// Returns an error if the server is not reachable.
    pub async fn health_check(&self) -> Result<bool, reqwest::Error> {
        let response = self
            .http_client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    /// List available models on the Ollama server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn list_models(&self) -> Result<Vec<String>, reqwest::Error> {
        let response = self
            .http_client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let models = response["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

/// Builder for [`OllamaClient`].
#[derive(Debug, Default)]
pub struct OllamaClientBuilder {
    base_url: Option<String>,
    timeout_secs: Option<u64>,
}

impl OllamaClientBuilder {
    /// Set a custom base URL.
    ///
    /// Useful for connecting to remote Ollama servers.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the request timeout in seconds.
    ///
    /// Default is no timeout (Ollama inference can be slow).
    #[must_use]
    pub const fn timeout_secs(mut self, timeout: u64) -> Self {
        self.timeout_secs = Some(timeout);
        self
    }

    /// Build the client.
    #[must_use]
    pub fn build(self) -> OllamaClient {
        let base_url = self
            .base_url
            .unwrap_or_else(|| OLLAMA_API_BASE_URL.to_string());

        let mut client_builder = reqwest::Client::builder();

        if let Some(timeout) = self.timeout_secs {
            client_builder = client_builder.timeout(std::time::Duration::from_secs(timeout));
        }

        let http_client = client_builder.build().expect("Failed to build HTTP client");

        OllamaClient {
            http_client,
            base_url: base_url.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_builder() {
        let client = OllamaClient::builder()
            .base_url("http://192.168.1.100:11434")
            .timeout_secs(300)
            .build();

        assert_eq!(client.base_url(), "http://192.168.1.100:11434");
    }

    #[test]
    fn test_default_base_url() {
        let client = OllamaClient::new();
        assert_eq!(client.base_url(), OLLAMA_API_BASE_URL);
    }
}
