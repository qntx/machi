//! Anthropic API client implementation.

use super::ANTHROPIC_VERSION_LATEST;
use super::completion::CompletionModel;
use crate::providers::common::FromEnv;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::Arc;

/// Default Anthropic API base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic API client for creating completion models.
///
/// # Example
///
/// ```rust,ignore
/// use machi::providers::anthropic::AnthropicClient;
///
/// // From environment variable ANTHROPIC_API_KEY
/// let client = AnthropicClient::from_env();
///
/// // With explicit API key
/// let client = AnthropicClient::new("sk-ant-...");
///
/// // With custom configuration
/// let client = AnthropicClient::builder()
///     .api_key("sk-ant-...")
///     .anthropic_version("2023-06-01")
///     .build();
/// ```
#[derive(Clone)]
pub struct AnthropicClient {
    pub(crate) http_client: reqwest::Client,
    pub(crate) api_key: Arc<str>,
    pub(crate) base_url: Arc<str>,
    pub(crate) anthropic_version: Arc<str>,
    pub(crate) anthropic_betas: Vec<String>,
}

impl std::fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("base_url", &self.base_url)
            .field("anthropic_version", &self.anthropic_version)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

impl AnthropicClient {
    /// Create a new Anthropic client with the given API key.
    ///
    /// Uses the default Anthropic API base URL and latest API version.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder().api_key(api_key).build()
    }

    /// Create a new client builder.
    #[must_use]
    pub fn builder() -> AnthropicClientBuilder {
        AnthropicClientBuilder::default()
    }

    /// Create a completion model with the specified model ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier (e.g., "claude-3-5-sonnet-latest")
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
    pub(crate) fn auth_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).expect("Invalid API key format"),
        );

        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&self.anthropic_version).expect("Invalid version format"),
        );

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Add beta headers if any
        if !self.anthropic_betas.is_empty()
            && let Ok(value) = HeaderValue::from_str(&self.anthropic_betas.join(",")) {
                headers.insert("anthropic-beta", value);
            }

        headers
    }
}

impl FromEnv for AnthropicClient {
    /// Create a new Anthropic client from environment variables.
    ///
    /// Uses `ANTHROPIC_API_KEY` for the API key and optionally
    /// `ANTHROPIC_BASE_URL` for a custom base URL.
    ///
    /// # Panics
    ///
    /// Panics if `ANTHROPIC_API_KEY` is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY environment variable not set");

        let mut builder = Self::builder().api_key(api_key);

        if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
            builder = builder.base_url(base_url);
        }

        builder.build()
    }
}

/// Builder for [`AnthropicClient`].
#[derive(Debug)]
pub struct AnthropicClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    anthropic_version: String,
    anthropic_betas: Vec<String>,
    timeout_secs: Option<u64>,
}

impl Default for AnthropicClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            anthropic_version: ANTHROPIC_VERSION_LATEST.to_string(),
            anthropic_betas: Vec::new(),
            timeout_secs: None,
        }
    }
}

impl AnthropicClientBuilder {
    /// Set the API key.
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set a custom base URL.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the Anthropic API version.
    #[must_use]
    pub fn anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self
    }

    /// Add a beta feature flag.
    #[must_use]
    pub fn anthropic_beta(mut self, beta: impl Into<String>) -> Self {
        self.anthropic_betas.push(beta.into());
        self
    }

    /// Add multiple beta feature flags.
    #[must_use]
    pub fn anthropic_betas(mut self, betas: &[&str]) -> Self {
        self.anthropic_betas
            .extend(betas.iter().map(|s| (*s).to_string()));
        self
    }

    /// Set the request timeout in seconds.
    #[must_use]
    pub const fn timeout_secs(mut self, timeout: u64) -> Self {
        self.timeout_secs = Some(timeout);
        self
    }

    /// Build the client.
    ///
    /// # Panics
    ///
    /// Panics if the API key is not set.
    #[must_use]
    pub fn build(self) -> AnthropicClient {
        let api_key = self.api_key.expect("API key is required");
        let base_url = self
            .base_url
            .unwrap_or_else(|| ANTHROPIC_API_BASE_URL.to_string());

        let mut client_builder = reqwest::Client::builder();

        if let Some(timeout) = self.timeout_secs {
            client_builder = client_builder.timeout(std::time::Duration::from_secs(timeout));
        }

        let http_client = client_builder.build().expect("Failed to build HTTP client");

        AnthropicClient {
            http_client,
            api_key: api_key.into(),
            base_url: base_url.into(),
            anthropic_version: self.anthropic_version.into(),
            anthropic_betas: self.anthropic_betas,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_builder() {
        let client = AnthropicClient::builder()
            .api_key("test-key")
            .base_url("https://custom.api.com")
            .anthropic_version("2023-06-01")
            .anthropic_beta("prompt-caching-2024-07-31")
            .timeout_secs(30)
            .build();

        assert_eq!(client.base_url(), "https://custom.api.com");
    }

    #[test]
    fn test_default_base_url() {
        let client = AnthropicClient::new("test-key");
        assert_eq!(client.base_url(), ANTHROPIC_API_BASE_URL);
    }
}
