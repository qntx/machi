//! OpenAI client configuration.

use crate::error::{LlmError, Result};

/// Configuration for the OpenAI client.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Base URL for the API (defaults to OpenAI's API).
    pub base_url: String,
    /// Default model to use.
    pub model: String,
    /// Optional organization ID.
    pub organization: Option<String>,
    /// Request timeout in seconds.
    pub timeout_secs: Option<u64>,
}

impl OpenAIConfig {
    /// Default OpenAI API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
    /// Default model.
    pub const DEFAULT_MODEL: &'static str = "gpt-4o";

    /// Creates a new configuration with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: Self::DEFAULT_BASE_URL.to_owned(),
            model: Self::DEFAULT_MODEL.to_owned(),
            organization: None,
            timeout_secs: Some(120),
        }
    }

    /// Creates configuration from environment variables.
    ///
    /// Reads from:
    /// - `OPENAI_API_KEY` - Required API key
    /// - `OPENAI_BASE_URL` - Optional base URL
    /// - `OPENAI_MODEL` - Optional default model
    /// - `OPENAI_ORGANIZATION` - Optional organization ID
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| LlmError::auth("openai", "OPENAI_API_KEY environment variable not set"))?;

        let base_url =
            std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| Self::DEFAULT_BASE_URL.to_owned());

        let model =
            std::env::var("OPENAI_MODEL").unwrap_or_else(|_| Self::DEFAULT_MODEL.to_owned());

        let organization = std::env::var("OPENAI_ORGANIZATION").ok();

        Ok(Self {
            api_key,
            base_url,
            model,
            organization,
            timeout_secs: Some(120),
        })
    }

    /// Sets the base URL.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the default model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the organization ID.
    #[must_use]
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Creates config for Azure OpenAI.
    #[must_use]
    pub fn azure(endpoint: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: endpoint.into(),
            model: "gpt-4o".to_owned(),
            organization: None,
            timeout_secs: Some(120),
        }
    }
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: Self::DEFAULT_BASE_URL.to_owned(),
            model: Self::DEFAULT_MODEL.to_owned(),
            organization: None,
            timeout_secs: Some(120),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = OpenAIConfig::new("test-key");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, OpenAIConfig::DEFAULT_BASE_URL);
        assert_eq!(config.model, OpenAIConfig::DEFAULT_MODEL);
    }

    #[test]
    fn test_config_builder() {
        let config = OpenAIConfig::new("key")
            .with_model("gpt-4")
            .with_timeout(60);

        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.timeout_secs, Some(60));
    }
}
