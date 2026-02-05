//! Ollama client configuration.

/// Configuration for the Ollama client.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Base URL for the Ollama API.
    pub base_url: String,
    /// Default model to use.
    pub model: String,
    /// Request timeout in seconds.
    pub timeout_secs: Option<u64>,
}

impl OllamaConfig {
    /// Default Ollama API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:11434";
    /// Default model.
    pub const DEFAULT_MODEL: &'static str = "qwen3";

    /// Creates a new configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates configuration with a specific model.
    #[must_use]
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Creates configuration from environment variables.
    ///
    /// Reads from:
    /// - `OLLAMA_BASE_URL` - Optional base URL
    /// - `OLLAMA_MODEL` - Optional default model
    #[must_use]
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| Self::DEFAULT_BASE_URL.to_owned());

        let model =
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| Self::DEFAULT_MODEL.to_owned());

        Self {
            base_url,
            model,
            timeout_secs: Some(300),
        }
    }

    /// Sets the base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the default model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub const fn timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: Self::DEFAULT_BASE_URL.to_owned(),
            model: Self::DEFAULT_MODEL.to_owned(),
            timeout_secs: Some(300),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OllamaConfig::new();
        assert_eq!(config.base_url, OllamaConfig::DEFAULT_BASE_URL);
        assert_eq!(config.model, OllamaConfig::DEFAULT_MODEL);
    }

    #[test]
    fn test_config_with_model() {
        let config = OllamaConfig::with_model("qwen2.5");
        assert_eq!(config.model, "qwen2.5");
    }

    #[test]
    fn test_config_builder() {
        let config = OllamaConfig::new().model("mistral").timeout(60);

        assert_eq!(config.model, "mistral");
        assert_eq!(config.timeout_secs, Some(60));
    }
}
