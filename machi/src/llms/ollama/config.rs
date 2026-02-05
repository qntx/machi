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
    /// Controls how long the model stays loaded in memory (e.g., "5m", "0" to unload immediately).
    pub keep_alive: Option<String>,
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
    /// - `OLLAMA_KEEP_ALIVE` - Optional keep alive duration
    #[must_use]
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| Self::DEFAULT_BASE_URL.to_owned());

        let model =
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| Self::DEFAULT_MODEL.to_owned());

        let keep_alive = std::env::var("OLLAMA_KEEP_ALIVE").ok();

        Self {
            base_url,
            model,
            timeout_secs: Some(300),
            keep_alive,
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

    /// Sets the keep alive duration.
    #[must_use]
    pub fn keep_alive(mut self, duration: impl Into<String>) -> Self {
        self.keep_alive = Some(duration.into());
        self
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: Self::DEFAULT_BASE_URL.to_owned(),
            model: Self::DEFAULT_MODEL.to_owned(),
            timeout_secs: Some(300),
            keep_alive: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod constants {
        use super::*;

        #[test]
        fn default_base_url_is_localhost() {
            assert_eq!(OllamaConfig::DEFAULT_BASE_URL, "http://localhost:11434");
        }

        #[test]
        fn default_model_is_qwen3() {
            assert_eq!(OllamaConfig::DEFAULT_MODEL, "qwen3");
        }
    }

    mod default {
        use super::*;

        #[test]
        fn default_uses_constant_values() {
            let config = OllamaConfig::default();

            assert_eq!(config.base_url, OllamaConfig::DEFAULT_BASE_URL);
            assert_eq!(config.model, OllamaConfig::DEFAULT_MODEL);
        }

        #[test]
        fn default_timeout_is_300_seconds() {
            let config = OllamaConfig::default();
            assert_eq!(config.timeout_secs, Some(300));
        }

        #[test]
        fn default_keep_alive_is_none() {
            let config = OllamaConfig::default();
            assert!(config.keep_alive.is_none());
        }
    }

    mod new {
        use super::*;

        #[test]
        fn new_returns_default_config() {
            let config = OllamaConfig::new();
            let default = OllamaConfig::default();

            assert_eq!(config.base_url, default.base_url);
            assert_eq!(config.model, default.model);
            assert_eq!(config.timeout_secs, default.timeout_secs);
            assert_eq!(config.keep_alive, default.keep_alive);
        }
    }

    mod with_model {
        use super::*;

        #[test]
        fn with_model_sets_model() {
            let config = OllamaConfig::with_model("llama3");
            assert_eq!(config.model, "llama3");
        }

        #[test]
        fn with_model_uses_default_base_url() {
            let config = OllamaConfig::with_model("llama3");
            assert_eq!(config.base_url, OllamaConfig::DEFAULT_BASE_URL);
        }

        #[test]
        fn with_model_uses_default_timeout() {
            let config = OllamaConfig::with_model("llama3");
            assert_eq!(config.timeout_secs, Some(300));
        }

        #[test]
        fn with_model_accepts_string() {
            let config = OllamaConfig::with_model(String::from("mistral"));
            assert_eq!(config.model, "mistral");
        }

        #[test]
        fn with_model_accepts_str() {
            let config = OllamaConfig::with_model("codellama");
            assert_eq!(config.model, "codellama");
        }
    }

    mod builder_methods {
        use super::*;

        #[test]
        fn base_url_sets_value() {
            let config = OllamaConfig::new().base_url("http://remote:11434");
            assert_eq!(config.base_url, "http://remote:11434");
        }

        #[test]
        fn base_url_accepts_string() {
            let url = String::from("http://custom:8080");
            let config = OllamaConfig::new().base_url(url);
            assert_eq!(config.base_url, "http://custom:8080");
        }

        #[test]
        fn model_sets_value() {
            let config = OllamaConfig::new().model("phi3");
            assert_eq!(config.model, "phi3");
        }

        #[test]
        fn model_accepts_string() {
            let model = String::from("gemma");
            let config = OllamaConfig::new().model(model);
            assert_eq!(config.model, "gemma");
        }

        #[test]
        fn timeout_sets_value() {
            let config = OllamaConfig::new().timeout(60);
            assert_eq!(config.timeout_secs, Some(60));
        }

        #[test]
        fn timeout_can_be_zero() {
            let config = OllamaConfig::new().timeout(0);
            assert_eq!(config.timeout_secs, Some(0));
        }

        #[test]
        fn keep_alive_sets_value() {
            let config = OllamaConfig::new().keep_alive("5m");
            assert_eq!(config.keep_alive, Some("5m".to_owned()));
        }

        #[test]
        fn keep_alive_accepts_string() {
            let duration = String::from("10m");
            let config = OllamaConfig::new().keep_alive(duration);
            assert_eq!(config.keep_alive, Some("10m".to_owned()));
        }

        #[test]
        fn keep_alive_zero_unloads_immediately() {
            let config = OllamaConfig::new().keep_alive("0");
            assert_eq!(config.keep_alive, Some("0".to_owned()));
        }
    }

    mod builder_chain {
        use super::*;

        #[test]
        fn full_builder_chain() {
            let config = OllamaConfig::new()
                .base_url("http://server:11434")
                .model("llama3:70b")
                .timeout(120)
                .keep_alive("30m");

            assert_eq!(config.base_url, "http://server:11434");
            assert_eq!(config.model, "llama3:70b");
            assert_eq!(config.timeout_secs, Some(120));
            assert_eq!(config.keep_alive, Some("30m".to_owned()));
        }

        #[test]
        fn builder_chain_order_independent() {
            let config1 = OllamaConfig::new()
                .model("test")
                .base_url("http://a")
                .timeout(10);

            let config2 = OllamaConfig::new()
                .timeout(10)
                .base_url("http://a")
                .model("test");

            assert_eq!(config1.base_url, config2.base_url);
            assert_eq!(config1.model, config2.model);
            assert_eq!(config1.timeout_secs, config2.timeout_secs);
        }

        #[test]
        fn builder_can_override_values() {
            let config = OllamaConfig::new()
                .model("first")
                .model("second")
                .base_url("http://first")
                .base_url("http://second");

            assert_eq!(config.model, "second");
            assert_eq!(config.base_url, "http://second");
        }
    }

    mod traits {
        use super::*;

        #[test]
        fn clone_creates_independent_copy() {
            let original = OllamaConfig::new().model("original").keep_alive("5m");

            let cloned = original;

            assert_eq!(cloned.model, "original");
            assert_eq!(cloned.keep_alive, Some("5m".to_owned()));
        }

        #[test]
        fn debug_is_implemented() {
            let config = OllamaConfig::new();
            let debug_str = format!("{config:?}");

            assert!(debug_str.contains("OllamaConfig"));
            assert!(debug_str.contains("base_url"));
            assert!(debug_str.contains("model"));
        }
    }

    mod from_env {
        use super::*;

        #[test]
        fn from_env_timeout_is_300() {
            let config = OllamaConfig::from_env();
            assert_eq!(config.timeout_secs, Some(300));
        }

        #[test]
        fn from_env_returns_valid_config() {
            let config = OllamaConfig::from_env();

            // Verify config has valid values (either from env or defaults)
            assert!(!config.base_url.is_empty());
            assert!(!config.model.is_empty());
            assert!(config.timeout_secs.is_some());
        }

        #[test]
        fn from_env_base_url_is_valid_url_format() {
            let config = OllamaConfig::from_env();
            assert!(config.base_url.starts_with("http"));
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn with_model_then_builder_chain() {
            let config = OllamaConfig::with_model("llama3")
                .base_url("http://gpu-server:11434")
                .timeout(600)
                .keep_alive("1h");

            assert_eq!(config.model, "llama3");
            assert_eq!(config.base_url, "http://gpu-server:11434");
            assert_eq!(config.timeout_secs, Some(600));
            assert_eq!(config.keep_alive, Some("1h".to_owned()));
        }

        #[test]
        fn typical_local_development_config() {
            let config = OllamaConfig::new();

            assert_eq!(config.base_url, "http://localhost:11434");
            assert_eq!(config.model, "qwen3");
            assert_eq!(config.timeout_secs, Some(300));
        }

        #[test]
        fn typical_production_config() {
            let config = OllamaConfig::new()
                .base_url("http://ollama-service:11434")
                .model("llama3:70b")
                .timeout(600)
                .keep_alive("0");

            assert_eq!(config.base_url, "http://ollama-service:11434");
            assert_eq!(config.model, "llama3:70b");
            assert_eq!(config.timeout_secs, Some(600));
            assert_eq!(config.keep_alive, Some("0".to_owned()));
        }
    }
}
