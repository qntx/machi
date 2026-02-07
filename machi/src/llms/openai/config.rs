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

    /// Sets the organization ID.
    #[must_use]
    pub fn organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub const fn timeout(mut self, secs: u64) -> Self {
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

/// # Note
///
/// The default configuration has an **empty API key** and is not usable
/// without calling [`OpenAIConfig::new`] or setting the key manually.
/// Prefer [`OpenAIConfig::new`] or [`OpenAIConfig::from_env`] for production use.
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    mod constants {
        use super::*;

        #[test]
        fn default_base_url_is_openai_api() {
            assert_eq!(OpenAIConfig::DEFAULT_BASE_URL, "https://api.openai.com/v1");
        }

        #[test]
        fn default_model_is_gpt4o() {
            assert_eq!(OpenAIConfig::DEFAULT_MODEL, "gpt-4o");
        }
    }

    mod new {
        use super::*;

        #[test]
        fn creates_with_api_key() {
            let config = OpenAIConfig::new("sk-test-key-123");
            assert_eq!(config.api_key, "sk-test-key-123");
        }

        #[test]
        fn uses_default_base_url() {
            let config = OpenAIConfig::new("key");
            assert_eq!(config.base_url, OpenAIConfig::DEFAULT_BASE_URL);
        }

        #[test]
        fn uses_default_model() {
            let config = OpenAIConfig::new("key");
            assert_eq!(config.model, OpenAIConfig::DEFAULT_MODEL);
        }

        #[test]
        fn organization_is_none() {
            let config = OpenAIConfig::new("key");
            assert!(config.organization.is_none());
        }

        #[test]
        fn timeout_defaults_to_120_seconds() {
            let config = OpenAIConfig::new("key");
            assert_eq!(config.timeout_secs, Some(120));
        }

        #[test]
        fn accepts_string_type() {
            let config = OpenAIConfig::new(String::from("string-key"));
            assert_eq!(config.api_key, "string-key");
        }

        #[test]
        fn accepts_str_reference() {
            let key = "str-ref-key";
            let config = OpenAIConfig::new(key);
            assert_eq!(config.api_key, "str-ref-key");
        }
    }

    mod default {
        use super::*;

        #[test]
        fn api_key_is_empty() {
            let config = OpenAIConfig::default();
            assert!(config.api_key.is_empty());
        }

        #[test]
        fn uses_default_base_url() {
            let config = OpenAIConfig::default();
            assert_eq!(config.base_url, OpenAIConfig::DEFAULT_BASE_URL);
        }

        #[test]
        fn uses_default_model() {
            let config = OpenAIConfig::default();
            assert_eq!(config.model, OpenAIConfig::DEFAULT_MODEL);
        }

        #[test]
        fn organization_is_none() {
            let config = OpenAIConfig::default();
            assert!(config.organization.is_none());
        }

        #[test]
        fn timeout_defaults_to_120_seconds() {
            let config = OpenAIConfig::default();
            assert_eq!(config.timeout_secs, Some(120));
        }
    }

    mod base_url_setter {
        use super::*;

        #[test]
        fn sets_custom_base_url() {
            let config = OpenAIConfig::new("key").base_url("https://custom.api.com/v1");
            assert_eq!(config.base_url, "https://custom.api.com/v1");
        }

        #[test]
        fn preserves_other_fields() {
            let config = OpenAIConfig::new("my-key")
                .model("gpt-4")
                .base_url("https://custom.com");

            assert_eq!(config.api_key, "my-key");
            assert_eq!(config.model, "gpt-4");
        }

        #[test]
        fn accepts_string_type() {
            let url = String::from("https://azure.openai.com");
            let config = OpenAIConfig::new("key").base_url(url);
            assert_eq!(config.base_url, "https://azure.openai.com");
        }
    }

    mod model_setter {
        use super::*;

        #[test]
        fn sets_custom_model() {
            let config = OpenAIConfig::new("key").model("gpt-4-turbo");
            assert_eq!(config.model, "gpt-4-turbo");
        }

        #[test]
        fn preserves_other_fields() {
            let config = OpenAIConfig::new("my-key")
                .base_url("https://custom.com")
                .model("gpt-3.5-turbo");

            assert_eq!(config.api_key, "my-key");
            assert_eq!(config.base_url, "https://custom.com");
        }

        #[test]
        fn accepts_various_model_names() {
            let models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini",
            ];
            for model in models {
                let config = OpenAIConfig::new("key").model(model);
                assert_eq!(config.model, model);
            }
        }
    }

    mod organization_setter {
        use super::*;

        #[test]
        fn sets_organization_id() {
            let config = OpenAIConfig::new("key").organization("org-123456");
            assert_eq!(config.organization, Some("org-123456".to_owned()));
        }

        #[test]
        fn preserves_other_fields() {
            let config = OpenAIConfig::new("my-key")
                .model("gpt-4")
                .organization("my-org");

            assert_eq!(config.api_key, "my-key");
            assert_eq!(config.model, "gpt-4");
        }

        #[test]
        fn overwrites_previous_organization() {
            let config = OpenAIConfig::new("key")
                .organization("org-1")
                .organization("org-2");
            assert_eq!(config.organization, Some("org-2".to_owned()));
        }
    }

    mod timeout_setter {
        use super::*;

        #[test]
        fn sets_custom_timeout() {
            let config = OpenAIConfig::new("key").timeout(60);
            assert_eq!(config.timeout_secs, Some(60));
        }

        #[test]
        fn allows_zero_timeout() {
            let config = OpenAIConfig::new("key").timeout(0);
            assert_eq!(config.timeout_secs, Some(0));
        }

        #[test]
        fn allows_large_timeout() {
            let config = OpenAIConfig::new("key").timeout(3600);
            assert_eq!(config.timeout_secs, Some(3600));
        }

        #[test]
        fn preserves_other_fields() {
            let config = OpenAIConfig::new("my-key").model("gpt-4").timeout(30);

            assert_eq!(config.api_key, "my-key");
            assert_eq!(config.model, "gpt-4");
        }
    }

    mod azure {
        use super::*;

        #[test]
        fn creates_azure_config() {
            let config =
                OpenAIConfig::azure("https://my-resource.openai.azure.com", "azure-api-key");
            assert_eq!(config.base_url, "https://my-resource.openai.azure.com");
            assert_eq!(config.api_key, "azure-api-key");
        }

        #[test]
        fn uses_gpt4o_as_default_model() {
            let config = OpenAIConfig::azure("https://azure.com", "key");
            assert_eq!(config.model, "gpt-4o");
        }

        #[test]
        fn organization_is_none() {
            let config = OpenAIConfig::azure("https://azure.com", "key");
            assert!(config.organization.is_none());
        }

        #[test]
        fn timeout_defaults_to_120() {
            let config = OpenAIConfig::azure("https://azure.com", "key");
            assert_eq!(config.timeout_secs, Some(120));
        }

        #[test]
        fn accepts_string_types() {
            let endpoint = String::from("https://my.azure.com");
            let key = String::from("my-key");
            let config = OpenAIConfig::azure(endpoint, key);
            assert_eq!(config.base_url, "https://my.azure.com");
            assert_eq!(config.api_key, "my-key");
        }
    }

    mod builder_chain {
        use super::*;

        #[test]
        fn full_configuration_chain() {
            let config = OpenAIConfig::new("sk-test-key")
                .base_url("https://custom.openai.com/v1")
                .model("gpt-4-turbo")
                .organization("org-abc123")
                .timeout(90);

            assert_eq!(config.api_key, "sk-test-key");
            assert_eq!(config.base_url, "https://custom.openai.com/v1");
            assert_eq!(config.model, "gpt-4-turbo");
            assert_eq!(config.organization, Some("org-abc123".to_owned()));
            assert_eq!(config.timeout_secs, Some(90));
        }

        #[test]
        fn order_independence() {
            let config1 = OpenAIConfig::new("key")
                .model("model")
                .timeout(60)
                .organization("org");

            let config2 = OpenAIConfig::new("key")
                .organization("org")
                .timeout(60)
                .model("model");

            assert_eq!(config1.model, config2.model);
            assert_eq!(config1.organization, config2.organization);
            assert_eq!(config1.timeout_secs, config2.timeout_secs);
        }
    }

    mod clone {
        use super::*;

        #[test]
        fn clone_preserves_all_fields() {
            let original = OpenAIConfig::new("key")
                .base_url("https://custom.com")
                .model("gpt-4")
                .organization("org")
                .timeout(60);

            let cloned = original.clone();

            assert_eq!(cloned.api_key, original.api_key);
            assert_eq!(cloned.base_url, original.base_url);
            assert_eq!(cloned.model, original.model);
            assert_eq!(cloned.organization, original.organization);
            assert_eq!(cloned.timeout_secs, original.timeout_secs);
        }

        #[test]
        fn clone_is_independent() {
            let original = OpenAIConfig::new("key");
            let mut cloned = original.clone();
            cloned.api_key = "new-key".to_owned();

            assert_eq!(original.api_key, "key");
            assert_eq!(cloned.api_key, "new-key");
        }
    }

    mod debug {
        use super::*;

        #[test]
        fn debug_format_contains_fields() {
            let config = OpenAIConfig::new("test-key").model("gpt-4");
            let debug_str = format!("{config:?}");

            assert!(debug_str.contains("OpenAIConfig"));
            assert!(debug_str.contains("api_key"));
            assert!(debug_str.contains("base_url"));
            assert!(debug_str.contains("model"));
        }
    }

    mod from_env {
        use super::*;

        // Note: from_env tests require environment variables to be set
        // Integration tests with actual env vars should be run separately

        #[test]
        fn uses_defaults_for_optional_env_vars() {
            // This test verifies the function signature and default behavior
            // without manipulating actual environment state
            // When OPENAI_API_KEY is not set, from_env returns an error
            let result = OpenAIConfig::from_env();

            // If env var is set (CI/local dev), verify config is valid
            // If not set, verify error is returned
            match result {
                Ok(config) => {
                    // API key should not be empty when successfully loaded
                    assert!(!config.api_key.is_empty());
                    // Base URL should be set (either from env or default)
                    assert!(!config.base_url.is_empty());
                    // Model should be set (either from env or default)
                    assert!(!config.model.is_empty());
                }
                Err(e) => {
                    // Error message should mention OPENAI_API_KEY
                    let msg = e.to_string();
                    assert!(
                        msg.contains("OPENAI_API_KEY") || msg.contains("api"),
                        "Error should mention API key: {msg}"
                    );
                }
            }
        }
    }
}
