//! Configuration schema definitions.
//!
//! This module provides type-safe configuration structures with validation
//! and builder patterns for constructing configurations programmatically.

use serde::{Deserialize, Serialize};

/// Root configuration structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BotConfig {
    /// LLM provider configuration.
    #[serde(default)]
    pub providers: ProviderConfig,

    /// Agent configuration.
    #[serde(default)]
    pub agents: AgentConfig,

    /// Channel configuration.
    #[serde(default)]
    pub channels: ChannelConfig,

    /// Tools configuration.
    #[serde(default)]
    pub tools: ToolsConfig,
}

/// LLM provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// OpenRouter configuration (recommended, access to all models).
    #[serde(default)]
    pub openrouter: Option<OpenRouterConfig>,

    /// OpenAI configuration.
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,

    /// Anthropic configuration.
    #[serde(default)]
    pub anthropic: Option<AnthropicConfig>,

    /// Ollama configuration (local models).
    #[serde(default)]
    pub ollama: Option<OllamaConfig>,

    /// Groq configuration (for LLM and voice transcription).
    #[serde(default)]
    pub groq: Option<GroqConfig>,

    /// Google Gemini configuration.
    #[serde(default)]
    pub gemini: Option<GeminiConfig>,

    /// vLLM configuration (local OpenAI-compatible server).
    #[serde(default)]
    pub vllm: Option<VllmConfig>,
}

/// OpenRouter provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    /// API key.
    pub api_key: String,
    /// Base URL override.
    #[serde(default)]
    pub api_base: Option<String>,
}

/// OpenAI provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key.
    pub api_key: String,
    /// Base URL override.
    #[serde(default)]
    pub api_base: Option<String>,
}

/// Anthropic provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key.
    pub api_key: String,
}

/// Ollama provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base URL (default: http://localhost:11434).
    #[serde(default = "default_ollama_url")]
    pub api_base: String,
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

/// Groq provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqConfig {
    /// API key.
    pub api_key: String,
}

/// Google Gemini provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    /// API key.
    pub api_key: String,
}

/// vLLM provider config (OpenAI-compatible local server).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    /// API key (can be any non-empty string for local servers).
    #[serde(default = "default_vllm_key")]
    pub api_key: String,
    /// Base URL for the vLLM server.
    #[serde(default = "default_vllm_url")]
    pub api_base: String,
}

fn default_vllm_key() -> String {
    "dummy".to_string()
}

fn default_vllm_url() -> String {
    "http://localhost:8000/v1".to_string()
}

/// Agent configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Default settings for all agents.
    #[serde(default)]
    pub defaults: AgentDefaults,
}

/// Default agent settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefaults {
    /// Default model to use.
    #[serde(default = "default_model")]
    pub model: String,
    /// Maximum iterations per request.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
}

fn default_model() -> String {
    "anthropic/claude-sonnet-4".to_string()
}

const fn default_max_iterations() -> usize {
    20
}

impl Default for AgentDefaults {
    fn default() -> Self {
        Self {
            model: default_model(),
            max_iterations: default_max_iterations(),
        }
    }
}

/// Channel configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Telegram channel config.
    #[serde(default)]
    pub telegram: TelegramConfig,

    /// WhatsApp channel config.
    #[serde(default)]
    pub whatsapp: WhatsAppConfig,
}

/// Telegram channel configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelegramConfig {
    /// Enable Telegram channel.
    #[serde(default)]
    pub enabled: bool,
    /// Bot token from @BotFather.
    #[serde(default)]
    pub token: Option<String>,
    /// Allowed user IDs (empty = allow all).
    #[serde(default)]
    pub allow_from: Vec<String>,
}

/// WhatsApp channel configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhatsAppConfig {
    /// Enable WhatsApp channel.
    #[serde(default)]
    pub enabled: bool,
    /// Allowed phone numbers (empty = allow all).
    #[serde(default)]
    pub allow_from: Vec<String>,
}

/// Tools configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// Web tools config.
    #[serde(default)]
    pub web: WebToolsConfig,
    /// Exec tool config.
    #[serde(default)]
    pub exec: ExecConfig,
}

/// Web tools configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebToolsConfig {
    /// Web search config.
    #[serde(default)]
    pub search: SearchConfig,
}

/// Search tool configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Brave Search API key.
    #[serde(default)]
    pub api_key: Option<String>,
}

/// Exec tool configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExecConfig {
    /// Timeout in seconds.
    #[serde(default = "default_exec_timeout")]
    pub timeout: u64,
    /// Restrict commands to workspace directory.
    #[serde(default = "default_true")]
    pub restrict_to_workspace: bool,
}

const fn default_exec_timeout() -> u64 {
    300
}

const fn default_true() -> bool {
    true
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            timeout: default_exec_timeout(),
            restrict_to_workspace: true,
        }
    }
}

impl BotConfig {
    /// Validate the configuration and return any issues found.
    #[must_use]
    pub fn validate(&self) -> Vec<ConfigIssue> {
        let mut issues = Vec::new();

        // Check Telegram config consistency
        if self.channels.telegram.enabled && self.channels.telegram.token.is_none() {
            issues.push(ConfigIssue::warning(
                "channels.telegram",
                "Telegram is enabled but no token is set. Set TELEGRAM_BOT_TOKEN env var.",
            ));
        }

        // Check agent iterations
        if self.agents.defaults.max_iterations == 0 {
            issues.push(ConfigIssue::error(
                "agents.defaults.maxIterations",
                "Max iterations must be at least 1",
            ));
        }

        // Check exec timeout
        if self.tools.exec.timeout == 0 {
            issues.push(ConfigIssue::warning(
                "tools.exec.timeout",
                "Exec timeout is 0, commands will timeout immediately",
            ));
        }

        issues
    }

    /// Check if the configuration is valid (no errors).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate()
            .iter()
            .all(|issue| issue.level != IssueLevel::Error)
    }

    /// Merge environment variables into the configuration.
    #[must_use]
    pub fn with_env(mut self) -> Self {
        // Telegram token from env
        if self.channels.telegram.token.is_none()
            && let Ok(token) = std::env::var("TELEGRAM_BOT_TOKEN")
        {
            self.channels.telegram.token = Some(token);
        }

        // Anthropic API key from env
        if self.providers.anthropic.is_none()
            && let Ok(key) = std::env::var("ANTHROPIC_API_KEY")
        {
            self.providers.anthropic = Some(AnthropicConfig { api_key: key });
        }

        // OpenAI API key from env
        if self.providers.openai.is_none()
            && let Ok(key) = std::env::var("OPENAI_API_KEY")
        {
            self.providers.openai = Some(OpenAIConfig {
                api_key: key,
                api_base: None,
            });
        }

        // Groq API key from env
        if self.providers.groq.is_none()
            && let Ok(key) = std::env::var("GROQ_API_KEY")
        {
            self.providers.groq = Some(GroqConfig { api_key: key });
        }

        self
    }
}

/// Configuration validation issue.
#[derive(Debug, Clone)]
pub struct ConfigIssue {
    /// Issue severity level.
    pub level: IssueLevel,
    /// Configuration path (e.g., "channels.telegram.token").
    pub path: String,
    /// Human-readable message.
    pub message: String,
}

impl ConfigIssue {
    /// Create an error-level issue.
    #[must_use]
    pub fn error(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            level: IssueLevel::Error,
            path: path.into(),
            message: message.into(),
        }
    }

    /// Create a warning-level issue.
    #[must_use]
    pub fn warning(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            level: IssueLevel::Warning,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ConfigIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.level {
            IssueLevel::Error => "ERROR",
            IssueLevel::Warning => "WARN",
        };
        write!(f, "[{}] {}: {}", prefix, self.path, self.message)
    }
}

/// Severity level for configuration issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueLevel {
    /// Error that prevents the bot from running correctly.
    Error,
    /// Warning about potential issues.
    Warning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BotConfig::default();
        assert_eq!(config.agents.defaults.model, "anthropic/claude-sonnet-4");
        assert_eq!(config.agents.defaults.max_iterations, 20);
        assert!(!config.channels.telegram.enabled);
    }

    #[test]
    fn test_config_serialization() {
        let config = BotConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: BotConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.agents.defaults.model, config.agents.defaults.model);
    }

    #[test]
    fn test_parse_sample_config() {
        let toml_str = r#"
[providers.openrouter]
api_key = "sk-or-v1-xxx"

[agents.defaults]
model = "anthropic/claude-opus-4-5"

[channels.telegram]
enabled = true
token = "123456:ABC"
allow_from = ["123456789"]
"#;

        let config: BotConfig = toml::from_str(toml_str).unwrap();
        assert!(config.providers.openrouter.is_some());
        assert!(config.channels.telegram.enabled);
        assert_eq!(config.agents.defaults.model, "anthropic/claude-opus-4-5");
    }

    #[test]
    fn test_validation() {
        let config = BotConfig::default();
        let issues = config.validate();
        assert!(issues.is_empty(), "Default config should have no issues");
        assert!(config.is_valid());
    }

    #[test]
    fn test_validation_telegram_without_token() {
        let mut config = BotConfig::default();
        config.channels.telegram.enabled = true;
        let issues = config.validate();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].level, IssueLevel::Warning);
    }

    #[test]
    fn test_validation_zero_iterations() {
        let mut config = BotConfig::default();
        config.agents.defaults.max_iterations = 0;
        let issues = config.validate();
        assert!(!config.is_valid());
        assert!(issues.iter().any(|i| i.level == IssueLevel::Error));
    }
}
