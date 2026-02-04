//! Configuration management for machi-bot.
//!
//! Provides a configuration system that loads settings from:
//! 1. Default values
//! 2. Config file (`~/.machi-bot/config.toml`)
//! 3. Environment variables

mod schema;

pub use schema::{
    AgentConfig, BotConfig, ChannelConfig, ConfigIssue, ExecConfig, GeminiConfig, GroqConfig,
    IssueLevel, ProviderConfig, TelegramConfig, ToolsConfig, VllmConfig,
};

use std::path::PathBuf;
use tracing::{debug, info};

/// Error type for configuration operations.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// TOML parsing error.
    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),
    /// TOML serialization error.
    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    /// Missing required field.
    #[error("missing required config: {0}")]
    MissingField(String),
    /// Invalid value.
    #[error("invalid config value: {0}")]
    InvalidValue(String),
}

/// Result type for configuration operations.
pub type ConfigResult<T> = Result<T, ConfigError>;

/// Get the default config directory path.
#[must_use]
pub fn default_config_dir() -> PathBuf {
    dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".machi")
}

/// Get the default config file path.
#[must_use]
pub fn config_path() -> PathBuf {
    default_config_dir().join("config.toml")
}

/// Load configuration from the default path.
pub async fn load_config() -> ConfigResult<BotConfig> {
    load_config_from(config_path()).await
}

/// Load configuration from a specific path.
pub async fn load_config_from(path: PathBuf) -> ConfigResult<BotConfig> {
    if !path.exists() {
        info!(path = %path.display(), "config file not found, using defaults");
        return Ok(BotConfig::default());
    }

    let content = tokio::fs::read_to_string(&path).await?;
    let config: BotConfig = toml::from_str(&content)?;
    debug!(path = %path.display(), "loaded config file");

    Ok(config)
}

/// Save configuration to the default path.
pub async fn save_config(config: &BotConfig) -> ConfigResult<()> {
    save_config_to(config, config_path()).await
}

/// Save configuration to a specific path.
pub async fn save_config_to(config: &BotConfig, path: PathBuf) -> ConfigResult<()> {
    // Ensure directory exists
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let content = toml::to_string_pretty(config)?;
    tokio::fs::write(&path, content).await?;
    info!(path = %path.display(), "saved config file");

    Ok(())
}

/// Initialize configuration directory and create default config if needed.
pub async fn init_config() -> ConfigResult<BotConfig> {
    let cfg_dir = default_config_dir();
    let cfg_path = config_path();

    // Create directories
    tokio::fs::create_dir_all(&cfg_dir).await?;
    tokio::fs::create_dir_all(cfg_dir.join("sessions")).await?;
    tokio::fs::create_dir_all(cfg_dir.join("workspace")).await?;

    // Create default config if not exists
    if !cfg_path.exists() {
        let config = BotConfig::default();
        save_config(&config).await?;
        info!("created default config at {}", cfg_path.display());
    }

    load_config().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_paths() {
        let cfg_dir = default_config_dir();
        assert!(cfg_dir.ends_with(".machi"));

        let cfg_path = config_path();
        assert!(cfg_path.ends_with("config.toml"));
    }
}
