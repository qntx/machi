//! Agent configuration.

use serde::{Deserialize, Serialize};

/// Configuration for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of steps the agent can take.
    pub max_steps: usize,
    /// Agent name (optional).
    pub name: Option<String>,
    /// Agent description (optional).
    pub description: Option<String>,
    /// Custom instructions for the agent.
    pub instructions: Option<String>,
    /// Whether to provide run summary when called as managed agent.
    pub provide_run_summary: bool,
    /// Temperature for LLM calls.
    pub temperature: Option<f32>,
    /// Maximum tokens per LLM call.
    pub max_tokens: Option<u32>,
}

impl AgentConfig {
    /// Default maximum steps.
    pub const DEFAULT_MAX_STEPS: usize = 20;

    /// Creates a new configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum steps.
    #[must_use]
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets the agent name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the agent description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets custom instructions.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Sets temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets max tokens.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: Self::DEFAULT_MAX_STEPS,
            name: None,
            description: None,
            instructions: None,
            provide_run_summary: false,
            temperature: None,
            max_tokens: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AgentConfig::new();
        assert_eq!(config.max_steps, AgentConfig::DEFAULT_MAX_STEPS);
        assert!(config.name.is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = AgentConfig::new()
            .with_max_steps(10)
            .with_name("test_agent")
            .with_temperature(0.7);

        assert_eq!(config.max_steps, 10);
        assert_eq!(config.name, Some("test_agent".to_owned()));
        assert_eq!(config.temperature, Some(0.7));
    }
}
