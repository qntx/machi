//! Error types for the machi framework.
//!
//! This module defines all error types used throughout the framework,
//! providing rich error context for debugging and error handling.

use std::fmt;

/// A type alias for `Result<T, AgentError>`.
pub type Result<T> = std::result::Result<T, AgentError>;

/// The main error type for agent operations.
#[derive(Debug)]
pub enum AgentError {
    /// Error during tool execution.
    ToolExecution {
        /// Name of the tool that failed.
        tool_name: String,
        /// The underlying error message.
        message: String,
    },

    /// Error parsing model output.
    Parsing {
        /// The output that failed to parse.
        output: String,
        /// The parsing error message.
        message: String,
    },

    /// Error from the model/LLM.
    Model {
        /// The underlying error message.
        message: String,
    },

    /// Agent reached maximum number of steps.
    MaxSteps {
        /// Number of steps taken.
        steps: usize,
        /// Maximum allowed steps.
        max_steps: usize,
    },

    /// Agent execution was interrupted.
    Interrupted,

    /// Invalid configuration.
    Configuration {
        /// Description of the configuration issue.
        message: String,
    },

    /// HTTP/network error.
    Http {
        /// The underlying error message.
        message: String,
    },

    /// JSON serialization/deserialization error.
    Json {
        /// The underlying error message.
        message: String,
    },

    /// Generic internal error.
    Internal {
        /// The underlying error message.
        message: String,
    },
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolExecution { tool_name, message } => {
                write!(f, "Tool execution error in '{tool_name}': {message}")
            }
            Self::Parsing { output, message } => {
                write!(f, "Parsing error: {message}. Output: {output}")
            }
            Self::Model { message } => {
                write!(f, "Model error: {message}")
            }
            Self::MaxSteps { steps, max_steps } => {
                write!(f, "Reached maximum steps ({steps}/{max_steps})")
            }
            Self::Interrupted => {
                write!(f, "Agent execution was interrupted")
            }
            Self::Configuration { message } => {
                write!(f, "Configuration error: {message}")
            }
            Self::Http { message } => {
                write!(f, "HTTP error: {message}")
            }
            Self::Json { message } => {
                write!(f, "JSON error: {message}")
            }
            Self::Internal { message } => {
                write!(f, "Internal error: {message}")
            }
        }
    }
}

impl std::error::Error for AgentError {}

impl From<reqwest::Error> for AgentError {
    fn from(err: reqwest::Error) -> Self {
        Self::Http {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json {
            message: err.to_string(),
        }
    }
}

impl AgentError {
    /// Create a new tool execution error.
    #[must_use]
    pub fn tool_execution(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ToolExecution {
            tool_name: tool_name.into(),
            message: message.into(),
        }
    }

    /// Create a new parsing error.
    #[must_use]
    pub fn parsing(output: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Parsing {
            output: output.into(),
            message: message.into(),
        }
    }

    /// Create a new model error.
    #[must_use]
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
        }
    }

    /// Create a new max steps error.
    #[must_use]
    pub const fn max_steps(steps: usize, max_steps: usize) -> Self {
        Self::MaxSteps { steps, max_steps }
    }

    /// Create a new configuration error.
    #[must_use]
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}
