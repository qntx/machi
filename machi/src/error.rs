//! Unified error types for the machi framework.
//!
//! This module provides a comprehensive error hierarchy covering:
//! - LLM provider errors (authentication, rate limiting, etc.)
//! - Tool execution errors
//! - Agent runtime errors

use std::fmt;

/// Result type alias for machi operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The main error type for the machi framework.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// LLM provider error.
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// Tool execution error.
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// Agent runtime error.
    #[error("Agent error: {0}")]
    Agent(String),

    /// Maximum steps reached during agent execution.
    #[error("Maximum steps ({max_steps}) reached without final answer")]
    MaxSteps {
        /// The maximum number of steps configured.
        max_steps: usize,
    },

    /// Agent execution was interrupted.
    #[error("Agent execution was interrupted")]
    Interrupted,

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP request error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

impl Error {
    /// Create an agent error with a message.
    #[must_use]
    pub fn agent(msg: impl Into<String>) -> Self {
        Self::Agent(msg.into())
    }

    /// Create a max steps error.
    #[must_use]
    pub const fn max_steps(max_steps: usize) -> Self {
        Self::MaxSteps { max_steps }
    }
}

/// Error type for LLM provider operations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LlmError {
    /// The error kind.
    pub kind: LlmErrorKind,
    /// The provider name (e.g., "openai", "ollama").
    pub provider: Option<String>,
    /// Additional error message.
    pub message: String,
    /// Optional error code from the provider.
    pub code: Option<String>,
}

/// Categories of LLM errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LlmErrorKind {
    /// Authentication or authorization failure.
    Auth,
    /// Rate limit exceeded.
    RateLimited,
    /// Context length exceeded.
    ContextExceeded,
    /// Invalid request parameters.
    InvalidRequest,
    /// Response format error.
    ResponseFormat,
    /// Network or connection error.
    Network,
    /// Streaming error.
    Stream,
    /// HTTP status error.
    HttpStatus,
    /// Provider-specific error.
    Provider,
    /// Internal error.
    Internal,
    /// Feature not supported.
    NotSupported,
}

impl LlmError {
    /// Create an authentication error.
    #[must_use]
    pub fn auth(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::Auth,
            provider: Some(provider.into()),
            message: message.into(),
            code: None,
        }
    }

    /// Create a rate limit error.
    #[must_use]
    pub fn rate_limited(provider: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::RateLimited,
            provider: Some(provider.into()),
            message: "Rate limit exceeded. Please retry after some time.".into(),
            code: None,
        }
    }

    /// Create a context exceeded error.
    #[must_use]
    pub fn context_exceeded(used: usize, max: usize) -> Self {
        Self {
            kind: LlmErrorKind::ContextExceeded,
            provider: None,
            message: format!("Context length exceeded: used {used}, max {max}"),
            code: None,
        }
    }

    /// Create a response format error.
    #[must_use]
    pub fn response_format(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::ResponseFormat,
            provider: None,
            message: format!("Expected {}, got {}", expected.into(), got.into()),
            code: None,
        }
    }

    /// Create a network error.
    #[must_use]
    pub fn network(message: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::Network,
            provider: None,
            message: message.into(),
            code: None,
        }
    }

    /// Create a streaming error.
    #[must_use]
    pub fn stream(message: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::Stream,
            provider: None,
            message: message.into(),
            code: None,
        }
    }

    /// Create an HTTP status error.
    #[must_use]
    pub fn http_status(status: u16, body: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::HttpStatus,
            provider: None,
            message: format!("HTTP {status}: {}", body.into()),
            code: Some(status.to_string()),
        }
    }

    /// Create a provider-specific error.
    #[must_use]
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::Provider,
            provider: Some(provider.into()),
            message: message.into(),
            code: None,
        }
    }

    /// Create a provider error with an error code.
    #[must_use]
    pub fn provider_code(
        provider: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind: LlmErrorKind::Provider,
            provider: Some(provider.into()),
            message: message.into(),
            code: Some(code.into()),
        }
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::Internal,
            provider: None,
            message: message.into(),
            code: None,
        }
    }

    /// Create a not supported error.
    #[must_use]
    pub fn not_supported(feature: impl Into<String>) -> Self {
        Self {
            kind: LlmErrorKind::NotSupported,
            provider: None,
            message: format!("Feature not supported: {}", feature.into()),
            code: None,
        }
    }

    /// Check if this is a retryable error.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self.kind, LlmErrorKind::RateLimited | LlmErrorKind::Network)
    }
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(provider) = &self.provider {
            write!(f, "[{provider}] ")?;
        }
        write!(f, "{}", self.message)?;
        if let Some(code) = &self.code {
            write!(f, " (code: {code})")?;
        }
        Ok(())
    }
}

impl std::error::Error for LlmError {}

impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::network("Request timed out")
        } else if err.is_connect() {
            Self::network(format!("Connection failed: {err}"))
        } else {
            Self::network(err.to_string())
        }
    }
}

/// Error type for tool execution failures.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
    /// Error during tool execution.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Invalid arguments provided to the tool.
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    /// Tool not found.
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Tool is not initialized.
    #[error("Tool not initialized")]
    NotInitialized,

    /// Tool execution is forbidden by policy.
    #[error("Tool '{0}' is forbidden by policy")]
    Forbidden(String),

    /// Tool execution was denied by human confirmation.
    #[error("Tool '{0}' execution denied by confirmation")]
    ConfirmationDenied(String),

    /// Generic error.
    #[error("Tool error: {0}")]
    Other(String),
}

impl ToolError {
    /// Create an execution error.
    #[must_use]
    pub fn execution(msg: impl Into<String>) -> Self {
        Self::Execution(msg.into())
    }

    /// Create an invalid arguments error.
    #[must_use]
    pub fn invalid_args(msg: impl Into<String>) -> Self {
        Self::InvalidArguments(msg.into())
    }

    /// Create a not found error.
    #[must_use]
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound(name.into())
    }

    /// Create a forbidden error.
    #[must_use]
    pub fn forbidden(tool_name: impl Into<String>) -> Self {
        Self::Forbidden(tool_name.into())
    }

    /// Create a confirmation denied error.
    #[must_use]
    pub fn confirmation_denied(tool_name: impl Into<String>) -> Self {
        Self::ConfirmationDenied(tool_name.into())
    }
}

impl From<String> for ToolError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for ToolError {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(err: serde_json::Error) -> Self {
        Self::InvalidArguments(err.to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    mod error {
        use super::*;

        #[test]
        fn agent_creates_error() {
            let err = Error::agent("something went wrong");
            assert!(matches!(err, Error::Agent(_)));
            assert!(err.to_string().contains("something went wrong"));
        }

        #[test]
        fn max_steps_creates_error() {
            let err = Error::max_steps(10);
            assert!(matches!(err, Error::MaxSteps { max_steps: 10 }));
            assert!(err.to_string().contains("10"));
        }

        #[test]
        fn from_llm_error() {
            let llm_err = LlmError::network("timeout");
            let err: Error = llm_err.into();
            assert!(matches!(err, Error::Llm(_)));
        }

        #[test]
        fn from_tool_error() {
            let tool_err = ToolError::not_found("my_tool");
            let err: Error = tool_err.into();
            assert!(matches!(err, Error::Tool(_)));
        }

        #[test]
        fn from_io_error() {
            let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
            let err: Error = io_err.into();
            assert!(matches!(err, Error::Io(_)));
        }

        #[test]
        fn from_json_error() {
            let json_err = serde_json::from_str::<i32>("invalid").unwrap_err();
            let err: Error = json_err.into();
            assert!(matches!(err, Error::Json(_)));
        }

        #[test]
        fn display_variants() {
            assert!(Error::agent("msg").to_string().contains("Agent"));
            assert!(Error::max_steps(5).to_string().contains("Maximum steps"));
            assert!(Error::Interrupted.to_string().contains("interrupted"));
        }
    }

    mod llm_error {
        use super::*;

        #[test]
        fn auth_creates_error() {
            let err = LlmError::auth("openai", "Invalid API key");
            assert_eq!(err.kind, LlmErrorKind::Auth);
            assert_eq!(err.provider.as_deref(), Some("openai"));
            assert!(err.message.contains("Invalid API key"));
            assert!(err.code.is_none());
        }

        #[test]
        fn rate_limited_creates_error() {
            let err = LlmError::rate_limited("openai");
            assert_eq!(err.kind, LlmErrorKind::RateLimited);
            assert_eq!(err.provider.as_deref(), Some("openai"));
            assert!(err.message.contains("Rate limit"));
        }

        #[test]
        fn context_exceeded_creates_error() {
            let err = LlmError::context_exceeded(10000, 8192);
            assert_eq!(err.kind, LlmErrorKind::ContextExceeded);
            assert!(err.provider.is_none());
            assert!(err.message.contains("10000"));
            assert!(err.message.contains("8192"));
        }

        #[test]
        fn response_format_creates_error() {
            let err = LlmError::response_format("json", "text");
            assert_eq!(err.kind, LlmErrorKind::ResponseFormat);
            assert!(err.message.contains("json"));
            assert!(err.message.contains("text"));
        }

        #[test]
        fn network_creates_error() {
            let err = LlmError::network("connection refused");
            assert_eq!(err.kind, LlmErrorKind::Network);
            assert!(err.message.contains("connection refused"));
        }

        #[test]
        fn stream_creates_error() {
            let err = LlmError::stream("stream interrupted");
            assert_eq!(err.kind, LlmErrorKind::Stream);
            assert!(err.message.contains("stream interrupted"));
        }

        #[test]
        fn http_status_creates_error() {
            let err = LlmError::http_status(429, "Too Many Requests");
            assert_eq!(err.kind, LlmErrorKind::HttpStatus);
            assert!(err.message.contains("429"));
            assert_eq!(err.code.as_deref(), Some("429"));
        }

        #[test]
        fn provider_creates_error() {
            let err = LlmError::provider("ollama", "model not found");
            assert_eq!(err.kind, LlmErrorKind::Provider);
            assert_eq!(err.provider.as_deref(), Some("ollama"));
        }

        #[test]
        fn provider_code_creates_error() {
            let err = LlmError::provider_code("openai", "model_not_found", "gpt-5 not available");
            assert_eq!(err.kind, LlmErrorKind::Provider);
            assert_eq!(err.code.as_deref(), Some("model_not_found"));
        }

        #[test]
        fn internal_creates_error() {
            let err = LlmError::internal("unexpected state");
            assert_eq!(err.kind, LlmErrorKind::Internal);
            assert!(err.provider.is_none());
        }

        #[test]
        fn not_supported_creates_error() {
            let err = LlmError::not_supported("vision");
            assert_eq!(err.kind, LlmErrorKind::NotSupported);
            assert!(err.message.contains("vision"));
        }

        #[test]
        fn is_retryable_rate_limited() {
            let err = LlmError::rate_limited("openai");
            assert!(err.is_retryable());
        }

        #[test]
        fn is_retryable_network() {
            let err = LlmError::network("timeout");
            assert!(err.is_retryable());
        }

        #[test]
        fn is_retryable_auth_false() {
            let err = LlmError::auth("openai", "bad key");
            assert!(!err.is_retryable());
        }

        #[test]
        fn is_retryable_internal_false() {
            let err = LlmError::internal("bug");
            assert!(!err.is_retryable());
        }

        #[test]
        fn display_with_provider() {
            let err = LlmError::auth("openai", "Invalid key");
            let s = err.to_string();
            assert!(s.contains("[openai]"));
            assert!(s.contains("Invalid key"));
        }

        #[test]
        fn display_without_provider() {
            let err = LlmError::network("timeout");
            let s = err.to_string();
            assert!(!s.contains('['));
            assert!(s.contains("timeout"));
        }

        #[test]
        fn display_with_code() {
            let err = LlmError::http_status(500, "Internal Server Error");
            let s = err.to_string();
            assert!(s.contains("(code: 500)"));
        }

        #[test]
        fn clone_trait() {
            let err1 = LlmError::auth("openai", "msg");
            let err2 = err1.clone();
            assert_eq!(err1.kind, err2.kind);
            assert_eq!(err1.message, err2.message);
        }

        #[test]
        fn implements_std_error() {
            let err = LlmError::network("test");
            let _: &dyn std::error::Error = &err;
        }
    }

    mod llm_error_kind {
        use super::*;

        #[test]
        fn copy_trait() {
            let k1 = LlmErrorKind::Auth;
            let k2 = k1;
            assert_eq!(k1, k2);
        }

        #[test]
        fn eq_trait() {
            assert_eq!(LlmErrorKind::Auth, LlmErrorKind::Auth);
            assert_ne!(LlmErrorKind::Auth, LlmErrorKind::Network);
        }

        #[test]
        fn all_variants_exist() {
            let kinds = [
                LlmErrorKind::Auth,
                LlmErrorKind::RateLimited,
                LlmErrorKind::ContextExceeded,
                LlmErrorKind::InvalidRequest,
                LlmErrorKind::ResponseFormat,
                LlmErrorKind::Network,
                LlmErrorKind::Stream,
                LlmErrorKind::HttpStatus,
                LlmErrorKind::Provider,
                LlmErrorKind::Internal,
                LlmErrorKind::NotSupported,
            ];
            assert_eq!(kinds.len(), 11);
        }
    }

    mod tool_error {
        use super::*;

        #[test]
        fn execution_creates_error() {
            let err = ToolError::execution("failed to run");
            assert!(matches!(err, ToolError::Execution(_)));
            assert!(err.to_string().contains("failed to run"));
        }

        #[test]
        fn invalid_args_creates_error() {
            let err = ToolError::invalid_args("missing field 'name'");
            assert!(matches!(err, ToolError::InvalidArguments(_)));
        }

        #[test]
        fn not_found_creates_error() {
            let err = ToolError::not_found("my_tool");
            assert!(matches!(err, ToolError::NotFound(_)));
            assert!(err.to_string().contains("my_tool"));
        }

        #[test]
        fn forbidden_creates_error() {
            let err = ToolError::forbidden("dangerous_tool");
            assert!(matches!(err, ToolError::Forbidden(_)));
            assert!(err.to_string().contains("forbidden"));
        }

        #[test]
        fn confirmation_denied_creates_error() {
            let err = ToolError::confirmation_denied("file_delete");
            assert!(matches!(err, ToolError::ConfirmationDenied(_)));
            assert!(err.to_string().contains("denied"));
        }

        #[test]
        fn not_initialized_variant() {
            let err = ToolError::NotInitialized;
            assert!(err.to_string().contains("not initialized"));
        }

        #[test]
        fn from_string() {
            let err: ToolError = "custom error".to_string().into();
            assert!(matches!(err, ToolError::Other(_)));
        }

        #[test]
        fn from_str() {
            let err: ToolError = "custom error".into();
            assert!(matches!(err, ToolError::Other(_)));
        }

        #[test]
        fn from_serde_json_error() {
            let json_err = serde_json::from_str::<i32>("invalid").unwrap_err();
            let err: ToolError = json_err.into();
            assert!(matches!(err, ToolError::InvalidArguments(_)));
        }

        #[test]
        fn clone_trait() {
            let err1 = ToolError::not_found("tool");
            let err2 = err1;
            assert!(matches!(err2, ToolError::NotFound(_)));
        }

        #[test]
        fn display_all_variants() {
            assert!(ToolError::execution("e").to_string().contains("Execution"));
            assert!(ToolError::invalid_args("a").to_string().contains("Invalid"));
            assert!(ToolError::not_found("n").to_string().contains("not found"));
            assert!(
                ToolError::NotInitialized
                    .to_string()
                    .contains("initialized")
            );
            assert!(ToolError::forbidden("f").to_string().contains("forbidden"));
            assert!(
                ToolError::confirmation_denied("c")
                    .to_string()
                    .contains("denied")
            );
            assert!(ToolError::Other("o".into()).to_string().contains("error"));
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn error_chain_llm_to_error() {
            fn inner() -> std::result::Result<(), LlmError> {
                Err(LlmError::network("test"))
            }

            fn outer() -> Result<()> {
                inner()?;
                Ok(())
            }

            let result = outer();
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), Error::Llm(_)));
        }

        #[test]
        fn error_chain_tool_to_error() {
            fn inner() -> std::result::Result<(), ToolError> {
                Err(ToolError::not_found("tool"))
            }

            fn outer() -> Result<()> {
                inner()?;
                Ok(())
            }

            let result = outer();
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), Error::Tool(_)));
        }

        #[test]
        fn llm_error_to_error_preserves_info() {
            let llm_err = LlmError::auth("openai", "bad key");
            let err: Error = llm_err.into();

            if let Error::Llm(inner) = err {
                assert_eq!(inner.kind, LlmErrorKind::Auth);
                assert_eq!(inner.provider.as_deref(), Some("openai"));
            } else {
                panic!("expected Error::Llm");
            }
        }
    }
}
