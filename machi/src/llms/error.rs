//! Error types for LLM provider operations.
//!
//! [`LlmError`] covers all failure modes when communicating with language model
//! backends (authentication, rate limiting, network issues, etc.).
//! It integrates into the global [`Error`](crate::Error) hierarchy via `Error::Llm`.

/// Error type for LLM provider operations.
///
/// Each variant represents a distinct failure mode, enabling callers to
/// pattern-match on specific cases (e.g., retrying transient errors).
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum LlmError {
    /// Authentication or authorization failure.
    #[error("[{provider}] {message}")]
    Auth {
        /// Provider name (e.g., "openai", "ollama").
        provider: String,
        /// Error description.
        message: String,
    },

    /// Rate limit exceeded.
    #[error("[{provider}] Rate limit exceeded. Please retry after some time.")]
    RateLimited {
        /// Provider name.
        provider: String,
    },

    /// Context length exceeded.
    #[error("Context length exceeded: used {used}, max {max}")]
    ContextExceeded {
        /// Tokens used.
        used: usize,
        /// Maximum allowed tokens.
        max: usize,
    },

    /// Response format error.
    #[error("Expected {expected}, got {got}")]
    ResponseFormat {
        /// Expected format description.
        expected: String,
        /// Actual format received.
        got: String,
    },

    /// Network or connection error.
    #[error("{0}")]
    Network(String),

    /// Streaming error.
    #[error("{0}")]
    Stream(String),

    /// HTTP status error.
    #[error("HTTP {status}: {body}")]
    HttpStatus {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },

    /// Provider-specific error.
    #[error("[{provider}] {message}")]
    Provider {
        /// Provider name.
        provider: String,
        /// Error description.
        message: String,
        /// Optional error code from the provider.
        code: Option<String>,
    },

    /// Internal error.
    #[error("{0}")]
    Internal(String),

    /// Feature not supported.
    #[error("Feature not supported: {0}")]
    NotSupported(String),
}

impl LlmError {
    /// Create an authentication error.
    #[must_use]
    pub fn auth(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Auth {
            provider: provider.into(),
            message: message.into(),
        }
    }

    /// Create a rate limit error.
    #[must_use]
    pub fn rate_limited(provider: impl Into<String>) -> Self {
        Self::RateLimited {
            provider: provider.into(),
        }
    }

    /// Create a context exceeded error.
    #[must_use]
    pub const fn context_exceeded(used: usize, max: usize) -> Self {
        Self::ContextExceeded { used, max }
    }

    /// Create a response format error.
    #[must_use]
    pub fn response_format(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::ResponseFormat {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Create a network error.
    #[must_use]
    pub fn network(message: impl Into<String>) -> Self {
        Self::Network(message.into())
    }

    /// Create a streaming error.
    #[must_use]
    pub fn stream(message: impl Into<String>) -> Self {
        Self::Stream(message.into())
    }

    /// Create an HTTP status error.
    #[must_use]
    pub fn http_status(status: u16, body: impl Into<String>) -> Self {
        Self::HttpStatus {
            status,
            body: body.into(),
        }
    }

    /// Create a provider-specific error.
    #[must_use]
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
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
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
            code: Some(code.into()),
        }
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    /// Create a not supported error.
    #[must_use]
    pub fn not_supported(feature: impl Into<String>) -> Self {
        Self::NotSupported(feature.into())
    }

    /// Check if this is a retryable error.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimited { .. } | Self::Network(_))
    }
}

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
