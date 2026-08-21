//! Tool-domain errors.

use ovo_types::{ErrorCode, OvoError, RetryClass};

/// Tool error alias mapped into [`OvoError`].
pub type ToolError = OvoError;

/// Helpers for constructing tool errors.
pub mod codes {
    use super::{ErrorCode, OvoError, RetryClass};

    /// Tool missing from registry.
    #[must_use]
    pub fn not_found(name: &str) -> OvoError {
        OvoError::new(ErrorCode::ToolNotFound, format!("tool not found: {name}"))
    }

    /// Invalid arguments.
    #[must_use]
    pub fn invalid_args(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolInvalidArgs, msg)
    }

    /// Execution failure.
    #[must_use]
    pub fn execution(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolExecution, msg)
    }

    /// Timeout.
    #[must_use]
    pub fn timeout(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolTimeout, msg).with_retry(RetryClass::Backoff)
    }

    /// Cancelled.
    #[must_use]
    pub fn cancelled() -> OvoError {
        OvoError::new(ErrorCode::ToolCancelled, "tool call cancelled")
    }

    /// Policy deny.
    #[must_use]
    pub fn denied(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolDenied, msg)
    }

    /// Approval gate rejected the call.
    #[must_use]
    pub fn approval_denied(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolApprovalDenied, msg)
    }

    /// Stream protocol violation.
    #[must_use]
    pub fn stream_protocol(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolStreamProtocol, msg)
    }

    /// Rate limited by upstream.
    #[must_use]
    pub fn rate_limited(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolRateLimited, msg).with_retry(RetryClass::Backoff)
    }

    /// Concurrency limit.
    #[must_use]
    pub fn concurrency_limit(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolConcurrencyLimit, msg)
    }

    /// Network failure.
    #[must_use]
    pub fn network(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolNetwork, msg).with_retry(RetryClass::Backoff)
    }

    /// Service unavailable.
    #[must_use]
    pub fn service_unavailable(msg: impl Into<String>) -> OvoError {
        OvoError::new(ErrorCode::ToolServiceUnavailable, msg).with_retry(RetryClass::Backoff)
    }
}
