//! Structured errors and stable error codes.
//!
//! Control planes must branch on [`ErrorCode`] / [`RetryClass`], never on
//! substring matching of [`Display`](std::fmt::Display) output.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Stable error code for control-plane handling.
///
/// Codes use dotted `domain.reason` strings via [`ErrorCode::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorCode {
    // --- types ---
    /// Invalid or empty identifier.
    TypesInvalidId,
    /// Message or payload failed validation.
    TypesValidation,
    /// Serialization failure.
    TypesSerde,

    // --- tool ---
    /// Tool not found in registry.
    ToolNotFound,
    /// Tool arguments failed schema/parse.
    ToolInvalidArgs,
    /// Tool execution failed.
    ToolExecution,
    /// Tool timed out.
    ToolTimeout,
    /// Tool cancelled.
    ToolCancelled,
    /// Tool denied by policy/capability.
    ToolDenied,
    /// Tool call rejected by approval gate.
    ToolApprovalDenied,
    /// Tool stream ended without a terminal item (protocol violation).
    ToolStreamProtocol,
    /// Tool rate limited by upstream service.
    ToolRateLimited,
    /// Tool concurrency limit exceeded.
    ToolConcurrencyLimit,
    /// Tool network failure.
    ToolNetwork,
    /// Tool upstream service unavailable.
    ToolServiceUnavailable,

    // --- llm ---
    /// LLM transport or provider failure.
    LlmProvider,
    /// LLM request cancelled.
    LlmCancelled,
    /// LLM response invalid.
    LlmInvalidResponse,
    /// LLM authentication / authorization failure.
    LlmAuth,
    /// LLM rate limited.
    LlmRateLimit,
    /// Stream/sample idle timeout between chunks.
    LlmIdleTimeout,
    /// Provider returned an empty completion (no text / tool calls).
    LlmEmptyResponse,
    /// Output truncated (max tokens / length limit).
    LlmTruncated,

    // --- agent ---
    /// Agent definition invalid.
    AgentInvalidDefinition,
    /// Agent build failure.
    AgentBuild,
    /// Agent type / definition not found for resolution.
    AgentNotFound,

    // --- runtime / turn ---
    /// Turn hit max steps.
    RuntimeMaxSteps,
    /// Turn cancelled.
    RuntimeCancelled,
    /// Runtime gate rejected the outcome.
    RuntimeGate,
    /// Structured output failed schema validation after retries.
    RuntimeStructuredOutput,
    /// Turn deadline exceeded.
    RuntimeDeadline,
    /// Identical tool calls repeated past the stationarity hard stop.
    RuntimeStationarity,

    // --- host ---
    /// Host spawn failed.
    HostSpawn,
    /// Agent budget exhausted.
    HostBudget,
    /// Nested spawn depth exceeded.
    HostDepth,
    /// Concurrent nested agent cap exceeded.
    HostConcurrency,
    /// Host capability unsupported.
    HostUnsupported,
    /// Host cancelled.
    HostCancelled,
    /// Isolation backend failure.
    HostIsolation,

    // --- workflow ---
    /// Workflow script compile/runtime failure.
    WorkflowScript,
    /// Journal divergence on resume.
    WorkflowDivergence,
    /// Journal I/O or integrity failure.
    WorkflowJournal,
    /// Workflow agent budget exceeded.
    WorkflowBudget,
    /// Workflow cancelled.
    WorkflowCancelled,
    /// Workflow validation / probe failure.
    WorkflowValidate,

    // --- state / memory ---
    /// Conversation state invariant violated (e.g. dangling tool call).
    StateInvariant,
    /// Persistence backend I/O failure.
    StatePersistence,

    // --- compaction ---
    /// Compaction strategy failed.
    CompactionFailed,
    /// Context still exceeds limits after compaction.
    CompactionOverflow,

    /// Generic internal failure.
    Internal,
}

impl ErrorCode {
    /// Stable `snake_case` dotted code string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TypesInvalidId => "types.invalid_id",
            Self::TypesValidation => "types.validation",
            Self::TypesSerde => "types.serde",
            Self::ToolNotFound => "tool.not_found",
            Self::ToolInvalidArgs => "tool.invalid_args",
            Self::ToolExecution => "tool.execution",
            Self::ToolTimeout => "tool.timeout",
            Self::ToolCancelled => "tool.cancelled",
            Self::ToolDenied => "tool.denied",
            Self::ToolApprovalDenied => "tool.approval_denied",
            Self::ToolStreamProtocol => "tool.stream_protocol",
            Self::ToolRateLimited => "tool.rate_limited",
            Self::ToolConcurrencyLimit => "tool.concurrency_limit",
            Self::ToolNetwork => "tool.network",
            Self::ToolServiceUnavailable => "tool.service_unavailable",
            Self::LlmProvider => "llm.provider",
            Self::LlmCancelled => "llm.cancelled",
            Self::LlmInvalidResponse => "llm.invalid_response",
            Self::LlmAuth => "llm.auth",
            Self::LlmRateLimit => "llm.rate_limit",
            Self::LlmIdleTimeout => "llm.idle_timeout",
            Self::LlmEmptyResponse => "llm.empty_response",
            Self::LlmTruncated => "llm.truncated",
            Self::AgentInvalidDefinition => "agent.invalid_definition",
            Self::AgentBuild => "agent.build",
            Self::AgentNotFound => "agent.not_found",
            Self::RuntimeMaxSteps => "runtime.max_steps",
            Self::RuntimeCancelled => "runtime.cancelled",
            Self::RuntimeGate => "runtime.gate",
            Self::RuntimeStructuredOutput => "runtime.structured_output",
            Self::RuntimeDeadline => "runtime.deadline",
            Self::RuntimeStationarity => "runtime.stationarity",
            Self::HostSpawn => "host.spawn",
            Self::HostBudget => "host.budget",
            Self::HostDepth => "host.depth",
            Self::HostConcurrency => "host.concurrency",
            Self::HostUnsupported => "host.unsupported",
            Self::HostCancelled => "host.cancelled",
            Self::HostIsolation => "host.isolation",
            Self::WorkflowScript => "workflow.script",
            Self::WorkflowDivergence => "workflow.divergence",
            Self::WorkflowJournal => "workflow.journal",
            Self::WorkflowBudget => "workflow.budget",
            Self::WorkflowCancelled => "workflow.cancelled",
            Self::WorkflowValidate => "workflow.validate",
            Self::StateInvariant => "state.invariant",
            Self::StatePersistence => "state.persistence",
            Self::CompactionFailed => "compaction.failed",
            Self::CompactionOverflow => "compaction.overflow",
            Self::Internal => "internal",
        }
    }

    /// Domain prefix (`types`, `tool`, `llm`, …).
    #[must_use]
    pub const fn domain(self) -> &'static str {
        match self {
            Self::TypesInvalidId | Self::TypesValidation | Self::TypesSerde => "types",
            Self::ToolNotFound
            | Self::ToolInvalidArgs
            | Self::ToolExecution
            | Self::ToolTimeout
            | Self::ToolCancelled
            | Self::ToolDenied
            | Self::ToolApprovalDenied
            | Self::ToolStreamProtocol
            | Self::ToolRateLimited
            | Self::ToolConcurrencyLimit
            | Self::ToolNetwork
            | Self::ToolServiceUnavailable => "tool",
            Self::LlmProvider
            | Self::LlmCancelled
            | Self::LlmInvalidResponse
            | Self::LlmAuth
            | Self::LlmRateLimit
            | Self::LlmIdleTimeout
            | Self::LlmEmptyResponse
            | Self::LlmTruncated => "llm",
            Self::AgentInvalidDefinition | Self::AgentBuild | Self::AgentNotFound => "agent",
            Self::RuntimeMaxSteps
            | Self::RuntimeCancelled
            | Self::RuntimeGate
            | Self::RuntimeStructuredOutput
            | Self::RuntimeDeadline
            | Self::RuntimeStationarity => "runtime",
            Self::HostSpawn
            | Self::HostBudget
            | Self::HostDepth
            | Self::HostConcurrency
            | Self::HostUnsupported
            | Self::HostCancelled
            | Self::HostIsolation => "host",
            Self::WorkflowScript
            | Self::WorkflowDivergence
            | Self::WorkflowJournal
            | Self::WorkflowBudget
            | Self::WorkflowCancelled
            | Self::WorkflowValidate => "workflow",
            Self::StateInvariant | Self::StatePersistence => "state",
            Self::CompactionFailed | Self::CompactionOverflow => "compaction",
            Self::Internal => "internal",
        }
    }

    /// Default retry classification for this code.
    ///
    /// Kernel paths set an explicit [`RetryClass`] when they know more; this
    /// is the baseline hosts may consult.
    #[must_use]
    pub const fn default_retry(self) -> RetryClass {
        match self {
            Self::LlmRateLimit | Self::LlmProvider | Self::LlmEmptyResponse => RetryClass::Backoff,
            Self::LlmAuth => RetryClass::AuthRefresh,
            Self::ToolTimeout => RetryClass::Immediate,
            Self::ToolCancelled
            | Self::LlmCancelled
            | Self::LlmIdleTimeout
            | Self::LlmTruncated
            | Self::RuntimeCancelled
            | Self::HostCancelled
            | Self::WorkflowCancelled
            | Self::ToolDenied
            | Self::ToolApprovalDenied
            | Self::ToolNotFound
            | Self::ToolInvalidArgs
            | Self::ToolStreamProtocol
            | Self::TypesInvalidId
            | Self::TypesValidation
            | Self::TypesSerde
            | Self::AgentInvalidDefinition
            | Self::AgentBuild
            | Self::AgentNotFound
            | Self::RuntimeMaxSteps
            | Self::RuntimeGate
            | Self::RuntimeStructuredOutput
            | Self::RuntimeDeadline
            | Self::RuntimeStationarity
            | Self::HostBudget
            | Self::HostDepth
            | Self::HostConcurrency
            | Self::HostUnsupported
            | Self::WorkflowDivergence
            | Self::WorkflowBudget
            | Self::WorkflowValidate
            | Self::StateInvariant
            | Self::CompactionOverflow
            | Self::Internal => RetryClass::Never,
            Self::ToolExecution
            | Self::ToolRateLimited
            | Self::ToolConcurrencyLimit
            | Self::ToolNetwork
            | Self::ToolServiceUnavailable
            | Self::LlmInvalidResponse
            | Self::HostSpawn
            | Self::HostIsolation
            | Self::WorkflowScript
            | Self::WorkflowJournal
            | Self::StatePersistence
            | Self::CompactionFailed => RetryClass::Never,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Whether an automatic retry may be appropriate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RetryClass {
    /// Do not retry.
    #[default]
    Never,
    /// Safe to retry immediately.
    Immediate,
    /// Retry with backoff.
    Backoff,
    /// Refresh credentials then retry.
    AuthRefresh,
}

/// Kernel error with stable code, message, optional transport metadata, and source.
#[derive(Debug, Clone, thiserror::Error)]
pub struct OvoError {
    code: ErrorCode,
    message: String,
    retry: RetryClass,
    /// HTTP status when the error originated from a provider response.
    http_status: Option<u16>,
    /// Provider `Retry-After` duration when present.
    retry_after: Option<std::time::Duration>,
    /// Provider `x-should-retry` when present.
    x_should_retry: Option<bool>,
    #[source]
    source: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl OvoError {
    /// Create an error with code and message.
    ///
    /// Retry class defaults to [`ErrorCode::default_retry`].
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retry: code.default_retry(),
            http_status: None,
            retry_after: None,
            x_should_retry: None,
            source: None,
        }
    }

    /// Attach retry classification (overrides default).
    #[must_use]
    pub const fn with_retry(mut self, retry: RetryClass) -> Self {
        self.retry = retry;
        self
    }

    /// Attach HTTP status for retry policy.
    #[must_use]
    pub const fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    /// Attach `Retry-After` for rate-limit backoff.
    #[must_use]
    pub const fn with_retry_after(mut self, after: std::time::Duration) -> Self {
        self.retry_after = Some(after);
        self
    }

    /// Attach `x-should-retry` header semantics.
    #[must_use]
    pub const fn with_x_should_retry(mut self, value: bool) -> Self {
        self.x_should_retry = Some(value);
        self
    }

    /// Attach a source error.
    #[must_use]
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Arc::new(source));
        self
    }

    /// Stable code.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        self.code
    }

    /// Retry class.
    #[must_use]
    pub const fn retry_class(&self) -> RetryClass {
        self.retry
    }

    /// HTTP status when set.
    #[must_use]
    pub const fn http_status(&self) -> Option<u16> {
        self.http_status
    }

    /// Retry-After when set.
    #[must_use]
    pub const fn retry_after(&self) -> Option<std::time::Duration> {
        self.retry_after
    }

    /// `x-should-retry` when set.
    #[must_use]
    pub const fn x_should_retry(&self) -> Option<bool> {
        self.x_should_retry
    }

    /// Human-readable message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Runtime-domain cancellation.
    #[must_use]
    pub fn runtime_cancelled(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::RuntimeCancelled, message)
    }

    /// LLM-domain cancellation.
    #[must_use]
    pub fn llm_cancelled(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::LlmCancelled, message)
    }
}

impl fmt::Display for OvoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

/// Result alias using [`OvoError`].
pub type Result<T> = std::result::Result<T, OvoError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_code() {
        let err = OvoError::new(ErrorCode::ToolTimeout, "exceeded 5s");
        assert!(err.to_string().contains("tool.timeout"), "{err}");
        assert_eq!(err.retry_class(), RetryClass::Immediate);
    }

    #[test]
    fn rate_limit_defaults_to_backoff() {
        let err = OvoError::new(ErrorCode::LlmRateLimit, "429");
        assert_eq!(err.retry_class(), RetryClass::Backoff);
        assert_eq!(err.code().domain(), "llm");
    }

    #[test]
    fn all_codes_have_domain_prefix_in_as_str() {
        let codes = [
            ErrorCode::TypesInvalidId,
            ErrorCode::ToolApprovalDenied,
            ErrorCode::ToolStreamProtocol,
            ErrorCode::LlmAuth,
            ErrorCode::LlmRateLimit,
            ErrorCode::AgentNotFound,
            ErrorCode::RuntimeStructuredOutput,
            ErrorCode::RuntimeDeadline,
            ErrorCode::HostIsolation,
            ErrorCode::WorkflowValidate,
            ErrorCode::StateInvariant,
            ErrorCode::StatePersistence,
            ErrorCode::CompactionFailed,
            ErrorCode::CompactionOverflow,
            ErrorCode::Internal,
        ];
        for code in codes {
            let s = code.as_str();
            assert!(
                s.starts_with(code.domain()) || code == ErrorCode::Internal,
                "code {s} should start with domain {}",
                code.domain()
            );
        }
    }
}

include!("error_code_matrix.rs");
