//! Unified error type for the machi framework.
//!
//! This module provides the top-level [`Error`] enum that aggregates all
//! domain-specific errors from submodules. Each submodule owns its own
//! error type (e.g., [`LlmError`], [`ToolError`], [`AgentError`]), and
//! this module re-exports them for convenience.

// Re-export submodule error types for backward compatibility.
pub use crate::agent::AgentError;
pub use crate::llms::LlmError;
pub use crate::memory::MemoryError;
pub use crate::tool::ToolError;
#[cfg(feature = "wallet")]
pub use crate::wallet::WalletError;

/// Result type alias for machi operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The main error type for the machi framework.
///
/// Each variant wraps a domain-specific error owned by its respective module.
/// Use `#[from]` conversions for ergonomic `?` propagation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// LLM provider error (authentication, rate limiting, network, etc.).
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// Tool execution error.
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// Memory/session error.
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    /// Agent runtime error (config, steps, guardrails, interruption).
    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP request error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// HTTP middleware error (e.g., x402 payment).
    #[error("HTTP middleware error: {0}")]
    HttpMiddleware(#[from] reqwest_middleware::Error),

    /// Wallet operation error.
    #[cfg(feature = "wallet")]
    #[error("Wallet error: {0}")]
    Wallet(#[from] WalletError),
}
