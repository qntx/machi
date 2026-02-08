//! Error types for wallet operations.
//!
//! [`WalletError`] covers key derivation, signing, RPC provider communication,
//! configuration, and transaction submission failures.
//! It integrates into the global [`Error`](crate::Error) hierarchy via `Error::Wallet`.

/// Error type for wallet operations.
///
/// Covers key derivation, signing, RPC provider communication,
/// configuration, and transaction submission failures.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum WalletError {
    /// HD wallet / key derivation error.
    #[error("Derivation error: {0}")]
    Derivation(String),

    /// Signing error.
    #[error("Signing error: {0}")]
    Signing(String),

    /// RPC / provider error.
    #[error("Provider error: {0}")]
    Provider(String),

    /// Invalid configuration (missing fields, bad parameters).
    #[error("Config error: {0}")]
    Config(String),

    /// Transaction-related error.
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// x402 payment protocol error.
    #[cfg(feature = "x402")]
    #[error("x402 payment error: {0}")]
    Payment(String),

    /// ERC-8004 registry interaction error.
    #[cfg(feature = "erc8004")]
    #[error("ERC-8004 error: {0}")]
    Erc8004(String),
}

impl WalletError {
    /// Create a derivation error.
    #[must_use]
    pub fn derivation(msg: impl Into<String>) -> Self {
        Self::Derivation(msg.into())
    }

    /// Create a signing error.
    #[must_use]
    pub fn signing(msg: impl Into<String>) -> Self {
        Self::Signing(msg.into())
    }

    /// Create a provider error.
    #[must_use]
    pub fn provider(msg: impl Into<String>) -> Self {
        Self::Provider(msg.into())
    }

    /// Create a config error.
    #[must_use]
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a transaction error.
    #[must_use]
    pub fn transaction(msg: impl Into<String>) -> Self {
        Self::Transaction(msg.into())
    }

    /// Create an x402 payment error.
    #[cfg(feature = "x402")]
    #[must_use]
    pub fn payment(msg: impl Into<String>) -> Self {
        Self::Payment(msg.into())
    }

    /// Create an ERC-8004 registry error.
    #[cfg(feature = "erc8004")]
    #[must_use]
    pub fn erc8004(msg: impl Into<String>) -> Self {
        Self::Erc8004(msg.into())
    }
}

impl From<WalletError> for crate::tool::ToolError {
    fn from(e: WalletError) -> Self {
        Self::Execution(e.to_string())
    }
}
