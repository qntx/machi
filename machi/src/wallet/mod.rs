//! Wallet module for AI agent blockchain interactions.
//!
//! This module provides wallet capabilities that allow machi agents to
//! autonomously interact with blockchains — reading on-chain data, signing
//! messages, and submitting transactions.
//!
//! # Architecture
//!
//! ```text
//! kobe::Wallet (BIP39 mnemonic + seed, Zeroize-protected)
//!   → EvmWallet::from_wallet() → kobe_eth derivation + alloy signer + JSON-RPC
//!     → Agent::wallet() → auto-registered tools for the AI agent
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use machi::agent::Agent;
//! use machi::wallet::EvmWallet;
//!
//! # async fn example() -> machi::Result<()> {
//! // From mnemonic — one step (derive + connect)
//! let wallet = EvmWallet::from_mnemonic(
//!     "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
//!     None,
//!     0,
//!     "https://eth.llamarpc.com",
//! ).await?;
//!
//! // Attach to an agent (tools auto-registered)
//! let agent = Agent::new("defi-bot")
//!     .wallet(wallet)
//!     .instructions("You are a DeFi assistant.");
//! # Ok(())
//! # }
//! ```

use std::fmt;

use async_trait::async_trait;

mod error;
pub mod evm;

pub use error::WalletError;
pub use evm::EvmWallet;
#[cfg(feature = "x402")]
pub use evm::x402::X402HttpClient;

// Re-export kobe for direct use as the multi-chain HD wallet.
pub use kobe::Wallet as HdWallet;

// Re-export kobe-eth types used in the public API.
pub use kobe_eth::{DerivationStyle, DerivedAddress};

/// Known EVM-compatible chains with their chain IDs and human-readable names.
///
/// Use this to explicitly specify a chain when constructing an [`EvmWallet`],
/// or to look up chain metadata by ID.
///
/// # Examples
///
/// ```rust
/// use machi::wallet::EvmChain;
///
/// let chain = EvmChain::Ethereum;
/// assert_eq!(chain.id(), 1);
/// assert_eq!(chain.name(), "ethereum");
///
/// // Infer from chain ID
/// let chain = EvmChain::from_id(137);
/// assert_eq!(chain.name(), "polygon");
///
/// // Custom / unknown chains
/// let chain = EvmChain::from_id(999);
/// assert_eq!(chain.name(), "evm-999");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvmChain {
    /// Ethereum Mainnet (chain ID: 1).
    Ethereum,
    /// Sepolia Testnet (chain ID: 11155111).
    Sepolia,
    /// Optimism (chain ID: 10).
    Optimism,
    /// BNB Smart Chain (chain ID: 56).
    Bsc,
    /// Gnosis Chain (chain ID: 100).
    Gnosis,
    /// Polygon `PoS` (chain ID: 137).
    Polygon,
    /// Fantom Opera (chain ID: 250).
    Fantom,
    /// `ZkSync` Era (chain ID: 324).
    ZkSync,
    /// Base (chain ID: 8453).
    Base,
    /// Arbitrum One (chain ID: 42161).
    Arbitrum,
    /// Avalanche C-Chain (chain ID: 43114).
    Avalanche,
    /// Linea (chain ID: 59144).
    Linea,
    /// Scroll (chain ID: 534352).
    Scroll,
    /// Monad (chain ID: 143).
    Monad,
    /// Custom chain with user-specified ID and name.
    Custom {
        /// Numeric chain ID.
        id: u64,
        /// Human-readable chain name.
        name: String,
    },
}

impl EvmChain {
    /// Get the numeric chain ID.
    #[must_use]
    pub const fn id(&self) -> u64 {
        match self {
            Self::Ethereum => 1,
            Self::Sepolia => 11_155_111,
            Self::Optimism => 10,
            Self::Bsc => 56,
            Self::Gnosis => 100,
            Self::Polygon => 137,
            Self::Fantom => 250,
            Self::ZkSync => 324,
            Self::Base => 8453,
            Self::Arbitrum => 42_161,
            Self::Avalanche => 43_114,
            Self::Linea => 59_144,
            Self::Scroll => 534_352,
            Self::Monad => 143,
            Self::Custom { id, .. } => *id,
        }
    }

    /// Get the human-readable chain name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Ethereum => "ethereum",
            Self::Sepolia => "sepolia",
            Self::Optimism => "optimism",
            Self::Bsc => "bsc",
            Self::Gnosis => "gnosis",
            Self::Polygon => "polygon",
            Self::Fantom => "fantom",
            Self::ZkSync => "zksync",
            Self::Base => "base",
            Self::Arbitrum => "arbitrum",
            Self::Avalanche => "avalanche",
            Self::Linea => "linea",
            Self::Scroll => "scroll",
            Self::Monad => "monad",
            Self::Custom { name, .. } => name,
        }
    }

    /// Infer an [`EvmChain`] from a numeric chain ID.
    ///
    /// Unknown chain IDs produce [`EvmChain::Custom`] with a generic name.
    #[must_use]
    pub fn from_id(id: u64) -> Self {
        match id {
            1 => Self::Ethereum,
            10 => Self::Optimism,
            56 => Self::Bsc,
            100 => Self::Gnosis,
            137 => Self::Polygon,
            250 => Self::Fantom,
            324 => Self::ZkSync,
            8453 => Self::Base,
            42_161 => Self::Arbitrum,
            43_114 => Self::Avalanche,
            59_144 => Self::Linea,
            143 => Self::Monad,
            534_352 => Self::Scroll,
            11_155_111 => Self::Sepolia,
            _ => Self::Custom {
                id,
                name: format!("evm-{id}"),
            },
        }
    }
}

impl fmt::Display for EvmChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.id())
    }
}

/// Chain-agnostic wallet trait for blockchain interactions.
///
/// All chain-specific wallets implement this trait, providing a
/// uniform interface for address access, signing, and chain metadata.
///
/// Currently implemented by [`EvmWallet`]. Designed for future
/// extension to Solana, Bitcoin, and other chains via `kobe-sol` / `kobe-btc`.
#[async_trait]
pub trait Wallet: Send + Sync + fmt::Debug {
    /// Display-friendly address string (e.g., EIP-55 checksummed hex for EVM).
    fn address(&self) -> &str;

    /// Chain identifier (e.g., `"ethereum"`, `"polygon"`, `"solana"`).
    fn chain(&self) -> &str;

    /// Numeric chain ID, if applicable (EVM chains).
    ///
    /// Returns `None` for non-EVM chains (e.g., Solana).
    fn chain_id(&self) -> Option<u64> {
        None
    }

    /// Sign an arbitrary message, returning the hex-encoded signature.
    ///
    /// # Errors
    ///
    /// Returns an error if the signing operation fails.
    async fn sign_message(&self, message: &[u8]) -> Result<String, WalletError>;
}
