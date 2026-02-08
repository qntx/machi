//! EVM wallet implementation composing [`kobe_eth`] and [`alloy`].
//!
//! [`EvmWallet`] directly embeds [`kobe_eth::DerivedAddress`] (with its
//! [`Zeroizing`] private key) alongside an [`alloy`] signer and provider,
//! preserving the full derivation metadata and security chain.

use std::sync::Arc;

use alloy::eips::BlockId;
use alloy::network::{Ethereum, TransactionBuilder};
use alloy::primitives::{Address, B256, Bytes, U256};
use alloy::providers::{DynProvider, Provider, ProviderBuilder};
use alloy::rpc::types::{Filter, Log, TransactionRequest};
use alloy::signers::local::PrivateKeySigner;
use alloy::signers::{Signer, SignerSync};
use tracing::info;

use crate::tool::BoxedTool;
use crate::wallet::{EvmChain, WalletError};

/// An EVM wallet for AI agent blockchain interactions.
///
/// Composes three layers:
/// - **[`kobe_eth::DerivedAddress`]** — HD derivation metadata with
///   [`Zeroizing`] private key (present when created via HD derivation)
/// - **[`alloy::signers::local::PrivateKeySigner`]** — EVM transaction
///   and message signing
/// - **[`alloy::providers::DynProvider`]** — JSON-RPC communication
pub struct EvmWallet {
    /// Derivation info from kobe-eth (present when created via HD derivation).
    ///
    /// Contains: derivation path, private key (Zeroizing), public key,
    /// and checksummed address. `None` when created from a raw private key.
    derived: Option<kobe_eth::DerivedAddress>,

    /// Alloy local signer for EVM signing operations.
    signer: PrivateKeySigner,

    /// Type-erased JSON-RPC provider for on-chain communication.
    provider: Arc<DynProvider<Ethereum>>,

    /// EIP-55 checksummed address.
    address: String,

    /// The chain this wallet is connected to.
    chain: EvmChain,
}

impl std::fmt::Debug for EvmWallet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvmWallet")
            .field("address", &self.address)
            .field("chain", &self.chain)
            .field(
                "derivation_path",
                &self.derived.as_ref().map(|d| d.path.as_str()),
            )
            .finish_non_exhaustive()
    }
}

/// Format a signature as a `0x`-prefixed hex string.
fn format_signature(bytes: &[u8]) -> String {
    format!("0x{}", alloy::primitives::hex::encode(bytes))
}

impl EvmWallet {
    /// Create a wallet from a BIP39 mnemonic phrase — derive + connect in one step.
    ///
    /// This is the **recommended** constructor for most users. It internally
    /// creates a [`kobe::Wallet`], derives an Ethereum key via [`kobe_eth::Deriver`],
    /// and connects to the RPC endpoint.
    ///
    /// # Arguments
    ///
    /// * `mnemonic` — BIP39 mnemonic phrase.
    /// * `passphrase` — Optional BIP39 passphrase ("25th word").
    /// * `index` — HD derivation index (Standard path: `m/44'/60'/0'/0/{index}`).
    /// * `rpc_url` — JSON-RPC endpoint URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the mnemonic is invalid, derivation fails, or RPC connection fails.
    pub async fn from_mnemonic(
        mnemonic: &str,
        passphrase: Option<&str>,
        index: u32,
        rpc_url: &str,
    ) -> crate::Result<Self> {
        let hd = kobe::Wallet::from_mnemonic(mnemonic, passphrase)
            .map_err(|e| WalletError::derivation(format!("invalid mnemonic: {e}")))?;
        Self::from_wallet(&hd, index, rpc_url).await
    }

    /// Create a wallet from a BIP39 mnemonic with a specific [`DerivationStyle`].
    ///
    /// Convenience wrapper combining [`from_mnemonic`](Self::from_mnemonic) flexibility
    /// with [`from_wallet_with`](Self::from_wallet_with) derivation style control.
    ///
    /// # Errors
    ///
    /// Returns an error if the mnemonic is invalid, derivation fails, or RPC connection fails.
    pub async fn from_mnemonic_with(
        mnemonic: &str,
        passphrase: Option<&str>,
        style: kobe_eth::DerivationStyle,
        index: u32,
        rpc_url: &str,
    ) -> crate::Result<Self> {
        let hd = kobe::Wallet::from_mnemonic(mnemonic, passphrase)
            .map_err(|e| WalletError::derivation(format!("invalid mnemonic: {e}")))?;
        Self::from_wallet_with(&hd, style, index, rpc_url).await
    }

    /// Create a wallet from a [`kobe::Wallet`] (HD wallet) with Standard derivation.
    ///
    /// Use this when you already hold a `kobe::Wallet` — e.g., for multi-chain
    /// scenarios where the same mnemonic derives both EVM and Solana keys.
    ///
    /// # Errors
    ///
    /// Returns an error if derivation fails or RPC connection fails.
    pub async fn from_wallet(
        wallet: &kobe::Wallet,
        index: u32,
        rpc_url: &str,
    ) -> crate::Result<Self> {
        let derived = kobe_eth::Deriver::new(wallet)
            .derive(index)
            .map_err(|e| WalletError::derivation(format!("ETH derivation failed: {e}")))?;
        Self::from_derived(derived, rpc_url).await
    }

    /// Create a wallet from a [`kobe::Wallet`] with a specific [`DerivationStyle`].
    ///
    /// Supports Standard (`MetaMask`/Trezor), Ledger Live, and Ledger Legacy paths.
    ///
    /// [`DerivationStyle`]: kobe_eth::DerivationStyle
    ///
    /// # Errors
    ///
    /// Returns an error if derivation fails or RPC connection fails.
    pub async fn from_wallet_with(
        wallet: &kobe::Wallet,
        style: kobe_eth::DerivationStyle,
        index: u32,
        rpc_url: &str,
    ) -> crate::Result<Self> {
        let derived = kobe_eth::Deriver::new(wallet)
            .derive_with(style, index)
            .map_err(|e| WalletError::derivation(format!("ETH derivation failed: {e}")))?;
        Self::from_derived(derived, rpc_url).await
    }

    /// Create a wallet from a pre-derived [`kobe_eth::DerivedAddress`].
    ///
    /// Use this when you have already derived the key externally.
    ///
    /// # Errors
    ///
    /// Returns an error if the signer cannot be created or RPC connection fails.
    pub async fn from_derived(
        derived: kobe_eth::DerivedAddress,
        rpc_url: &str,
    ) -> crate::Result<Self> {
        let signer: PrivateKeySigner = derived
            .private_key_hex
            .parse()
            .map_err(|e| WalletError::derivation(format!("signer creation failed: {e}")))?;
        let address = derived.address.clone();

        Self::build(Some(derived), signer, address, rpc_url).await
    }

    /// Create a wallet from a [`kobe_eth::StandardWallet`] (non-HD, random key).
    ///
    /// # Errors
    ///
    /// Returns an error if signer creation or RPC connection fails.
    pub async fn from_standard(
        standard: &kobe_eth::StandardWallet,
        rpc_url: &str,
    ) -> crate::Result<Self> {
        let signer: PrivateKeySigner = standard
            .secret_hex()
            .parse()
            .map_err(|e| WalletError::derivation(format!("signer creation failed: {e}")))?;
        let address = signer.address().to_checksum(None);

        Self::build(None, signer, address, rpc_url).await
    }

    /// Create a wallet from a raw private key hex string (with or without `0x` prefix).
    ///
    /// Use this for development, testing, or when importing a standalone key.
    /// No HD derivation metadata is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if the private key is invalid or RPC connection fails.
    pub async fn from_private_key(key: &str, rpc_url: &str) -> crate::Result<Self> {
        let key = key.strip_prefix("0x").unwrap_or(key);
        let signer: PrivateKeySigner = key
            .parse()
            .map_err(|e| WalletError::config(format!("invalid private key: {e}")))?;
        let address = signer.address().to_checksum(None);

        Self::build(None, signer, address, rpc_url).await
    }

    /// Internal builder shared by all constructors.
    async fn build(
        derived: Option<kobe_eth::DerivedAddress>,
        signer: PrivateKeySigner,
        address: String,
        rpc_url: &str,
    ) -> crate::Result<Self> {
        let provider: DynProvider<Ethereum> = ProviderBuilder::new()
            .wallet(signer.clone())
            .connect(rpc_url)
            .await
            .map_err(|e| {
                WalletError::provider(format!("RPC connection to '{rpc_url}' failed: {e}"))
            })?
            .erased();

        let chain_id = provider
            .get_chain_id()
            .await
            .map_err(|e| WalletError::provider(format!("failed to get chain ID: {e}")))?;

        let chain = EvmChain::from_id(chain_id);

        info!(
            address = %address,
            chain = %chain,
            path = ?derived.as_ref().map(|d| d.path.as_str()),
            "EVM wallet connected",
        );

        Ok(Self {
            derived,
            signer,
            provider: Arc::new(provider),
            address,
            chain,
        })
    }

    /// Override the detected chain after construction.
    ///
    /// Useful when the auto-detected chain needs a custom name.
    #[must_use]
    pub fn with_chain(mut self, chain: EvmChain) -> Self {
        self.chain = chain;
        self
    }
}

impl EvmWallet {
    /// Reference to the underlying JSON-RPC [`DynProvider`].
    #[must_use]
    pub fn provider(&self) -> &DynProvider<Ethereum> {
        &self.provider
    }

    /// Reference to the underlying [`PrivateKeySigner`].
    #[must_use]
    pub const fn signer(&self) -> &PrivateKeySigner {
        &self.signer
    }

    /// Numeric chain ID.
    #[must_use]
    pub const fn chain_id(&self) -> u64 {
        self.chain.id()
    }

    /// Human-readable chain name.
    #[must_use]
    pub fn chain_name(&self) -> &str {
        self.chain.name()
    }

    /// Reference to the [`EvmChain`] this wallet is connected to.
    #[must_use]
    pub const fn chain(&self) -> &EvmChain {
        &self.chain
    }

    /// EIP-55 checksummed address string.
    #[must_use]
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Typed [`Address`] value for use with alloy APIs.
    #[must_use]
    pub const fn address_typed(&self) -> Address {
        self.signer.address()
    }

    /// Public key hex string, if created via HD derivation.
    #[must_use]
    pub fn public_key(&self) -> Option<&str> {
        self.derived.as_ref().map(|d| d.public_key_hex.as_str())
    }

    /// HD derivation path (e.g., `m/44'/60'/0'/0/0`), if created via HD derivation.
    #[must_use]
    pub fn derivation_path(&self) -> Option<&str> {
        self.derived.as_ref().map(|d| d.path.as_str())
    }

    /// Reference to the kobe-eth derivation result, if available.
    #[must_use]
    pub const fn derived(&self) -> Option<&kobe_eth::DerivedAddress> {
        self.derived.as_ref()
    }
}

impl EvmWallet {
    /// Sign an arbitrary message using EIP-191 `personal_sign`.
    ///
    /// Returns the `0x`-prefixed hex-encoded signature.
    ///
    /// # Errors
    ///
    /// Returns an error if signing fails.
    pub async fn sign_message(&self, message: &[u8]) -> Result<String, WalletError> {
        let sig = self
            .signer
            .sign_message(message)
            .await
            .map_err(|e| WalletError::signing(format!("sign_message failed: {e}")))?;
        Ok(format_signature(&sig.as_bytes()))
    }

    /// Sign an arbitrary message synchronously (EIP-191).
    ///
    /// # Errors
    ///
    /// Returns an error if signing fails.
    pub fn sign_message_sync(&self, message: &[u8]) -> Result<String, WalletError> {
        let sig = self
            .signer
            .sign_message_sync(message)
            .map_err(|e| WalletError::signing(format!("sign_message_sync failed: {e}")))?;
        Ok(format_signature(&sig.as_bytes()))
    }

    /// Sign EIP-712 typed structured data.
    ///
    /// Used for `DeFi` protocols like ERC-20 Permit, Uniswap Permit2,
    /// and off-chain order books. The caller must provide a type that
    /// implements [`alloy::sol_types::SolStruct`] and an [`Eip712Domain`].
    ///
    /// Returns the `0x`-prefixed hex-encoded signature.
    ///
    /// [`Eip712Domain`]: alloy::sol_types::Eip712Domain
    ///
    /// # Errors
    ///
    /// Returns an error if signing fails.
    pub async fn sign_typed_data<T: alloy::sol_types::SolStruct + Sync>(
        &self,
        payload: &T,
        domain: &alloy::sol_types::Eip712Domain,
    ) -> Result<String, WalletError> {
        let hash = payload.eip712_signing_hash(domain);
        let sig = self
            .signer
            .sign_hash(&hash)
            .await
            .map_err(|e| WalletError::signing(format!("sign_typed_data failed: {e}")))?;
        Ok(format_signature(&sig.as_bytes()))
    }

    /// Sign EIP-712 typed structured data synchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if signing fails.
    pub fn sign_typed_data_sync<T: alloy::sol_types::SolStruct + Sync>(
        &self,
        payload: &T,
        domain: &alloy::sol_types::Eip712Domain,
    ) -> Result<String, WalletError> {
        let hash = payload.eip712_signing_hash(domain);
        let sig = self
            .signer
            .sign_hash_sync(&hash)
            .map_err(|e| WalletError::signing(format!("sign_typed_data_sync failed: {e}")))?;
        Ok(format_signature(&sig.as_bytes()))
    }
}

impl EvmWallet {
    /// Get native token balance for the wallet's own address (in wei).
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn balance(&self) -> Result<U256, WalletError> {
        self.balance_of(self.signer.address()).await
    }

    /// Get native token balance for any address (in wei).
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn balance_of(&self, address: Address) -> Result<U256, WalletError> {
        self.provider
            .get_balance(address)
            .await
            .map_err(|e| WalletError::provider(format!("get_balance failed: {e}")))
    }

    /// Get the latest block number.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn block_number(&self) -> Result<u64, WalletError> {
        self.provider
            .get_block_number()
            .await
            .map_err(|e| WalletError::provider(format!("get_block_number failed: {e}")))
    }

    /// Get the current gas price in wei.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn gas_price(&self) -> Result<u128, WalletError> {
        self.provider
            .get_gas_price()
            .await
            .map_err(|e| WalletError::provider(format!("get_gas_price failed: {e}")))
    }

    /// Get the transaction count (nonce) for the wallet's address.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn nonce(&self) -> Result<u64, WalletError> {
        self.provider
            .get_transaction_count(self.signer.address())
            .await
            .map_err(|e| WalletError::provider(format!("get_transaction_count failed: {e}")))
    }

    /// Get the bytecode at an address (`eth_getCode`).
    ///
    /// Returns empty bytes for EOA (non-contract) addresses.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn code_at(&self, address: Address) -> Result<Bytes, WalletError> {
        self.provider
            .get_code_at(address)
            .await
            .map_err(|e| WalletError::provider(format!("get_code failed: {e}")))
    }

    /// Get a transaction receipt by its hash.
    ///
    /// Returns `None` if the transaction is not yet mined.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn transaction_receipt(
        &self,
        hash: B256,
    ) -> Result<Option<alloy::rpc::types::TransactionReceipt>, WalletError> {
        self.provider
            .get_transaction_receipt(hash)
            .await
            .map_err(|e| WalletError::provider(format!("get_transaction_receipt failed: {e}")))
    }

    /// Get a transaction by its hash.
    ///
    /// Returns `None` if the transaction is not found.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn transaction_by_hash(
        &self,
        hash: B256,
    ) -> Result<Option<alloy::rpc::types::Transaction>, WalletError> {
        self.provider
            .get_transaction_by_hash(hash)
            .await
            .map_err(|e| WalletError::provider(format!("get_transaction_by_hash failed: {e}")))
    }

    /// Query historical event logs matching a [`Filter`].
    ///
    /// Essential for tracking ERC-20 transfers, DEX swaps, and other
    /// on-chain events.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn logs(&self, filter: &Filter) -> Result<Vec<Log>, WalletError> {
        self.provider
            .get_logs(filter)
            .await
            .map_err(|e| WalletError::provider(format!("get_logs failed: {e}")))
    }

    /// Read a raw storage slot from a contract (`eth_getStorageAt`).
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn storage_at(&self, address: Address, key: U256) -> Result<U256, WalletError> {
        self.provider
            .get_storage_at(address, key)
            .await
            .map_err(|e| WalletError::provider(format!("get_storage_at failed: {e}")))
    }

    /// Get a block by [`BlockId`] (number, hash, or tag like `latest`).
    ///
    /// Returns `None` if the block is not found. By default only includes
    /// transaction hashes; use `provider()` directly for full transactions.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call fails.
    pub async fn block(
        &self,
        id: BlockId,
    ) -> Result<Option<alloy::rpc::types::Block>, WalletError> {
        self.provider
            .get_block(id)
            .await
            .map_err(|e| WalletError::provider(format!("get_block failed: {e}")))
    }

    /// Estimate EIP-1559 fee parameters (`max_fee_per_gas`, `max_priority_fee_per_gas`).
    ///
    /// More accurate than [`gas_price()`](Self::gas_price) for EIP-1559 chains.
    /// Returns a tuple of `(max_fee_per_gas, max_priority_fee_per_gas)` in wei.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call or estimation fails.
    pub async fn estimate_eip1559_fees(&self) -> Result<(u128, u128), WalletError> {
        let fees = self
            .provider
            .estimate_eip1559_fees()
            .await
            .map_err(|e| WalletError::provider(format!("estimate_eip1559_fees failed: {e}")))?;
        Ok((fees.max_fee_per_gas, fees.max_priority_fee_per_gas))
    }
}

impl EvmWallet {
    /// Transfer native token (ETH) to an address.
    ///
    /// Returns the `0x`-prefixed transaction hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction fails to send or confirm.
    pub async fn transfer(&self, to: Address, value: U256) -> Result<String, WalletError> {
        let tx = TransactionRequest::default().with_to(to).with_value(value);
        self.send_transaction(tx).await
    }

    /// Send an arbitrary transaction and wait for the receipt.
    ///
    /// Returns the `0x`-prefixed transaction hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction fails to send or confirm.
    pub async fn send_transaction(&self, tx: TransactionRequest) -> Result<String, WalletError> {
        let receipt = self
            .provider
            .send_transaction(tx)
            .await
            .map_err(|e| WalletError::transaction(format!("send_transaction failed: {e}")))?
            .get_receipt()
            .await
            .map_err(|e| WalletError::transaction(format!("get_receipt failed: {e}")))?;

        Ok(format!("{:#x}", receipt.transaction_hash))
    }

    /// Estimate gas for a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if gas estimation fails.
    pub async fn estimate_gas(&self, tx: TransactionRequest) -> Result<u64, WalletError> {
        self.provider
            .estimate_gas(tx)
            .await
            .map_err(|e| WalletError::transaction(format!("estimate_gas failed: {e}")))
    }

    /// Execute a read-only contract call (`eth_call`).
    ///
    /// Does not submit a transaction — useful for querying contract state.
    ///
    /// # Errors
    ///
    /// Returns an error if the call fails.
    pub async fn call(&self, tx: TransactionRequest) -> Result<Bytes, WalletError> {
        self.provider
            .call(tx)
            .await
            .map_err(|e| WalletError::provider(format!("eth_call failed: {e}")))
    }
}

impl EvmWallet {
    /// Create an x402-enabled HTTP client from this wallet's signer.
    ///
    /// The returned client transparently handles HTTP 402 responses by
    /// signing ERC-3009 payment authorizations using this wallet's
    /// [`PrivateKeySigner`]. No gas is consumed for payment signing.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use machi::wallet::EvmWallet;
    ///
    /// # async fn example() -> machi::Result<()> {
    /// let wallet = EvmWallet::from_private_key("0x...", "https://base-rpc.example.com").await?;
    /// let client = wallet.x402_client();
    /// let body = client.get("https://api.example.com/paid").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "x402")]
    #[must_use]
    pub fn x402_client(&self) -> super::x402::X402HttpClient {
        super::x402::X402HttpClient::from_wallet(self)
    }

    /// Convert this wallet into agent-callable tools, consuming it.
    ///
    /// Provides tools for:
    /// - `get_wallet_info` — address, chain, derivation path
    /// - `get_balance` — query native token balance
    /// - `get_nonce` — transaction count / sequence number
    /// - `get_block_number` — latest block number
    /// - `get_gas_price` — gas price and EIP-1559 fee estimates
    /// - `is_contract` — check if an address has bytecode
    /// - `get_transaction_receipt` — receipt for a mined transaction
    /// - `get_transaction` — transaction details by hash
    /// - `sign_message` — EIP-191 personal sign
    /// - `transfer` — send native token
    /// - `erc20_balance` — ERC-20 token balance with symbol/decimals
    /// - `erc20_transfer` — transfer ERC-20 tokens
    /// - `resolve_ens` — ENS name → address (mainnet only)
    /// - `reverse_ens` — address → ENS name (mainnet only)
    #[must_use]
    pub fn into_tools(self) -> Vec<BoxedTool> {
        let wallet = Arc::new(self);
        Self::shared_tools(&wallet)
    }

    /// Create agent-callable tools from a shared wallet reference.
    ///
    /// Use this when you need to retain direct access to the wallet
    /// while also providing tools to an agent.
    #[must_use]
    pub fn shared_tools(wallet: &Arc<Self>) -> Vec<BoxedTool> {
        super::tools::create_tools(wallet)
    }
}

#[async_trait::async_trait]
impl crate::wallet::Wallet for EvmWallet {
    fn address(&self) -> &str {
        &self.address
    }

    fn chain(&self) -> &str {
        self.chain.name()
    }

    fn chain_id(&self) -> Option<u64> {
        Some(self.chain.id())
    }

    async fn sign_message(&self, message: &[u8]) -> Result<String, WalletError> {
        self.sign_message(message).await
    }
}
