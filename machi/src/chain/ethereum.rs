//! Ethereum chain adapter.
//!
//! This module provides Ethereum blockchain integration using kobe-eth and alloy.

use std::str::FromStr;

use alloy::network::{EthereumWallet, TransactionBuilder};
use alloy::primitives::{Address, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;
use kobe_core::Wallet;
use kobe_eth::Deriver;

use crate::chain::{Chain, TransactionRequest, TxHash};
use crate::error::{Error, Result};

/// Ethereum network configuration.
#[derive(Debug, Clone)]
pub struct Ethereum {
    /// RPC endpoint URL.
    rpc_url: String,
    /// Chain ID.
    chain_id: u64,
}

impl Ethereum {
    /// Create a new Ethereum adapter with custom RPC endpoint.
    pub fn new(rpc_url: impl Into<String>, chain_id: u64) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            chain_id,
        }
    }

    /// Ethereum mainnet (chain ID 1).
    pub fn mainnet(rpc_url: impl Into<String>) -> Self {
        Self::new(rpc_url, 1)
    }

    /// Sepolia testnet (chain ID 11155111).
    pub fn sepolia(rpc_url: impl Into<String>) -> Self {
        Self::new(rpc_url, 11155111)
    }

    /// Get the RPC URL.
    pub fn rpc_url(&self) -> &str {
        &self.rpc_url
    }

    /// Get the chain ID.
    pub const fn chain_id(&self) -> u64 {
        self.chain_id
    }

    /// Create an alloy provider.
    fn provider(&self) -> Result<impl Provider + Clone> {
        let url = self
            .rpc_url
            .parse()
            .map_err(|e| Error::Chain(format!("invalid RPC URL: {e}")))?;
        let provider = ProviderBuilder::new().connect_http(url);
        Ok(provider)
    }

    /// Derive private key from wallet seed at index.
    fn derive_private_key(&self, wallet: &Wallet, index: u32) -> Result<PrivateKeySigner> {
        let deriver = Deriver::new(wallet);
        let derived = deriver.derive(0, false, index)?;
        let signer = PrivateKeySigner::from_str(&derived.private_key_hex)
            .map_err(|e| Error::Chain(format!("failed to create signer: {e}")))?;
        Ok(signer)
    }
}

impl Chain for Ethereum {
    type Address = String;

    fn name(&self) -> &'static str {
        "ethereum"
    }

    fn derive_address(&self, wallet: &Wallet, index: u32) -> Result<Self::Address> {
        let deriver = Deriver::new(wallet);
        let derived = deriver.derive(0, false, index)?;
        Ok(derived.address)
    }

    async fn balance(&self, address: &str) -> Result<u128> {
        let provider = self.provider()?;
        let addr = Address::from_str(address)
            .map_err(|e| Error::Chain(format!("invalid address: {e}")))?;
        let balance = provider
            .get_balance(addr)
            .await
            .map_err(|e| Error::Chain(format!("failed to get balance: {e}")))?;
        Ok(balance.to::<u128>())
    }

    async fn send_transaction(
        &self,
        wallet: &Wallet,
        index: u32,
        tx: TransactionRequest,
    ) -> Result<TxHash> {
        let signer = self.derive_private_key(wallet, index)?;
        let eth_wallet = EthereumWallet::from(signer);

        let url = self
            .rpc_url
            .parse()
            .map_err(|e| Error::Chain(format!("invalid RPC URL: {e}")))?;
        let provider = ProviderBuilder::new().wallet(eth_wallet).connect_http(url);

        let to_addr = Address::from_str(&tx.to)
            .map_err(|e| Error::Chain(format!("invalid recipient address: {e}")))?;

        let tx_request = alloy::rpc::types::TransactionRequest::default()
            .with_to(to_addr)
            .with_value(U256::from(tx.value));

        let tx_request = if let Some(data) = tx.data {
            tx_request.with_input(data)
        } else {
            tx_request
        };

        let pending = provider
            .send_transaction(tx_request)
            .await
            .map_err(|e| Error::Chain(format!("failed to send transaction: {e}")))?;

        let tx_hash = pending.tx_hash();
        Ok(TxHash(format!("{tx_hash:?}")))
    }
}
