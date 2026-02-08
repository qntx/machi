//! ERC-20 token interactions for [`EvmWallet`].
//!
//! Provides balance queries, transfers, and metadata lookups (symbol,
//! decimals) by encoding standard ERC-20 ABI calls and dispatching them
//! through the wallet's provider.

use alloy::network::TransactionBuilder;
use alloy::primitives::{Address, Bytes, U256};
use alloy::rpc::types::TransactionRequest;
use alloy::sol_types::SolCall;

use super::wallet::EvmWallet;
use crate::wallet::WalletError;

/// Minimal ERC-20 ABI fragments for balance queries and transfers.
mod abi {
    alloy::sol! {
        function balanceOf(address owner) external view returns (uint256);
        function transfer(address to, uint256 amount) external returns (bool);
        function decimals() external view returns (uint8);
        function symbol() external view returns (string);
    }
}

impl EvmWallet {
    /// Query the ERC-20 token balance of `owner` on contract `token`.
    ///
    /// Returns the raw token amount (not adjusted for decimals).
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call or ABI decoding fails.
    pub async fn erc20_balance(&self, token: Address, owner: Address) -> Result<U256, WalletError> {
        let calldata = abi::balanceOfCall { owner }.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(token)
            .with_input(Bytes::from(calldata));
        let result = self.call(tx).await?;

        abi::balanceOfCall::abi_decode_returns(&result)
            .map_err(|e| WalletError::provider(format!("ERC-20 balanceOf decode failed: {e}")))
    }

    /// Transfer ERC-20 tokens to `to` from the agent's address.
    ///
    /// `amount` is the raw token amount (not adjusted for decimals).
    /// Returns the `0x`-prefixed transaction hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction fails, or the contract
    /// returns `false` (non-standard tokens may not revert on failure).
    pub async fn erc20_transfer(
        &self,
        token: Address,
        to: Address,
        amount: U256,
    ) -> Result<String, WalletError> {
        let calldata = abi::transferCall { to, amount }.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(token)
            .with_input(Bytes::from(calldata));
        self.send_transaction(tx).await
    }

    /// Query the decimals of an ERC-20 token.
    ///
    /// Most tokens return 18; stablecoins like USDC/USDT return 6.
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call or ABI decoding fails.
    pub async fn erc20_decimals(&self, token: Address) -> Result<u8, WalletError> {
        let calldata = abi::decimalsCall {}.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(token)
            .with_input(Bytes::from(calldata));
        let result = self.call(tx).await?;

        abi::decimalsCall::abi_decode_returns(&result)
            .map_err(|e| WalletError::provider(format!("ERC-20 decimals decode failed: {e}")))
    }

    /// Query the symbol of an ERC-20 token (e.g. "USDC", "WETH").
    ///
    /// # Errors
    ///
    /// Returns an error if the RPC call or ABI decoding fails.
    pub async fn erc20_symbol(&self, token: Address) -> Result<String, WalletError> {
        let calldata = abi::symbolCall {}.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(token)
            .with_input(Bytes::from(calldata));
        let result = self.call(tx).await?;

        abi::symbolCall::abi_decode_returns(&result)
            .map_err(|e| WalletError::provider(format!("ERC-20 symbol decode failed: {e}")))
    }
}
