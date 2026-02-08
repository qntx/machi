//! ENS (Ethereum Name Service) resolution for [`EvmWallet`].
//!
//! Provides forward resolution (name → address) and reverse resolution
//! (address → name) by interacting with the on-chain ENS registry and
//! resolver contracts via `eth_call`.

use alloy::network::TransactionBuilder;
use alloy::primitives::{Address, B256, Bytes};
use alloy::rpc::types::TransactionRequest;
use alloy::sol_types::SolCall;

use super::wallet::EvmWallet;
use crate::wallet::WalletError;

/// ENS registry contract address on Ethereum mainnet.
const ENS_REGISTRY: Address =
    alloy::primitives::address!("00000000000C2E074eC69A0dFb2997BA6C7d2e1e");

/// ABI fragments for ENS registry and resolver calls.
mod abi {
    alloy::sol! {
        function resolver(bytes32 node) external view returns (address);
        function addr(bytes32 node) external view returns (address);
        function name(bytes32 node) external view returns (string);
    }
}

/// Compute the ENS namehash for a domain name (EIP-137).
fn namehash(name: &str) -> B256 {
    let mut node = B256::ZERO;
    if name.is_empty() {
        return node;
    }
    for label in name.rsplit('.') {
        let label_hash = alloy::primitives::keccak256(label.as_bytes());
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(node.as_slice());
        buf[32..].copy_from_slice(label_hash.as_slice());
        node = alloy::primitives::keccak256(buf);
    }
    node
}

impl EvmWallet {
    /// Resolve an ENS name to an Ethereum [`Address`].
    ///
    /// Only available on Ethereum mainnet (chain ID 1). Returns `None`
    /// if the name has no resolver or no address record.
    ///
    /// # Errors
    ///
    /// Returns an error if the chain is not mainnet or RPC calls fail.
    pub async fn resolve_ens(&self, name: &str) -> Result<Option<Address>, WalletError> {
        if self.chain_id() != 1 {
            return Err(WalletError::provider(
                "ENS resolution is only available on Ethereum mainnet".to_owned(),
            ));
        }

        let node = namehash(name);

        // Step 1: look up resolver in the ENS registry.
        let calldata = abi::resolverCall { node }.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(ENS_REGISTRY)
            .with_input(Bytes::from(calldata));
        let result = self.call(tx).await?;

        let resolver = abi::resolverCall::abi_decode_returns(&result)
            .map_err(|e| WalletError::provider(format!("ENS resolver decode failed: {e}")))?;

        if resolver.is_zero() {
            return Ok(None);
        }

        // Step 2: ask the resolver for the address.
        let calldata = abi::addrCall { node }.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(resolver)
            .with_input(Bytes::from(calldata));
        let result = self.call(tx).await?;

        let addr = abi::addrCall::abi_decode_returns(&result)
            .map_err(|e| WalletError::provider(format!("ENS addr decode failed: {e}")))?;

        if addr.is_zero() {
            Ok(None)
        } else {
            Ok(Some(addr))
        }
    }

    /// Reverse-resolve an Ethereum [`Address`] to its ENS name.
    ///
    /// Only available on Ethereum mainnet (chain ID 1). Returns `None`
    /// if the address has no reverse record.
    ///
    /// # Errors
    ///
    /// Returns an error if the chain is not mainnet or RPC calls fail.
    pub async fn reverse_ens(&self, address: Address) -> Result<Option<String>, WalletError> {
        if self.chain_id() != 1 {
            return Err(WalletError::provider(
                "ENS resolution is only available on Ethereum mainnet".to_owned(),
            ));
        }

        let addr_hex = alloy::primitives::hex::encode(address.as_slice());
        let reverse_name = format!("{addr_hex}.addr.reverse");
        let node = namehash(&reverse_name);

        // Step 1: look up resolver in the ENS registry.
        let calldata = abi::resolverCall { node }.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(ENS_REGISTRY)
            .with_input(Bytes::from(calldata));
        let result = self.call(tx).await?;

        let resolver = abi::resolverCall::abi_decode_returns(&result)
            .map_err(|e| WalletError::provider(format!("ENS resolver decode failed: {e}")))?;

        if resolver.is_zero() {
            return Ok(None);
        }

        // Step 2: ask the resolver for the name.
        let calldata = abi::nameCall { node }.abi_encode();
        let tx = TransactionRequest::default()
            .with_to(resolver)
            .with_input(Bytes::from(calldata));
        let result = self.call(tx).await?;

        let name = abi::nameCall::abi_decode_returns(&result)
            .map_err(|e| WalletError::provider(format!("ENS name decode failed: {e}")))?;

        if name.is_empty() {
            Ok(None)
        } else {
            Ok(Some(name))
        }
    }
}
