//! Agent-callable tool implementations for [`EvmWallet`](super::EvmWallet).
//!
//! Each tool wraps a shared `Arc<EvmWallet>` and exposes a specific wallet
//! capability through the [`DynTool`] interface.

#![allow(clippy::unnecessary_literal_bound)]

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::wallet::EvmWallet;
use crate::tool::{BoxedTool, DynTool, ToolDefinition, ToolError};

/// Create all wallet tools from a shared wallet reference.
pub fn create_tools(wallet: &Arc<EvmWallet>) -> Vec<BoxedTool> {
    vec![
        Box::new(GetWalletInfoTool(Arc::clone(wallet))),
        Box::new(GetBalanceTool(Arc::clone(wallet))),
        Box::new(SignMessageTool(Arc::clone(wallet))),
        Box::new(TransferTool(Arc::clone(wallet))),
    ]
}

/// Returns the wallet's identity: address, chain, derivation path.
#[derive(Debug)]
struct GetWalletInfoTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for GetWalletInfoTool {
    fn name(&self) -> &str {
        "get_wallet_info"
    }

    fn description(&self) -> String {
        format!(
            "Get the agent's wallet identity on {} (chain ID: {}). \
             Returns address, chain, and derivation path.",
            self.0.chain_name(),
            self.0.chain_id(),
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, _args: Value) -> Result<Value, ToolError> {
        Ok(serde_json::json!({
            "address": self.0.address(),
            "chain": self.0.chain_name(),
            "chain_id": self.0.chain_id(),
            "derivation_path": self.0.derivation_path(),
            "public_key": self.0.public_key(),
        }))
    }
}

/// Query native token balance for any address or the agent's own.
#[derive(Debug)]
struct GetBalanceTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for GetBalanceTool {
    fn name(&self) -> &str {
        "get_balance"
    }

    fn description(&self) -> String {
        format!(
            "Get the native token balance (in wei) on {}. \
             Omit address to check the agent's own balance.",
            self.0.chain_name(),
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "0x-prefixed Ethereum address. Omit for the agent's own balance."
                    }
                },
                "required": [],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let balance = if let Some(addr_str) = args.get("address").and_then(Value::as_str) {
            let address: alloy::primitives::Address = addr_str
                .parse()
                .map_err(|e| ToolError::invalid_args(format!("invalid address: {e}")))?;
            self.0.balance_of(address).await?
        } else {
            self.0.balance().await?
        };

        Ok(serde_json::json!({
            "balance_wei": balance.to_string(),
            "address": args.get("address").and_then(Value::as_str).unwrap_or_else(|| self.0.address()),
            "chain": self.0.chain_name(),
        }))
    }
}

/// Sign an arbitrary message using EIP-191 `personal_sign`.
#[derive(Debug)]
struct SignMessageTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for SignMessageTool {
    fn name(&self) -> &str {
        "sign_message"
    }

    fn description(&self) -> String {
        String::from(
            "Sign an arbitrary message using EIP-191 personal_sign. \
             Returns the 0x-prefixed hex signature.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to sign."
                    }
                },
                "required": ["message"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let message = args
            .get("message")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'message'"))?;

        let signature = self.0.sign_message(message.as_bytes()).await?;

        Ok(serde_json::json!({
            "signature": signature,
            "signer": self.0.address(),
        }))
    }
}

/// Transfer native token to an address.
#[derive(Debug)]
struct TransferTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for TransferTool {
    fn name(&self) -> &str {
        "transfer"
    }

    fn description(&self) -> String {
        format!(
            "Transfer native token on {} to a recipient. \
             Amount is in wei (1 ETH = 10^18 wei). Returns the transaction hash.",
            self.0.chain_name(),
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient address (0x-prefixed hex)."
                    },
                    "amount": {
                        "type": "string",
                        "description": "Amount in wei (e.g. \"1000000000000000000\" for 1 ETH)."
                    }
                },
                "required": ["to", "amount"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let to_str = args
            .get("to")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'to'"))?;
        let amount_str = args
            .get("amount")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'amount'"))?;

        let to: alloy::primitives::Address = to_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid address: {e}")))?;
        let amount = alloy::primitives::U256::from_str_radix(amount_str, 10)
            .map_err(|e| ToolError::invalid_args(format!("invalid amount: {e}")))?;

        let tx_hash = self.0.transfer(to, amount).await?;

        Ok(serde_json::json!({
            "tx_hash": tx_hash,
            "from": self.0.address(),
            "to": to_str,
            "amount_wei": amount_str,
            "chain": self.0.chain_name(),
        }))
    }
}
