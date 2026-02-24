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
    let mut tools: Vec<BoxedTool> = vec![
        // Identity & state
        Box::new(GetWalletInfoTool(Arc::clone(wallet))),
        Box::new(GetBalanceTool(Arc::clone(wallet))),
        Box::new(GetNonceTool(Arc::clone(wallet))),
        // Chain queries
        Box::new(GetBlockNumberTool(Arc::clone(wallet))),
        Box::new(GetGasPriceTool(Arc::clone(wallet))),
        Box::new(IsContractTool(Arc::clone(wallet))),
        // Transaction inspection
        Box::new(GetTransactionReceiptTool(Arc::clone(wallet))),
        Box::new(GetTransactionTool(Arc::clone(wallet))),
        // Signing
        Box::new(SignMessageTool(Arc::clone(wallet))),
        // Transfers
        Box::new(TransferTool(Arc::clone(wallet))),
        // ERC-20 token operations
        Box::new(Erc20BalanceTool(Arc::clone(wallet))),
        Box::new(Erc20TransferTool(Arc::clone(wallet))),
    ];

    // ENS tools are only useful on Ethereum mainnet.
    if wallet.chain_id() == 1 {
        tools.push(Box::new(ResolveEnsTool(Arc::clone(wallet))));
        tools.push(Box::new(ReverseEnsTool(Arc::clone(wallet))));
    }

    // x402 payment-enabled HTTP fetch tool.
    #[cfg(feature = "x402")]
    tools.extend(super::x402::create_tools(wallet));

    tools
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

/// Query the latest block number on the connected chain.
#[derive(Debug)]
struct GetBlockNumberTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for GetBlockNumberTool {
    fn name(&self) -> &str {
        "get_block_number"
    }

    fn description(&self) -> String {
        format!(
            "Get the latest block number on {} (chain ID: {}).",
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
        let block_number = self.0.block_number().await?;
        Ok(serde_json::json!({
            "block_number": block_number,
            "chain": self.0.chain_name(),
        }))
    }
}

/// Query gas price and EIP-1559 fee estimates.
#[derive(Debug)]
struct GetGasPriceTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for GetGasPriceTool {
    fn name(&self) -> &str {
        "get_gas_price"
    }

    fn description(&self) -> String {
        format!(
            "Get current gas price and EIP-1559 fee estimates (in wei) on {}.",
            self.0.chain_name(),
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
        let gas_price = self.0.gas_price().await?;

        // Try EIP-1559 fees; fall back gracefully on legacy chains.
        let eip1559 = self.0.estimate_eip1559_fees().await.ok();

        let mut result = serde_json::json!({
            "gas_price_wei": gas_price.to_string(),
            "chain": self.0.chain_name(),
        });

        if let Some((max_fee, max_priority_fee)) = eip1559 {
            result["max_fee_per_gas_wei"] = Value::String(max_fee.to_string());
            result["max_priority_fee_per_gas_wei"] = Value::String(max_priority_fee.to_string());
        }

        Ok(result)
    }
}

/// Query the transaction count (nonce) for the agent's address.
#[derive(Debug)]
struct GetNonceTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for GetNonceTool {
    fn name(&self) -> &str {
        "get_nonce"
    }

    fn description(&self) -> String {
        String::from(
            "Get the agent wallet's current transaction count (nonce). \
             Useful for determining the next transaction sequence number.",
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
        let nonce = self.0.nonce().await?;
        Ok(serde_json::json!({
            "nonce": nonce,
            "address": self.0.address(),
        }))
    }
}

/// Look up a transaction receipt by hash.
#[derive(Debug)]
struct GetTransactionReceiptTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for GetTransactionReceiptTool {
    fn name(&self) -> &str {
        "get_transaction_receipt"
    }

    fn description(&self) -> String {
        String::from(
            "Get the receipt for a mined transaction by its hash. \
             Returns status, gas used, block number, and logs count. \
             Returns null fields if the transaction is still pending.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "0x-prefixed transaction hash (66 characters)."
                    }
                },
                "required": ["hash"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let hash_str = args
            .get("hash")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'hash'"))?;

        let hash: alloy::primitives::B256 = hash_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid tx hash: {e}")))?;

        let receipt = self.0.transaction_receipt(hash).await?;

        match receipt {
            Some(r) => Ok(serde_json::json!({
                "found": true,
                "status": r.status(),
                "block_number": r.block_number,
                "gas_used": r.gas_used.to_string(),
                "effective_gas_price": r.effective_gas_price.to_string(),
                "logs_count": r.inner.logs().len(),
                "from": format!("{:#x}", r.from),
                "to": r.to.map(|a| format!("{a:#x}")),
                "contract_address": r.contract_address.map(|a| format!("{a:#x}")),
            })),
            None => Ok(serde_json::json!({
                "found": false,
                "message": "Transaction not yet mined or not found.",
            })),
        }
    }
}

/// Look up a transaction by hash.
#[derive(Debug)]
struct GetTransactionTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for GetTransactionTool {
    fn name(&self) -> &str {
        "get_transaction"
    }

    fn description(&self) -> String {
        String::from(
            "Get transaction details by hash. Returns sender, recipient, \
             value, gas, nonce, and input data length.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "0x-prefixed transaction hash (66 characters)."
                    }
                },
                "required": ["hash"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let hash_str = args
            .get("hash")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'hash'"))?;

        let hash: alloy::primitives::B256 = hash_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid tx hash: {e}")))?;

        let tx = self.0.transaction_by_hash(hash).await?;

        match tx {
            Some(t) => {
                // Serialize the full RPC transaction to avoid hardcoded field paths
                // across different alloy transaction envelope variants.
                let tx_json = serde_json::to_value(&t)
                    .map_err(|e| ToolError::execution(format!("serialize tx: {e}")))?;
                Ok(serde_json::json!({
                    "found": true,
                    "block_number": t.block_number,
                    "effective_gas_price": t.effective_gas_price,
                    "transaction": tx_json,
                }))
            }
            None => Ok(serde_json::json!({
                "found": false,
                "message": "Transaction not found.",
            })),
        }
    }
}

/// Check whether an address is a smart contract or an EOA.
#[derive(Debug)]
struct IsContractTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for IsContractTool {
    fn name(&self) -> &str {
        "is_contract"
    }

    fn description(&self) -> String {
        String::from(
            "Check whether an Ethereum address is a smart contract or an \
             externally owned account (EOA). Returns true if bytecode exists.",
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
                        "description": "0x-prefixed Ethereum address to check."
                    }
                },
                "required": ["address"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let addr_str = args
            .get("address")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'address'"))?;

        let address: alloy::primitives::Address = addr_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid address: {e}")))?;

        let code = self.0.code_at(address).await?;
        let is_contract = !code.is_empty();

        Ok(serde_json::json!({
            "address": addr_str,
            "is_contract": is_contract,
            "code_size_bytes": code.len(),
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

/// Resolve an ENS name to an Ethereum address (mainnet only).
#[derive(Debug)]
struct ResolveEnsTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for ResolveEnsTool {
    fn name(&self) -> &str {
        "resolve_ens"
    }

    fn description(&self) -> String {
        String::from(
            "Resolve an ENS name (e.g. 'vitalik.eth') to its Ethereum address. \
             Only available on Ethereum mainnet (chain ID 1).",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "ENS name to resolve (e.g. 'vitalik.eth')."
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'name'"))?;

        let address = self.0.resolve_ens(name).await?;

        address.map_or_else(
            || {
                Ok(serde_json::json!({
                    "resolved": false,
                    "name": name,
                    "message": "No address record found for this ENS name.",
                }))
            },
            |addr| {
                Ok(serde_json::json!({
                    "resolved": true,
                    "name": name,
                    "address": format!("{addr:#x}"),
                }))
            },
        )
    }
}

/// Reverse-resolve an Ethereum address to its ENS name (mainnet only).
#[derive(Debug)]
struct ReverseEnsTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for ReverseEnsTool {
    fn name(&self) -> &str {
        "reverse_ens"
    }

    fn description(&self) -> String {
        String::from(
            "Look up the ENS name associated with an Ethereum address \
             (reverse resolution). Only available on Ethereum mainnet (chain ID 1).",
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
                        "description": "0x-prefixed Ethereum address to look up."
                    }
                },
                "required": ["address"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let addr_str = args
            .get("address")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'address'"))?;

        let address: alloy::primitives::Address = addr_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid address: {e}")))?;

        let name = self.0.reverse_ens(address).await?;

        name.map_or_else(
            || {
                Ok(serde_json::json!({
                    "resolved": false,
                    "address": addr_str,
                    "message": "No reverse ENS record found for this address.",
                }))
            },
            |n| {
                Ok(serde_json::json!({
                    "resolved": true,
                    "address": addr_str,
                    "name": n,
                }))
            },
        )
    }
}

/// Query an ERC-20 token balance, including symbol and decimals.
#[derive(Debug)]
struct Erc20BalanceTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for Erc20BalanceTool {
    fn name(&self) -> &str {
        "erc20_balance"
    }

    fn description(&self) -> String {
        format!(
            "Get the ERC-20 token balance for any address on {}. \
             Also returns the token symbol and decimals. \
             Omit 'owner' to check the agent's own balance.",
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
                    "token": {
                        "type": "string",
                        "description": "ERC-20 token contract address (0x-prefixed)."
                    },
                    "owner": {
                        "type": "string",
                        "description": "Address to query. Omit for the agent's own balance."
                    }
                },
                "required": ["token"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let token_str = args
            .get("token")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'token'"))?;

        let token: alloy::primitives::Address = token_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid token address: {e}")))?;

        let owner = if let Some(s) = args.get("owner").and_then(Value::as_str) {
            s.parse()
                .map_err(|e| ToolError::invalid_args(format!("invalid owner address: {e}")))?
        } else {
            self.0.address_typed()
        };

        let balance = self.0.erc20_balance(token, owner).await?;

        // Best-effort metadata — some non-standard tokens may not implement these.
        let symbol = self.0.erc20_symbol(token).await.ok();
        let decimals = self.0.erc20_decimals(token).await.ok();

        Ok(serde_json::json!({
            "token": token_str,
            "owner": format!("{owner:#x}"),
            "balance_raw": balance.to_string(),
            "symbol": symbol,
            "decimals": decimals,
        }))
    }
}

/// Transfer ERC-20 tokens to a recipient.
#[derive(Debug)]
struct Erc20TransferTool(Arc<EvmWallet>);

#[async_trait]
impl DynTool for Erc20TransferTool {
    fn name(&self) -> &str {
        "erc20_transfer"
    }

    fn description(&self) -> String {
        format!(
            "Transfer ERC-20 tokens to a recipient on {}. \
             Amount is in raw token units (not adjusted for decimals). \
             Returns the transaction hash.",
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
                    "token": {
                        "type": "string",
                        "description": "ERC-20 token contract address (0x-prefixed)."
                    },
                    "to": {
                        "type": "string",
                        "description": "Recipient address (0x-prefixed)."
                    },
                    "amount": {
                        "type": "string",
                        "description": "Amount in raw token units (e.g. \"1000000\" for 1 USDC with 6 decimals)."
                    }
                },
                "required": ["token", "to", "amount"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let token_str = args
            .get("token")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'token'"))?;
        let to_str = args
            .get("to")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'to'"))?;
        let amount_str = args
            .get("amount")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'amount'"))?;

        let token: alloy::primitives::Address = token_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid token address: {e}")))?;
        let to: alloy::primitives::Address = to_str
            .parse()
            .map_err(|e| ToolError::invalid_args(format!("invalid recipient address: {e}")))?;
        let amount = alloy::primitives::U256::from_str_radix(amount_str, 10)
            .map_err(|e| ToolError::invalid_args(format!("invalid amount: {e}")))?;

        let tx_hash = self.0.erc20_transfer(token, to, amount).await?;

        Ok(serde_json::json!({
            "tx_hash": tx_hash,
            "token": token_str,
            "from": self.0.address(),
            "to": to_str,
            "amount_raw": amount_str,
            "chain": self.0.chain_name(),
        }))
    }
}
