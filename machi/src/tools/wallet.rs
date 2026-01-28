//! Built-in wallet tools for agent operations.
//!
//! These tools are automatically registered when creating an agent with a wallet.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use super::executor::{Tool, ToolContext};
use super::tool::{ToolDefinition, ToolOutput};
use crate::chain::{Chain, TransactionRequest};

// ============================================================================
// GetAddress Tool
// ============================================================================

/// Tool to get the agent's wallet address on the current chain.
pub struct GetAddress;

impl GetAddress {
    /// Get the tool definition (for display without Chain bound).
    pub fn def() -> ToolDefinition {
        ToolDefinition::new(
            "get_address",
            "Get the agent's wallet address on the current blockchain",
        )
        .optional_param("index", "number", "Address derivation index (default: 0)")
    }
}

impl<C: Chain + Send + Sync> Tool<C> for GetAddress {
    fn definition(&self) -> ToolDefinition {
        Self::def()
    }

    fn call<'a>(
        &'a self,
        ctx: &'a ToolContext<'a, C>,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let index = args
                .get("index")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .unwrap_or(ctx.wallet.default_index());

            match ctx.chain.derive_address(ctx.wallet.inner(), index) {
                Ok(addr) => ToolOutput::ok(serde_json::json!({
                    "address": addr.as_ref(),
                    "chain": ctx.chain.name(),
                    "index": index
                })),
                Err(e) => ToolOutput::err(format!("Failed to derive address: {e}")),
            }
        })
    }
}

// ============================================================================
// GetBalance Tool
// ============================================================================

/// Tool to get the balance of an address.
pub struct GetBalance;

impl GetBalance {
    /// Get the tool definition.
    pub fn def() -> ToolDefinition {
        ToolDefinition::new("get_balance", "Get the native token balance of an address")
            .optional_param(
                "address",
                "string",
                "Address to check (default: agent's address)",
            )
    }
}

impl<C: Chain + Send + Sync> Tool<C> for GetBalance {
    fn definition(&self) -> ToolDefinition {
        Self::def()
    }

    fn call<'a>(
        &'a self,
        ctx: &'a ToolContext<'a, C>,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            // Get address - use provided or derive agent's address
            let address = match args.get("address").and_then(|v| v.as_str()) {
                Some(addr) => addr.to_string(),
                None => {
                    match ctx
                        .chain
                        .derive_address(ctx.wallet.inner(), ctx.wallet.default_index())
                    {
                        Ok(addr) => addr.as_ref().to_string(),
                        Err(e) => return ToolOutput::err(format!("Failed to derive address: {e}")),
                    }
                }
            };

            match ctx.chain.balance(&address).await {
                Ok(balance) => ToolOutput::ok(serde_json::json!({
                    "address": address,
                    "balance": balance.to_string(),
                    "chain": ctx.chain.name()
                })),
                Err(e) => ToolOutput::err(format!("Failed to get balance: {e}")),
            }
        })
    }
}

// ============================================================================
// SendTransaction Tool
// ============================================================================

/// Tool to send a transaction.
pub struct SendTransaction;

impl SendTransaction {
    /// Get the tool definition.
    pub fn def() -> ToolDefinition {
        ToolDefinition::new(
            "send_transaction",
            "Send native tokens to a recipient address",
        )
        .param("to", "string", "Recipient address")
        .param("value", "string", "Amount in smallest unit (e.g., wei)")
        .optional_param("data", "string", "Hex-encoded calldata for contract calls")
    }
}

impl<C: Chain + Send + Sync> Tool<C> for SendTransaction {
    fn definition(&self) -> ToolDefinition {
        Self::def()
    }

    fn call<'a>(
        &'a self,
        ctx: &'a ToolContext<'a, C>,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let to = match args.get("to").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => return ToolOutput::err("Missing required argument: to"),
            };

            let value_str = match args.get("value").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => return ToolOutput::err("Missing required argument: value"),
            };

            let value: u128 = match value_str.parse() {
                Ok(v) => v,
                Err(_) => return ToolOutput::err("Invalid value: must be a valid number"),
            };

            let data = args.get("data").and_then(|v| v.as_str()).and_then(|s| {
                let s = s.strip_prefix("0x").unwrap_or(s);
                hex::decode(s).ok()
            });

            let tx = TransactionRequest { to, value, data };

            match ctx
                .chain
                .send_transaction(ctx.wallet.inner(), ctx.wallet.default_index(), tx)
                .await
            {
                Ok(tx_hash) => ToolOutput::ok(serde_json::json!({
                    "success": true,
                    "tx_hash": tx_hash.to_string(),
                    "chain": ctx.chain.name()
                })),
                Err(e) => ToolOutput::err(format!("Transaction failed: {e}")),
            }
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get all built-in wallet tool definitions.
pub fn builtin_tool_definitions() -> Vec<ToolDefinition> {
    vec![GetAddress::def(), GetBalance::def(), SendTransaction::def()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_address_definition() {
        let def = GetAddress::def();
        assert_eq!(def.name, "get_address");
    }

    #[test]
    fn test_get_balance_definition() {
        let def = GetBalance::def();
        assert_eq!(def.name, "get_balance");
    }

    #[test]
    fn test_send_transaction_definition() {
        let def = SendTransaction::def();
        assert_eq!(def.name, "send_transaction");
        assert_eq!(def.parameters.len(), 3);
    }
}
