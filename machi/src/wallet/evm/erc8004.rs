//! ERC-8004 Trustless Agents integration for [`EvmWallet`](super::EvmWallet).
//!
//! Provides agent-callable tools for interacting with the three ERC-8004
//! on-chain registries: **Identity**, **Reputation**, and **Validation**.
//!
//! # Architecture
//!
//! ```text
//! EvmWallet.provider() ─→ erc8004::Erc8004<&DynProvider<Ethereum>>
//!   ├─ .identity()?   ─→ register / lookup / metadata
//!   ├─ .reputation()? ─→ feedback / summaries
//!   └─ .validation()? ─→ request / status
//! ```
//!
//! The [`erc8004`] crate handles all contract ABI bindings via alloy `sol!`
//! macros — this module provides a thin adapter layer and four Agent tools.

use std::sync::Arc;

use alloy::primitives::{Address, U256};
use async_trait::async_trait;
use erc8004::{Erc8004, Network};
use serde_json::Value;
use tracing::debug;

use super::wallet::EvmWallet;
use crate::tool::{BoxedTool, DynTool, ToolDefinition, ToolError};
use crate::wallet::WalletError;

/// Map an [`erc8004::Error`] to a [`WalletError`].
#[allow(clippy::needless_pass_by_value)]
fn map_err(e: erc8004::Error) -> WalletError {
    WalletError::erc8004(e.to_string())
}

/// Infer the [`Network`] from a chain ID, if known.
pub(crate) const fn network_from_chain_id(chain_id: u64) -> Option<Network> {
    match chain_id {
        1 => Some(Network::EthereumMainnet),
        11_155_111 => Some(Network::EthereumSepolia),
        8453 => Some(Network::BaseMainnet),
        84532 => Some(Network::BaseSepolia),
        137 => Some(Network::PolygonMainnet),
        80002 => Some(Network::PolygonAmoy),
        42161 => Some(Network::ArbitrumMainnet),
        421_614 => Some(Network::ArbitrumSepolia),
        42220 => Some(Network::CeloMainnet),
        44787 => Some(Network::CeloAlfajores),
        100 => Some(Network::GnosisMainnet),
        534_352 => Some(Network::ScrollMainnet),
        534_351 => Some(Network::ScrollSepolia),
        167_000 => Some(Network::TaikoMainnet),
        143 => Some(Network::MonadMainnet),
        10143 => Some(Network::MonadTestnet),
        56 => Some(Network::BscMainnet),
        97 => Some(Network::BscTestnet),
        _ => None,
    }
}

/// Agent tool: register an agent identity on-chain via ERC-8004.
#[derive(Debug)]
struct RegisterAgentTool {
    wallet: Arc<EvmWallet>,
}

#[async_trait]
impl DynTool for RegisterAgentTool {
    fn name(&self) -> &'static str {
        "erc8004_register"
    }

    fn description(&self) -> String {
        String::from(
            "Register a new AI agent identity on-chain via the ERC-8004 Identity \
             Registry. Returns the newly minted agent ID (ERC-721 token ID). \
             Optionally provide an agent URI pointing to a registration JSON file.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_uri": {
                        "type": "string",
                        "description": "URI to the agent registration file (e.g. ipfs://... or https://...). Optional."
                    }
                },
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let network = network_from_chain_id(self.wallet.chain_id()).ok_or_else(|| {
            ToolError::Execution(format!(
                "ERC-8004 is not deployed on chain {}",
                self.wallet.chain_id()
            ))
        })?;

        let client = Erc8004::new(self.wallet.provider()).with_network(network);

        let identity = client.identity().map_err(map_err)?;

        let agent_uri = args.get("agent_uri").and_then(Value::as_str);

        let agent_id = match agent_uri {
            Some(uri) => identity.register_with_uri(uri).await.map_err(map_err)?,
            None => identity.register().await.map_err(map_err)?,
        };

        debug!(
            agent_id = %agent_id,
            chain = %self.wallet.chain_name(),
            "ERC-8004 agent registered",
        );

        Ok(serde_json::json!({
            "agent_id": agent_id.to_string(),
            "chain": self.wallet.chain_name(),
            "chain_id": self.wallet.chain_id(),
            "agent_uri": agent_uri.unwrap_or(""),
        }))
    }
}

/// Agent tool: look up an agent's on-chain identity by ID.
#[derive(Debug)]
struct LookupAgentTool {
    wallet: Arc<EvmWallet>,
}

#[async_trait]
impl DynTool for LookupAgentTool {
    fn name(&self) -> &'static str {
        "erc8004_lookup"
    }

    fn description(&self) -> String {
        String::from(
            "Look up an ERC-8004 agent by its on-chain ID. Returns the agent's \
             registration URI, owner address, and payment wallet address.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The on-chain agent ID (uint256) to look up."
                    }
                },
                "required": ["agent_id"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let agent_id_str = args
            .get("agent_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing required field 'agent_id'"))?;

        let agent_id: U256 = agent_id_str
            .parse()
            .map_err(|_| ToolError::invalid_args("agent_id must be a valid uint256"))?;

        let network = network_from_chain_id(self.wallet.chain_id()).ok_or_else(|| {
            ToolError::Execution(format!(
                "ERC-8004 is not deployed on chain {}",
                self.wallet.chain_id()
            ))
        })?;

        let client = Erc8004::new(self.wallet.provider()).with_network(network);

        let identity = client.identity().map_err(map_err)?;

        let uri = identity.token_uri(agent_id).await.map_err(map_err)?;
        let owner = identity.owner_of(agent_id).await.map_err(map_err)?;
        let wallet_addr = identity.get_agent_wallet(agent_id).await.map_err(map_err)?;

        Ok(serde_json::json!({
            "agent_id": agent_id_str,
            "agent_uri": uri,
            "owner": owner.to_string(),
            "wallet": wallet_addr.to_string(),
            "chain": self.wallet.chain_name(),
        }))
    }
}

/// Agent tool: query an agent's reputation summary.
#[derive(Debug)]
struct ReputationTool {
    wallet: Arc<EvmWallet>,
}

#[async_trait]
impl DynTool for ReputationTool {
    fn name(&self) -> &'static str {
        "erc8004_reputation"
    }

    fn description(&self) -> String {
        String::from(
            "Query the on-chain reputation summary for an ERC-8004 agent. \
             Returns the total feedback count and aggregated score. \
             Optionally filter by client addresses and tags.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The on-chain agent ID (uint256)."
                    },
                    "client_addresses": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by specific reviewer addresses. MUST be non-empty to avoid Sybil attacks."
                    },
                    "tag1": {
                        "type": "string",
                        "description": "Primary tag filter (e.g. 'a2a.task'). Optional."
                    },
                    "tag2": {
                        "type": "string",
                        "description": "Secondary tag filter. Optional."
                    }
                },
                "required": ["agent_id", "client_addresses"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let agent_id: U256 = args
            .get("agent_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing 'agent_id'"))?
            .parse()
            .map_err(|_| ToolError::invalid_args("agent_id must be a valid uint256"))?;

        let client_addrs: Vec<Address> = args
            .get("client_addresses")
            .and_then(Value::as_array)
            .ok_or_else(|| ToolError::invalid_args("missing 'client_addresses'"))?
            .iter()
            .filter_map(Value::as_str)
            .map(str::parse::<Address>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ToolError::invalid_args(format!("invalid address: {e}")))?;

        let tag1 = args.get("tag1").and_then(Value::as_str).unwrap_or("");
        let tag2 = args.get("tag2").and_then(Value::as_str).unwrap_or("");

        let network = network_from_chain_id(self.wallet.chain_id()).ok_or_else(|| {
            ToolError::Execution(format!(
                "ERC-8004 is not deployed on chain {}",
                self.wallet.chain_id()
            ))
        })?;

        let client = Erc8004::new(self.wallet.provider()).with_network(network);

        let reputation = client.reputation().map_err(map_err)?;

        let summary = reputation
            .get_summary(agent_id, client_addrs, tag1, tag2)
            .await
            .map_err(map_err)?;

        Ok(serde_json::json!({
            "agent_id": agent_id.to_string(),
            "count": summary.count,
            "summary_value": summary.summary_value.to_string(),
            "summary_value_decimals": summary.summary_value_decimals,
            "chain": self.wallet.chain_name(),
        }))
    }
}

/// Agent tool: submit on-chain feedback for an agent.
#[derive(Debug)]
struct FeedbackTool {
    wallet: Arc<EvmWallet>,
}

#[async_trait]
impl DynTool for FeedbackTool {
    fn name(&self) -> &'static str {
        "erc8004_feedback"
    }

    fn description(&self) -> String {
        String::from(
            "Submit on-chain feedback (reputation score) for an ERC-8004 agent. \
             This is a write transaction that costs gas. The score value uses \
             fixed-point representation with configurable decimal places.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name(),
            self.description(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The on-chain agent ID (uint256) to rate."
                    },
                    "score": {
                        "type": "integer",
                        "description": "Feedback score (e.g. 0-100). Stored as int128."
                    },
                    "score_decimals": {
                        "type": "integer",
                        "description": "Decimal places for the score value. Default: 0."
                    },
                    "tag1": {
                        "type": "string",
                        "description": "Primary categorization (e.g. 'a2a.task', 'mcp.tool'). Default: empty."
                    },
                    "tag2": {
                        "type": "string",
                        "description": "Secondary categorization. Default: empty."
                    },
                    "endpoint": {
                        "type": "string",
                        "description": "The service endpoint this feedback relates to. Default: empty."
                    }
                },
                "required": ["agent_id", "score"],
                "additionalProperties": false
            }),
        )
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let agent_id: U256 = args
            .get("agent_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::invalid_args("missing 'agent_id'"))?
            .parse()
            .map_err(|_| ToolError::invalid_args("agent_id must be a valid uint256"))?;

        let score = i128::from(
            args.get("score")
                .and_then(Value::as_i64)
                .ok_or_else(|| ToolError::invalid_args("missing 'score'"))?,
        );

        let score_decimals = u8::try_from(
            args.get("score_decimals")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        )
        .map_err(|_| ToolError::invalid_args("score_decimals must be 0-255"))?;

        let tag1 = args.get("tag1").and_then(Value::as_str).unwrap_or("");
        let tag2 = args.get("tag2").and_then(Value::as_str).unwrap_or("");
        let endpoint = args.get("endpoint").and_then(Value::as_str).unwrap_or("");

        let network = network_from_chain_id(self.wallet.chain_id()).ok_or_else(|| {
            ToolError::Execution(format!(
                "ERC-8004 is not deployed on chain {}",
                self.wallet.chain_id()
            ))
        })?;

        let client = Erc8004::new(self.wallet.provider()).with_network(network);

        let reputation = client.reputation().map_err(map_err)?;

        // Empty URI and zero hash — feedback details are on-chain only.
        let empty_hash = alloy::primitives::FixedBytes::ZERO;

        reputation
            .give_feedback(
                agent_id,
                score,
                score_decimals,
                tag1,
                tag2,
                endpoint,
                "",
                empty_hash,
            )
            .await
            .map_err(map_err)?;

        debug!(
            agent_id = %agent_id,
            score = score,
            chain = %self.wallet.chain_name(),
            "ERC-8004 feedback submitted",
        );

        Ok(serde_json::json!({
            "agent_id": agent_id.to_string(),
            "score": score,
            "score_decimals": score_decimals,
            "tag1": tag1,
            "tag2": tag2,
            "chain": self.wallet.chain_name(),
            "status": "submitted",
        }))
    }
}

/// Create ERC-8004 agent tools from a shared wallet reference.
pub(crate) fn create_tools(wallet: &Arc<EvmWallet>) -> Vec<BoxedTool> {
    vec![
        Box::new(RegisterAgentTool {
            wallet: Arc::clone(wallet),
        }),
        Box::new(LookupAgentTool {
            wallet: Arc::clone(wallet),
        }),
        Box::new(ReputationTool {
            wallet: Arc::clone(wallet),
        }),
        Box::new(FeedbackTool {
            wallet: Arc::clone(wallet),
        }),
    ]
}
