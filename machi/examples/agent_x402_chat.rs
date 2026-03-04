//! x402 payment-gated LLM chat example.
//!
//! Demonstrates using an EVM wallet to pay for LLM API calls via the
//! x402 protocol. The wallet signs ERC-3009 payment authorizations
//! transparently — no gas required.
//!
//! ```bash
//! EVM_PRIVATE_KEY="0x..." cargo run --example agent_x402_chat
//! EVM_PRIVATE_KEY="0x..." EVM_CHAIN_ID=8453 cargo run --example agent_x402_chat
//! ```

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use machi::prelude::*;
use machi::wallet::EvmWallet;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,machi=debug")
        .init();

    let private_key =
        std::env::var("EVM_PRIVATE_KEY").expect("EVM_PRIVATE_KEY environment variable required");

    // Signer-only wallet — no RPC needed for x402 payment signing.
    // Defaults to Monad (chain 143). Set EVM_CHAIN_ID to override.
    let mut wallet = EvmWallet::from_private_key(&private_key)?;
    if let Ok(id) = std::env::var("EVM_CHAIN_ID").and_then(|s| s.parse::<u64>().map_err(|_| std::env::VarError::NotPresent)) {
        wallet = wallet.with_chain(machi::wallet::EvmChain::from_id(id));
    }

    // OpenAI::from_wallet() creates an x402-enabled LLM client that
    // transparently signs payments when the gateway returns HTTP 402.
    let provider: SharedChatProvider = Arc::new(OpenAI::from_wallet(&wallet)?);

    let agent = Agent::new("x402-agent")
        .instructions("You are a helpful assistant. Keep answers concise.")
        .model("openai/gpt-4o-mini")
        .provider(provider)
        .wallet(wallet);

    let result = agent
        .run("What is the x402 payment protocol?", RunConfig::default())
        .await?;

    println!("{}", result.output);

    Ok(())
}
