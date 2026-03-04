//! x402 payment-gated streaming chat example.
//!
//! Demonstrates streaming LLM responses paid via x402 protocol.
//! The wallet signs ERC-3009 authorizations transparently.
//!
//! ```bash
//! EVM_PRIVATE_KEY="0x..." cargo run --example x402_streaming
//! EVM_PRIVATE_KEY="0x..." EVM_CHAIN_ID=8453 cargo run --example x402_streaming
//! ```

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use futures::StreamExt;
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
    let wallet = EvmWallet::from_private_key(&private_key)?;
    let provider: SharedChatProvider = Arc::new(OpenAI::from_wallet(&wallet)?);

    let request = ChatRequest::new("openai/gpt-4o-mini")
        .system("You are a helpful assistant.")
        .user("Explain the x402 payment protocol.");

    let mut stream = provider.chat_stream(&request).await?;

    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamChunk::Text(text) => print!("{text}"),
            StreamChunk::Done { .. } => println!(),
            _ => {}
        }
    }

    Ok(())
}
