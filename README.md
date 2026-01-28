# Machi

[![Crates.io](https://img.shields.io/crates/v/machi.svg)](https://crates.io/crates/machi)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

A Web3-native AI Agent Framework with embedded wallet capabilities.

## Features

- **Native Wallet Identity**: Every agent has a built-in HD wallet (powered by [kobe](https://crates.io/crates/kobe))
- **Multi-chain Support**: Ethereum, Solana, Bitcoin
- **Flexible LLM Backends**: rig, OpenAI, and more
- **Policy Control**: Fine-grained control over autonomous agent actions

## Quick Start

```rust
use machi::{Agent, AgentBuilder};
use machi::backend::rig::RigBackend;
use machi::chain::ethereum::Ethereum;

#[tokio::main]
async fn main() -> machi::Result<()> {
    // Create an agent with embedded wallet
    let agent = AgentBuilder::new()
        .backend(RigBackend::new(model))
        .chain(Ethereum::mainnet("https://eth.rpc.url"))
        .generate_wallet(12)?
        .build()?;

    println!("Agent address: {}", agent.address()?);
    Ok(())
}
```

## License

This project is licensed under either of the following licenses, at your option:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dually licensed as above, without any additional terms or conditions.
