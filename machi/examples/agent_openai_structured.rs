//! Agent with structured output example using OpenAI.
//!
//! Demonstrates `Agent::output_type::<T>()` for type-safe structured
//! output that is automatically parsed into a Rust struct.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_structured --features schema
//! ```

#![allow(clippy::print_stdout)]

use machi::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize, JsonSchema)]
struct Country {
    name: String,
    capital: String,
    population: u64,
    languages: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("geo")
        .instructions("You provide country facts as structured JSON.")
        .model("gpt-4o-mini")
        .provider(provider)
        .output_type::<Country>();

    let result = agent
        .run("Tell me about Japan.", RunConfig::default())
        .await?;

    let country: Country = result.parse().expect("valid Country JSON");
    println!("Name:       {}", country.name);
    println!("Capital:    {}", country.capital);
    println!("Population: {}", country.population);
    println!("Languages:  {}", country.languages.join(", "));

    Ok(())
}
