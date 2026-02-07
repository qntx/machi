//! Agent with vision (multimodal input) example using OpenAI.
//!
//! Demonstrates passing an image to an agent via `ContentPart`
//! for visual question answering.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_vision
//! ```

#![allow(clippy::print_stdout)]

use machi::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("vision")
        .instructions("You describe images concisely.")
        .model("gpt-4o-mini")
        .provider(provider);

    // Multimodal input: text + image URL.
    let input = vec![
        ContentPart::text("What is in this image?"),
        ContentPart::image_url("https://picsum.photos/id/237/400/300"),
    ];

    let result = agent.run(input, RunConfig::default()).await?;
    println!("{}", result.output);

    Ok(())
}
