//! Vision example with image input using model directly.
//!
//! ```bash
//! ollama pull qwen3-vl
//! cargo run --example agent_vision
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr)]

use machi::prelude::*;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model = OllamaClient::new().completion_model("qwen3-vl");

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/data")
        .join("camponotus_flavomarginatus_ant.jpg");
    let image = AgentImage::load_from_path(&path).await?;

    let content = vec![
        MessageContent::text("What is in this image? Describe it in detail."),
        MessageContent::from_agent_image(&image).expect("image should have data"),
    ];
    let messages = vec![ChatMessage::with_contents(MessageRole::User, content)];

    let response = model.generate(messages, GenerateOptions::default()).await?;
    println!("{}", response.message.text_content().unwrap_or_default());

    Ok(())
}
