//! Streaming agent example with real-time output.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_streaming
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::unused_async)]

use futures::StreamExt;
use machi::prelude::*;

/// Adds two numbers.
#[machi::tool]
async fn add(a: i64, b: i64) -> std::result::Result<i64, ToolError> {
    Ok(a + b)
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model = OllamaClient::new().completion_model("qwen3");
    let mut agent = Agent::builder()
        .model(model)
        .tool(Box::new(Add))
        .max_steps(3)
        .build();

    let mut stream = std::pin::pin!(agent.stream("What is 15 + 27?"));

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::TextDelta(text)) => print!("{text}"),
            Ok(StreamEvent::ToolCallStart { name, .. }) => println!("\n[Calling: {name}]"),
            Ok(StreamEvent::ToolCallComplete { name, result, .. }) => {
                println!("[{name} returned: {}]", result.unwrap_or_else(|e| e));
            }
            Ok(StreamEvent::FinalAnswer { answer }) => println!("\nAnswer: {answer}"),
            Ok(StreamEvent::Error(e)) => eprintln!("\n[Error] {e}"),
            Ok(_) => {}
            Err(e) => eprintln!("\n[Stream error] {e}"),
        }
    }

    Ok(())
}
