//! Token-level streaming example with typing effect.
//!
//! Each token from the model is yielded immediately, enabling real-time
//! output similar to ChatGPT's typing effect.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_stream
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::unused_async)]

use futures::StreamExt;
use machi::prelude::*;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model = OllamaClient::new().completion_model("qwen3");
    let mut agent = Agent::builder()
        .model(model)
        .tool(Box::new(WebSearchTool::default()))
        .tool(Box::new(VisitWebpageTool::default()))
        .max_steps(5)
        .build();

    let mut stream = std::pin::pin!(agent.stream("What's the weather in Tokyo?"));

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::TextDelta(text)) => {
                print!("{text}");
                stdout().flush()?;
            }
            Ok(StreamEvent::ToolCallStart { name, .. }) => println!("\n[Calling: {name}]"),
            Ok(StreamEvent::ToolCallComplete { name, result, .. }) => {
                println!("[{name}: {}]", result.unwrap_or_else(|e| e));
            }
            Ok(StreamEvent::StepComplete { step, .. }) => println!("\n[Step {step} done]\n"),
            Ok(StreamEvent::FinalAnswer { answer }) => println!("\nAnswer: {answer}"),
            Ok(StreamEvent::Error(e)) => eprintln!("\n[Error] {e}"),
            Ok(_) => {}
            Err(e) => eprintln!("\n[Stream error] {e}"),
        }
    }

    Ok(())
}
