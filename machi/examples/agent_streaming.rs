//! Streaming agent example with real-time output using Ollama.
//!
//! This example demonstrates how to use the streaming API to observe
//! agent progress in real-time, including tool calls and step completions.
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

/// Multiplies two numbers.
#[machi::tool]
async fn multiply(a: i64, b: i64) -> std::result::Result<i64, ToolError> {
    Ok(a * b)
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model = OllamaClient::new().completion_model("qwen3");

    let mut agent = Agent::builder()
        .model(model)
        .tool(Box::new(Add))
        .tool(Box::new(Multiply))
        .max_steps(5)
        .build();

    let task = "What is 15 + 27? Then multiply the result by 3.";
    println!("Task: {task}\n");

    // Use async-stream based API with futures::StreamExt
    // Scope the stream to release the borrow before accessing agent.memory()
    {
        let mut stream = std::pin::pin!(agent.run_stream(task));

        // Process events using standard Stream interface
        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::StepComplete { step, action_step }) => {
                    println!("\n[Step {step} completed]");
                    if let Some(output) = &action_step.model_output {
                        let preview: String = output.chars().take(100).collect();
                        println!("  Model: {preview}...");
                    }
                    if let Some(calls) = &action_step.tool_calls {
                        for call in calls {
                            println!("  Tool: {} -> {}", call.name, call.arguments);
                        }
                    }
                    if let Some(obs) = &action_step.observations {
                        println!("  Observation: {obs}");
                    }
                }
                Ok(StreamEvent::FinalAnswer { answer }) => {
                    println!("\nFinal Answer: {answer}");
                }
                Ok(StreamEvent::Error(e)) => {
                    eprintln!("\n[Error] {e}");
                }
                Ok(StreamEvent::TextDelta(text)) => {
                    print!("{text}");
                }
                Ok(StreamEvent::ToolCallStart { name, .. }) => {
                    println!("\n[Calling tool: {name}]");
                }
                Ok(StreamEvent::ToolCallComplete { name, result, .. }) => match result {
                    Ok(r) => println!("[Tool {name} returned: {r}]"),
                    Err(e) => println!("[Tool {name} failed: {e}]"),
                },
                Ok(StreamEvent::TokenUsage(usage)) => {
                    println!("\n[Tokens: {} total]", usage.total());
                }
                Err(e) => {
                    eprintln!("\n[Stream error] {e}");
                }
            }
        }
    }

    let usage = agent.memory().total_token_usage();
    println!(
        "\nTokens: {} (in: {}, out: {})",
        usage.total(),
        usage.input_tokens,
        usage.output_tokens
    );

    Ok(())
}
