//! Simple agent example with custom tools using Ollama.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example simple_agent
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::unused_async)]

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

    let mut agent = ToolCallingAgent::builder()
        .model(model)
        .tool(Box::new(Add))
        .tool(Box::new(Multiply))
        .max_steps(5)
        .build();

    let task = "What is 15 + 27? Then multiply the result by 3.";
    println!("Task: {task}\n");

    match agent.run(task).await {
        Ok(result) => println!("Result: {result}"),
        Err(e) => eprintln!("Error: {e}"),
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
