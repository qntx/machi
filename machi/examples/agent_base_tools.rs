//! Web search agent example with built-in tools using Ollama.
//!
//! ```bash
//! ollama pull qwen3
//! cargo run --example agent_base_tools
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr)]

use machi::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model = OllamaClient::new().completion_model("qwen3");

    let mut agent = Agent::builder()
        .model(model)
        .tool(Box::new(WebSearchTool::default()))
        .tool(Box::new(VisitWebpageTool::default()))
        .max_steps(10)
        .build();

    let task = "Search for the latest Rust programming news and summarize it.";
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
