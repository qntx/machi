//! Web search agent example demonstrating built-in tools.
//!
//! This example shows how to create an agent with web search and webpage visiting tools.
//!
//! # Running
//!
//! Set your OpenAI API key:
//! ```bash
//! export OPENAI_API_KEY=your_key_here
//! cargo run --example web_search_agent
//! ```

use machi::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    // Check for API key
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("Please set OPENAI_API_KEY environment variable");
        eprintln!("Example: export OPENAI_API_KEY=sk-...");
        std::process::exit(1);
    }

    // Create model
    let model = OpenAIModel::new("gpt-4o-mini");

    // Create agent with web tools
    let mut agent = ToolCallingAgent::builder()
        .model(model)
        .tool(Box::new(WebSearchTool::new()))
        .tool(Box::new(VisitWebpageTool::new()))
        .max_steps(10)
        .build();

    // Run the agent with a web search task
    let task =
        "Search for the latest news about Rust programming language and summarize the top result.";
    println!("Running agent with task: '{}'", task);
    println!("---");

    match agent.run(task).await {
        Ok(result) => {
            println!("---");
            println!("Agent completed successfully!");
            println!("Result: {}", result);
        }
        Err(e) => {
            eprintln!("Agent error: {}", e);
        }
    }

    // Print token usage
    let usage = agent.memory().total_token_usage();
    println!(
        "Total tokens used: {} (input: {}, output: {})",
        usage.total(),
        usage.input_tokens,
        usage.output_tokens
    );

    Ok(())
}
