//! Simple agent example demonstrating basic usage of machi.
//!
//! This example shows how to create a tool-calling agent with a custom tool.
//!
//! # Running
//!
//! Set your OpenAI API key:
//! ```bash
//! export OPENAI_API_KEY=your_key_here
//! cargo run --example simple_agent
//! ```

use async_trait::async_trait;
use machi::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A simple calculator tool that adds two numbers.
#[derive(Debug, Clone, Copy, Default)]
struct AddTool;

/// Arguments for the add tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AddArgs {
    /// First number to add.
    a: i64,
    /// Second number to add.
    b: i64,
}

#[async_trait]
impl Tool for AddTool {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Adds two numbers together and returns the result.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "First number to add"
                },
                "b": {
                    "type": "integer",
                    "description": "Second number to add"
                }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: Self::Args) -> std::result::Result<Self::Output, Self::Error> {
        Ok(args.a + args.b)
    }
}

/// A tool that multiplies two numbers.
#[derive(Debug, Clone, Copy, Default)]
struct MultiplyTool;

/// Arguments for the multiply tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MultiplyArgs {
    /// First number to multiply.
    a: i64,
    /// Second number to multiply.
    b: i64,
}

#[async_trait]
impl Tool for MultiplyTool {
    const NAME: &'static str = "multiply";
    type Args = MultiplyArgs;
    type Output = i64;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Multiplies two numbers together and returns the result.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "First number to multiply"
                },
                "b": {
                    "type": "integer",
                    "description": "Second number to multiply"
                }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: Self::Args) -> std::result::Result<Self::Output, Self::Error> {
        Ok(args.a * args.b)
    }
}

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

    // Create agent with tools
    let mut agent = ToolCallingAgent::builder()
        .model(model)
        .tool(Box::new(AddTool))
        .tool(Box::new(MultiplyTool))
        .max_steps(10)
        .build();

    // Run the agent
    println!("Running agent with task: 'What is 15 + 27? Then multiply the result by 3.'");
    println!("---");

    match agent
        .run("What is 15 + 27? Then multiply the result by 3.")
        .await
    {
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
