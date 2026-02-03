//! Basic example demonstrating how to use machi with a tool-calling agent.
//!
//! Run with: `cargo run --example basic_agent`

use async_trait::async_trait;
use machi::tool::{Tool, ToolError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Arguments for the calculator tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculatorArgs {
    /// First operand.
    pub a: f64,
    /// Second operand.
    pub b: f64,
    /// Operation to perform: add, subtract, multiply, divide.
    pub operation: String,
}

/// A simple calculator tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    const NAME: &'static str = "calculator";
    type Args = CalculatorArgs;
    type Output = f64;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Perform basic arithmetic operations (add, subtract, multiply, divide).".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand"
                },
                "operation": {
                    "type": "string",
                    "description": "Operation: add, subtract, multiply, divide",
                    "enum": ["add", "subtract", "multiply", "divide"]
                }
            },
            "required": ["a", "b", "operation"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.operation.as_str() {
            "add" => Ok(args.a + args.b),
            "subtract" => Ok(args.a - args.b),
            "multiply" => Ok(args.a * args.b),
            "divide" => {
                if args.b == 0.0 {
                    Err(ToolError::ExecutionError("Division by zero".to_string()))
                } else {
                    Ok(args.a / args.b)
                }
            }
            _ => Err(ToolError::InvalidArguments(format!(
                "Unknown operation: {}",
                args.operation
            ))),
        }
    }
}

/// Arguments for the string tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringToolArgs {
    /// Input text.
    pub text: String,
    /// Operation: uppercase, lowercase, reverse, length.
    pub operation: String,
}

/// A simple string manipulation tool.
#[derive(Debug, Clone, Copy, Default)]
pub struct StringTool;

#[async_trait]
impl Tool for StringTool {
    const NAME: &'static str = "string_tool";
    type Args = StringToolArgs;
    type Output = String;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Perform string operations (uppercase, lowercase, reverse, length).".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input text"
                },
                "operation": {
                    "type": "string",
                    "description": "Operation: uppercase, lowercase, reverse, length",
                    "enum": ["uppercase", "lowercase", "reverse", "length"]
                }
            },
            "required": ["text", "operation"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.operation.as_str() {
            "uppercase" => Ok(args.text.to_uppercase()),
            "lowercase" => Ok(args.text.to_lowercase()),
            "reverse" => Ok(args.text.chars().rev().collect()),
            "length" => Ok(args.text.len().to_string()),
            _ => Err(ToolError::InvalidArguments(format!(
                "Unknown operation: {}",
                args.operation
            ))),
        }
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("=== Machi Agent Example ===\n");

    // Create tools
    let calculator = CalculatorTool;
    let string_tool = StringTool;

    // Show tool definitions
    println!("Available tools:");
    println!("  - {}: {}", calculator.name(), calculator.description());
    println!("  - {}: {}", string_tool.name(), string_tool.description());
    println!();

    // Test tools directly
    println!("Testing tools directly:");

    let calc_args = CalculatorArgs {
        a: 10.0,
        b: 5.0,
        operation: "multiply".to_string(),
    };
    match calculator.call(calc_args).await {
        Ok(result) => println!("  Calculator: 10 * 5 = {}", result),
        Err(e) => println!("  Calculator error: {}", e),
    }

    let str_args = StringToolArgs {
        text: "Hello, Machi!".to_string(),
        operation: "uppercase".to_string(),
    };
    match string_tool.call(str_args).await {
        Ok(result) => println!("  String tool: uppercase of 'Hello, Machi!' = {}", result),
        Err(e) => println!("  String tool error: {}", e),
    }

    println!();

    // Note: To run with an actual LLM, you would need to set OPENAI_API_KEY
    // and use a real model:
    //
    // let model = machi::model::OpenAIModel::new("gpt-4");
    // let mut agent = ToolCallingAgent::builder()
    //     .model(model)
    //     .tool(Box::new(calculator))
    //     .tool(Box::new(string_tool))
    //     .max_steps(10)
    //     .build();
    //
    // let result = agent.run("What is 25 * 4?").await;

    println!("To run with an actual LLM, set OPENAI_API_KEY environment variable.");
    println!("Example usage:");
    println!(
        r#"
    let model = OpenAIModel::new("gpt-4");
    let mut agent = ToolCallingAgent::builder()
        .model(model)
        .tool(Box::new(CalculatorTool))
        .tool(Box::new(StringTool))
        .max_steps(10)
        .build();

    let result = agent.run("What is 25 * 4?").await;
    "#
    );
}
