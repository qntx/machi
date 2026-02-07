//! Agent with tool execution policies and human confirmation.
//!
//! Demonstrates `ToolExecutionPolicy` to control which tools require
//! human approval before execution. The user is prompted in the
//! terminal to approve or deny each tool call.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example agent_openai_policy
//! ```

#![allow(clippy::print_stdout, clippy::unused_async, clippy::unnecessary_wraps)]

use async_trait::async_trait;
use machi::prelude::*;
use std::io::{Write, stdout};
use std::sync::Arc;

/// A confirmation handler that prompts the user in the terminal.
struct TerminalConfirmation;

#[async_trait]
impl ConfirmationHandler for TerminalConfirmation {
    async fn confirm(&self, request: &ToolConfirmationRequest) -> ToolConfirmationResponse {
        println!("\n========================================");
        println!("Tool:      {}", request.name);
        println!("Arguments: {}", request.arguments);
        println!("========================================");
        print!("Approve? [y]es / [n]o / [a]ll > ");
        stdout().flush().expect("flush stdout");

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).expect("read stdin");

        match input.trim().to_lowercase().as_str() {
            "y" | "yes" => ToolConfirmationResponse::Approved,
            "a" | "all" => ToolConfirmationResponse::ApproveAll,
            _ => ToolConfirmationResponse::Denied,
        }
    }
}

/// Delete a file (mock — prints instead of actually deleting).
#[tool]
async fn delete_file(path: String) -> ToolResult<String> {
    // In a real app this would delete a file — hence it needs confirmation.
    Ok(format!("Deleted file: {path}"))
}

/// Read a file (mock — safe, no confirmation needed).
#[tool]
async fn read_file(path: String) -> ToolResult<String> {
    Ok(format!("Contents of {path}: Hello, world!"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: SharedChatProvider = Arc::new(OpenAI::from_env()?);

    let agent = Agent::new("file-manager")
        .instructions(
            "You are a file manager assistant. \
             Use read_file to read files and delete_file to delete files.",
        )
        .model("gpt-4o-mini")
        .provider(provider)
        .tool(Box::new(READ_FILE))
        .tool(Box::new(DELETE_FILE))
        // read_file runs automatically; delete_file requires human confirmation.
        .tool_policy("read_file", ToolExecutionPolicy::Auto)
        .tool_policy("delete_file", ToolExecutionPolicy::RequireConfirmation);

    let config = RunConfig::default()
        .confirmation_handler(Arc::new(TerminalConfirmation));

    let result = agent
        .run("Read config.txt, then delete temp.log", config)
        .await?;

    println!("\n{}", result.output);

    Ok(())
}
