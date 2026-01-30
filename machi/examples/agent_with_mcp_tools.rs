//! MCP Client example - connects to an MCP server and uses tools via machi agent.
//!
//! First start the server: `cargo run -p machi --example mcp_server --features rmcp`
//! Then run this client: `cargo run -p machi --example agent_with_mcp_tools --features rmcp`

use machi::client::Nothing;
use machi::completion::Prompt;
use machi::mcp::McpClient;
use machi::prelude::*;
use machi::providers::ollama::{self, QWEN3};

const SERVER_URL: &str = "http://localhost:8080";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Connect to MCP server - supports both HTTP and stdio transports
    // HTTP (remote server):
    let mcp = McpClient::http(SERVER_URL).await?;
    // Stdio (local process): McpClient::stdio("python", &["server.py"]).await?

    println!("Available tools: {:?}", mcp.tool_names());

    // Create machi agent with MCP tools
    let client = ollama::Client::from_val(Nothing);
    let agent = client
        .agent(QWEN3)
        .preamble("You are a helpful assistant. Use the available tools to answer questions.")
        .mcp(mcp)
        .build();

    // Use agent with MCP tools
    let response = agent.prompt("What is 2 + 5?").multi_turn(2).await?;

    println!("Response: {response}");

    Ok(())
}
