//! Model Context Protocol (MCP) integration module.
//!
//! This module provides a high-level API for connecting to MCP servers,
//! supporting both local (stdio) and remote (HTTP) transports.
//!
//! # Quick Start
//!
//! ## Remote Server (HTTP)
//!
//! ```rust,ignore
//! use machi::mcp::McpClient;
//!
//! let client = McpClient::http("http://localhost:8080").await?;
//! println!("Tools: {:?}", client.tool_names());
//! ```
//!
//! ## Local Server (stdio subprocess)
//!
//! ```rust,ignore
//! use machi::mcp::McpClient;
//!
//! let client = McpClient::stdio("python", &["my_mcp_server.py"]).await?;
//! println!("Tools: {:?}", client.tool_names());
//! ```
//!
//! # Multiple Servers
//!
//! ```rust,ignore
//! use machi::mcp::McpClientBuilder;
//!
//! let clients = McpClientBuilder::new()
//!     .http("math", "http://localhost:8080")
//!     .stdio("local", "python", &["server.py"])
//!     .connect_all()
//!     .await?;
//! ```
//!
//! # Integration with Agent
//!
//! ```rust,ignore
//! let mcp = McpClient::http("http://localhost:8080").await?;
//! let agent = client
//!     .agent(model)
//!     .preamble("You are a helpful assistant.")
//!     .mcp(mcp)
//!     .build();
//! ```

mod client;
mod error;
mod tool;
mod transport;

pub use client::{IntoMcpTools, McpClient, McpClientBuilder, McpClientConfig, MergedMcpClients};
pub use error::McpError;
pub use tool::McpTool;
pub use transport::TransportConfig;
