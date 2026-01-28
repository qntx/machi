//! Tool system for agent capabilities.
//!
//! This module provides the tool abstraction that allows agents to interact
//! with wallets, blockchains, and external services.
//!
//! # Architecture
//!
//! - [`Tool`] trait: Define executable tools
//! - [`ToolDefinition`]: Schema for LLM function calling
//! - [`ToolContext`]: Runtime context with wallet/chain access
//! - [`builtin`]: Built-in wallet operation tools
//!
//! # Custom Tools Example
//!
//! ```ignore
//! use machi::tools::{Tool, ToolDefinition, ToolContext, ToolOutput};
//!
//! struct MyCustomTool;
//!
//! impl<C: Chain> Tool<C> for MyCustomTool {
//!     fn definition(&self) -> ToolDefinition {
//!         ToolDefinition::new("my_tool", "Does something")
//!     }
//!
//!     fn call<'a>(&'a self, ctx: &'a ToolContext<'a, C>, args: Value)
//!         -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>>
//!     {
//!         Box::pin(async move { ToolOutput::ok("done") })
//!     }
//! }
//! ```

pub mod executor;
pub mod tool;
pub mod wallet;

// Re-exports for convenience
pub use executor::{BoxedTool, Tool, ToolContext};
pub use tool::{ToolDefinition, ToolOutput, ToolParam};
pub use wallet::{GetAddress, GetBalance, SendTransaction, builtin_tool_definitions};
