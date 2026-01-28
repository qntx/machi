//! Tool execution context and executor.
//!
//! Provides the runtime context for executing tools with access to
//! wallet and chain resources.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use super::tool::{ToolDefinition, ToolOutput};
use crate::wallet::AgentWallet;

/// Context passed to tools during execution.
///
/// Provides access to the agent's wallet and chain adapter.
pub struct ToolContext<'a, C> {
    /// Reference to the agent's wallet.
    pub wallet: &'a AgentWallet,
    /// Reference to the chain adapter.
    pub chain: &'a C,
}

/// Trait for executable tools.
///
/// Implement this trait to create custom tools that can be used by agents.
///
/// # Example
///
/// ```ignore
/// struct MyTool;
///
/// impl<C: Chain> Tool<C> for MyTool {
///     fn definition(&self) -> ToolDefinition {
///         ToolDefinition::new("my_tool", "Does something useful")
///             .param("input", "string", "The input value")
///     }
///
///     fn call<'a>(
///         &'a self,
///         ctx: &'a ToolContext<'a, C>,
///         args: Value,
///     ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
///         Box::pin(async move {
///             // Implementation
///             ToolOutput::ok("result")
///         })
///     }
/// }
/// ```
pub trait Tool<C>: Send + Sync {
    /// Get the tool definition for LLM function calling.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given arguments.
    fn call<'a>(
        &'a self,
        ctx: &'a ToolContext<'a, C>,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>>;
}

/// A type-erased tool that can be stored in collections.
pub type BoxedTool<C> = Box<dyn Tool<C>>;
