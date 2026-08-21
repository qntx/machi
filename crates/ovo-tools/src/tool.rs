//! Tool definition and dynamic execution surface.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::metadata::ToolMetadata;
use crate::stream::{ToolStream, terminal_only};

/// JSON-schema facing tool definition for model APIs.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "JSON Schema Value is not Eq"
)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Description.
    pub description: String,
    /// JSON Schema for parameters.
    pub parameters: Value,
}

/// Successful tool output returned to the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "optional JSON Value is not Eq"
)]
pub struct ToolResult {
    /// Text content for the tool message.
    pub content: String,
    /// Optional structured payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured: Option<Value>,
    /// Whether the tool reported a logical failure (still a completed call).
    #[serde(default)]
    pub is_error: bool,
}

impl ToolResult {
    /// Successful text result.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            structured: None,
            is_error: false,
        }
    }

    /// Error-shaped tool result (for model consumption).
    #[must_use]
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            structured: None,
            is_error: true,
        }
    }
}

/// Object-safe tool.
#[async_trait]
pub trait DynTool: Send + Sync {
    /// Tool name.
    fn name(&self) -> &str;
    /// Description.
    fn description(&self) -> &str;
    /// JSON schema parameters.
    fn parameters(&self) -> Value;
    /// Metadata.
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::default()
    }
    /// Model-facing definition.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_owned(),
            description: self.description().to_owned(),
            parameters: self.parameters(),
        }
    }
    /// Execute with JSON arguments (blocking convenience).
    ///
    /// Prefer overriding [`DynTool::execute`] when the tool emits progress.
    async fn call(&self, ctx: ToolCallContext, arguments: Value) -> Result<ToolResult, ToolError>;

    /// Streaming entry point. Default wraps [`DynTool::call`] as a single terminal.
    async fn execute(&self, ctx: ToolCallContext, arguments: Value) -> ToolStream {
        let result = self.call(ctx, arguments).await;
        terminal_only(result)
    }
}

/// Shared tool handle.
pub type SharedTool = Arc<dyn DynTool>;

/// Boxed async future for dispatch internals.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
