//! Approval gate for destructive or privileged tool calls.

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{ToolError, codes};
use crate::metadata::ToolMetadata;
use crate::tool::DynTool;

/// Decision for a pending tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ApprovalDecision {
    /// Allow execution.
    Allow,
    /// Deny execution (fail-closed to the model as a tool error).
    Deny,
}

/// Host-supplied gate consulted before running tools that need confirmation.
#[async_trait]
pub trait ApprovalGate: Send + Sync {
    /// Decide whether `tool` may run with `arguments`.
    async fn approve(
        &self,
        tool: &dyn DynTool,
        metadata: &ToolMetadata,
        arguments: &Value,
    ) -> Result<ApprovalDecision, ToolError>;
}

/// Always allows (library tests / trusted offline hosts).
#[derive(Debug, Default, Clone, Copy)]
pub struct AutoApprove;

#[async_trait]
impl ApprovalGate for AutoApprove {
    async fn approve(
        &self,
        _tool: &dyn DynTool,
        _metadata: &ToolMetadata,
        _arguments: &Value,
    ) -> Result<ApprovalDecision, ToolError> {
        Ok(ApprovalDecision::Allow)
    }
}

/// Always denies (negative tests).
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysDeny;

#[async_trait]
impl ApprovalGate for AlwaysDeny {
    async fn approve(
        &self,
        tool: &dyn DynTool,
        _metadata: &ToolMetadata,
        _arguments: &Value,
    ) -> Result<ApprovalDecision, ToolError> {
        Err(codes::approval_denied(format!(
            "approval denied for tool {}",
            tool.name()
        )))
    }
}

/// Map deny decision to a tool error.
#[must_use]
pub fn denied_error(tool_name: &str) -> ToolError {
    codes::approval_denied(format!("approval denied for tool {tool_name}"))
}
