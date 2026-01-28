//! LLM backend adapters for agent reasoning.
//!
//! This module defines the [`Backend`] trait that abstracts over different
//! LLM providers (OpenAI, Anthropic, local models, etc.).
//!
//! # Supported Backends
//!
//! - `rig` - Via the rig crate (feature = "rig")

use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::tools::ToolDefinition;

#[cfg(feature = "rig")]
pub mod rig;

/// A tool call request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call (used in multi-turn conversations).
    pub id: String,
    /// The name of the tool to call.
    pub name: String,
    /// The arguments as a JSON value.
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// Result of a tool execution to send back to LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool call ID this result corresponds to.
    pub tool_call_id: String,
    /// The result content.
    pub content: String,
}

impl ToolResult {
    /// Create a new tool result.
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
}

/// Response from the LLM backend.
#[derive(Debug, Clone)]
pub enum BackendResponse {
    /// A final text response (conversation complete).
    Text(String),
    /// A request to call one or more tools.
    ToolCalls(Vec<ToolCall>),
}

impl BackendResponse {
    /// Check if this is a final text response.
    pub fn is_text(&self) -> bool {
        matches!(self, BackendResponse::Text(_))
    }

    /// Check if this response contains tool calls.
    pub fn is_tool_calls(&self) -> bool {
        matches!(self, BackendResponse::ToolCalls(_))
    }

    /// Get the text content if this is a text response.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            BackendResponse::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get the tool calls if this is a tool call response.
    pub fn as_tool_calls(&self) -> Option<&[ToolCall]> {
        match self {
            BackendResponse::ToolCalls(calls) => Some(calls),
            _ => None,
        }
    }
}

/// Trait for LLM backend adapters.
///
/// Implement this trait to add support for new LLM providers.
pub trait Backend: Send + Sync {
    /// Complete a prompt and return a text response.
    fn complete(&self, prompt: &str) -> impl Future<Output = Result<String>> + Send;

    /// Complete a prompt with tool definitions available.
    /// Returns either a text response or tool calls.
    fn complete_with_tools(
        &self,
        prompt: &str,
        tools: &[ToolDefinition],
    ) -> impl Future<Output = Result<BackendResponse>> + Send;

    /// Continue conversation after tool execution.
    /// Sends tool results back to LLM and gets next response.
    fn continue_with_tool_results(
        &self,
        tool_results: &[ToolResult],
        tools: &[ToolDefinition],
    ) -> impl Future<Output = Result<BackendResponse>> + Send;
}
