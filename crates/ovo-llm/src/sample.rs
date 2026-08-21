//! Sample request/response types.

use ovo_tools::ToolDefinition;
use ovo_types::{Deadline, Message, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// Whether the model must/may/must-not call tools.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// Provider default.
    #[default]
    Auto,
    /// Disallow tools.
    None,
    /// Require a tool call.
    Required,
    /// Force a specific tool name.
    Named(String),
}

/// One sampling request.
#[derive(Debug, Clone)]
pub struct SampleRequest {
    /// Model id.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Available tools.
    pub tools: Vec<ToolDefinition>,
    /// Tool choice policy.
    pub tool_choice: ToolChoice,
    /// Optional JSON schema / response format name.
    pub response_format: Option<Value>,
    /// Max output tokens.
    pub max_output_tokens: Option<u32>,
    /// Temperature.
    pub temperature: Option<f32>,
    /// Cancel token.
    pub cancel: CancellationToken,
    /// Optional deadline.
    pub deadline: Option<Deadline>,
}

/// Sampling result.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleResponse {
    /// Assistant message (text and/or tool calls).
    pub message: Message,
    /// Token usage when known.
    pub usage: Usage,
    /// Provider stop reason (opaque).
    pub stop_reason: Option<String>,
}
