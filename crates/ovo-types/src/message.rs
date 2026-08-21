//! Conversation message model.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::id::ToolCallId;

/// Participant role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    /// System instructions.
    System,
    /// End-user content.
    #[default]
    User,
    /// Model content.
    Assistant,
    /// Tool result content.
    Tool,
    /// Provider-specific developer role.
    Developer,
}

impl Role {
    /// Stable string form.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Developer => "developer",
        }
    }
}

/// Image MIME types commonly used in multimodal prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ImageMime {
    /// JPEG.
    #[default]
    Jpeg,
    /// PNG.
    Png,
    /// GIF.
    Gif,
    /// WebP.
    WebP,
}

impl ImageMime {
    /// MIME string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::WebP => "image/webp",
        }
    }
}

/// One content part of a multimodal message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    /// Plain text.
    Text {
        /// Text body.
        text: String,
    },
    /// Inline image bytes (base64) or URL.
    Image {
        /// MIME type.
        mime: ImageMime,
        /// Data URL or https URL.
        url: String,
    },
}

/// A model-emitted tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "serde_json::Value is not Eq"
)]
pub struct ToolCall {
    /// Call id for pairing with tool results.
    pub id: ToolCallId,
    /// Tool name as presented to the model.
    pub name: String,
    /// JSON arguments object or raw string payload.
    pub arguments: Value,
}

/// A single conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(
    clippy::derive_partial_eq_without_eq,
    reason = "contains ToolCall with JSON Value"
)]
pub struct Message {
    /// Role.
    pub role: Role,
    /// Text content when not multimodal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Multimodal parts (preferred when present).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<ContentPart>,
    /// Tool calls from the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Tool call id when role is tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<ToolCallId>,
    /// Tool name when role is tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// System message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// User message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Assistant text message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Assistant message with tool calls.
    #[must_use]
    pub const fn assistant_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            parts: Vec::new(),
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }

    /// Tool result message.
    #[must_use]
    pub fn tool_result(
        tool_call_id: ToolCallId,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id),
            name: Some(name.into()),
        }
    }

    /// Best-effort plain text extraction.
    #[must_use]
    pub fn text(&self) -> String {
        if let Some(content) = &self.content {
            return content.clone();
        }
        self.parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                ContentPart::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_user() {
        let m = Message::user("hello");
        let json = serde_json::to_string(&m).expect("ser");
        let back: Message = serde_json::from_str(&json).expect("de");
        assert_eq!(back.role, Role::User);
        assert_eq!(back.text(), "hello");
    }

    #[test]
    fn tool_call_message() {
        let id = ToolCallId::new("call_1").expect("id");
        let m = Message::assistant_tools(vec![ToolCall {
            id: id.clone(),
            name: "add".into(),
            arguments: serde_json::json!({"a":1,"b":2}),
        }]);
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls.first().map(|c| &c.id), Some(&id));
    }
}
