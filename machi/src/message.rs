//! Message types for agent-model communication.
//!
//! This module defines the message format used for communication between
//! agents and language models, following the chat completion API conventions.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Write;

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message providing instructions.
    System,
    /// User message.
    User,
    /// Assistant (model) message.
    Assistant,
    /// Tool call message.
    #[serde(rename = "tool-call")]
    ToolCall,
    /// Tool response message.
    #[serde(rename = "tool-response")]
    ToolResponse,
}

impl MessageRole {
    /// Get the string representation of the role.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::ToolCall => "tool-call",
            Self::ToolResponse => "tool-response",
        }
    }
}

/// Content of a message, which can be text, image, audio, or other types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MessageContent {
    /// Text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Image content (base64 encoded).
    Image {
        /// The image data (base64 encoded).
        image: String,
        /// Optional format hint (e.g., "png", "jpeg").
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
    /// Image URL content (for vision model APIs).
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The image URL details.
        image_url: ImageUrlDetail,
    },
    /// Audio content (base64 encoded).
    Audio {
        /// The audio data (base64 encoded).
        audio: String,
        /// Sample rate in Hz.
        #[serde(skip_serializing_if = "Option::is_none")]
        sample_rate: Option<u32>,
        /// Optional format hint (e.g., "wav", "mp3").
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
}

/// Image URL structure with detail level for vision APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlDetail {
    /// The URL of the image (can be http(s) URL or data URL).
    pub url: String,
    /// Detail level for image processing: "low", "high", or "auto".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Type alias for backward compatibility.
pub type ImageUrl = ImageUrlDetail;

impl MessageContent {
    /// Create a new text content.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a new image content from base64 data.
    #[must_use]
    pub fn image(image: impl Into<String>) -> Self {
        Self::Image {
            image: image.into(),
            format: None,
        }
    }

    /// Create a new image content with format hint.
    #[must_use]
    pub fn image_with_format(image: impl Into<String>, format: impl Into<String>) -> Self {
        Self::Image {
            image: image.into(),
            format: Some(format.into()),
        }
    }

    /// Create a new image URL content.
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrlDetail {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create a new image URL content with detail level.
    ///
    /// Detail can be "low", "high", or "auto" (default).
    #[must_use]
    pub fn image_url_with_detail(url: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrlDetail {
                url: url.into(),
                detail: Some(detail.into()),
            },
        }
    }

    /// Create image content from an `AgentImage`.
    ///
    /// This will use the data URL if available, otherwise the URL reference.
    /// Returns `None` if the image has no accessible data.
    #[must_use]
    pub fn from_agent_image(image: &crate::multimodal::AgentImage) -> Option<Self> {
        if let Some(url) = image.as_url() {
            // Use URL directly for URL-based images
            Some(Self::image_url_with_detail(url, "auto"))
        } else {
            image
                .to_data_url()
                .map(|data_url| Self::image_url_with_detail(data_url, "auto"))
        }
    }

    /// Create a new audio content from base64 data.
    #[must_use]
    pub fn audio(audio: impl Into<String>) -> Self {
        Self::Audio {
            audio: audio.into(),
            sample_rate: None,
            format: None,
        }
    }

    /// Create a new audio content with sample rate.
    #[must_use]
    pub fn audio_with_sample_rate(audio: impl Into<String>, sample_rate: u32) -> Self {
        Self::Audio {
            audio: audio.into(),
            sample_rate: Some(sample_rate),
            format: None,
        }
    }

    /// Create audio content from an `AgentAudio`.
    #[must_use]
    pub fn from_agent_audio(audio: &crate::multimodal::AgentAudio) -> Option<Self> {
        audio.to_base64().map(|b64| Self::Audio {
            audio: b64.into_owned(),
            sample_rate: Some(audio.sample_rate()),
            format: Some(audio.format().extension().to_string()),
        })
    }

    /// Get the text content if this is a text message.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Check if this is an image content (either embedded or URL).
    #[must_use]
    pub const fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. } | Self::ImageUrl { .. })
    }

    /// Check if this is an audio content.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Audio { .. })
    }

    /// Check if this content is multimodal (image or audio).
    #[must_use]
    pub const fn is_multimodal(&self) -> bool {
        self.is_image() || self.is_audio()
    }
}

/// Function call information in a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Name of the function to call.
    pub name: String,
    /// Arguments to pass to the function (as JSON string or object).
    pub arguments: Value,
    /// Optional description of the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A tool call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageToolCall {
    /// Unique identifier for the tool call.
    pub id: String,
    /// Type of the tool call (usually "function").
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function to call.
    pub function: ToolCallFunction,
}

impl ChatMessageToolCall {
    /// Create a new tool call.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.into(),
                arguments,
                description: None,
            },
        }
    }

    /// Get the name of the function being called.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// Get the arguments as a JSON value.
    #[must_use]
    pub const fn arguments(&self) -> &Value {
        &self.function.arguments
    }

    /// Parse arguments as a typed value.
    pub fn parse_arguments<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        // Handle both string and object arguments
        match &self.function.arguments {
            Value::String(s) => serde_json::from_str(s),
            other => serde_json::from_value(other.clone()),
        }
    }

    /// Get arguments as a JSON string.
    #[must_use]
    pub fn arguments_string(&self) -> String {
        match &self.function.arguments {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }
}

/// A chat message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender.
    pub role: MessageRole,
    /// Content of the message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<MessageContent>>,
    /// Tool calls made by the model (for assistant messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatMessageToolCall>>,
    /// Tool call ID (for tool response messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a new system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(vec![MessageContent::text(content)]),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a new user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(vec![MessageContent::text(content)]),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a new assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(vec![MessageContent::text(content)]),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a new assistant message with tool calls.
    #[must_use]
    pub const fn assistant_with_tool_calls(tool_calls: Vec<ChatMessageToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Create a new tool response message.
    #[must_use]
    pub fn tool_response(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::ToolResponse,
            content: Some(vec![MessageContent::text(content)]),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Create a new message with multiple content items.
    #[must_use]
    pub const fn with_contents(role: MessageRole, contents: Vec<MessageContent>) -> Self {
        Self {
            role,
            content: Some(contents),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Get the text content of the message.
    #[must_use]
    pub fn text_content(&self) -> Option<String> {
        self.content.as_ref().map(|contents| {
            contents
                .iter()
                .filter_map(MessageContent::as_text)
                .collect::<Vec<_>>()
                .join("\n")
        })
    }

    /// Check if this message has tool calls.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }

    /// Render the message as markdown.
    #[must_use]
    pub fn render_as_markdown(&self) -> String {
        let mut result = self.text_content().unwrap_or_default();

        if let Some(tool_calls) = &self.tool_calls {
            for call in tool_calls {
                let _ = write!(
                    result,
                    "\n[Tool Call: {} with args: {}]",
                    call.function.name, call.function.arguments
                );
            }
        }

        result
    }
}

/// Streaming delta for incremental message updates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessageStreamDelta {
    /// Incremental content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Incremental tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatMessageToolCallStreamDelta>>,
    /// Token usage information (usually only in final delta).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<crate::providers::common::TokenUsage>,
}

/// Type alias for backwards compatibility.
pub type ToolCallStreamDelta = ChatMessageToolCallStreamDelta;

/// Streaming delta for tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageToolCallStreamDelta {
    /// Index of the tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Tool call ID (may be partial).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Type of tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    /// Function information (may be partial).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ChatMessageToolCallFunction>,
}

/// Tool call function information for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageToolCallFunction {
    /// Function name.
    pub name: String,
    /// Arguments as JSON value.
    pub arguments: Value,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Aggregate streaming deltas into a complete message.
#[must_use]
pub fn aggregate_stream_deltas(deltas: &[ChatMessageStreamDelta]) -> ChatMessage {
    let mut content = String::new();
    let mut tool_calls: std::collections::HashMap<usize, ChatMessageToolCall> =
        std::collections::HashMap::new();

    for delta in deltas {
        if let Some(c) = &delta.content {
            content.push_str(c);
        }

        if let Some(tc_deltas) = &delta.tool_calls {
            for tc_delta in tc_deltas {
                let index = tc_delta.index.unwrap_or(0);
                let entry = tool_calls
                    .entry(index)
                    .or_insert_with(|| ChatMessageToolCall {
                        id: String::new(),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: String::new(),
                            arguments: Value::String(String::new()),
                            description: None,
                        },
                    });

                if let Some(id) = &tc_delta.id {
                    entry.id.clone_from(id);
                }
                if let Some(func) = &tc_delta.function {
                    entry.function.name.clone_from(&func.name);
                    // Serialize arguments to string and append
                    if let Ok(args_str) = serde_json::to_string(&func.arguments)
                        && let Value::String(s) = &mut entry.function.arguments
                    {
                        s.push_str(&args_str);
                    }
                }
            }
        }
    }

    let tool_calls_vec: Vec<ChatMessageToolCall> = if tool_calls.is_empty() {
        Vec::new()
    } else {
        let mut calls: Vec<_> = tool_calls.into_iter().collect();
        calls.sort_by_key(|(idx, _)| *idx);
        calls.into_iter().map(|(_, call)| call).collect()
    };

    ChatMessage {
        role: MessageRole::Assistant,
        content: if content.is_empty() {
            None
        } else {
            Some(vec![MessageContent::text(content)])
        },
        tool_calls: if tool_calls_vec.is_empty() {
            None
        } else {
            Some(tool_calls_vec)
        },
        tool_call_id: None,
    }
}
