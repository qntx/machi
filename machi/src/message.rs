//! Unified message types for LLM chat interactions.
//!
//! This module provides a clean, type-safe API for constructing and manipulating
//! chat messages compatible with OpenAI-style LLM APIs.
//!
//! # Design Principles
//!
//! - **Single Source of Truth**: One `Role` enum, one `Message` struct
//! - **Builder Pattern**: Fluent API for message construction
//! - **Type Safety**: Strong typing for content variants
//! - **Serialization Ready**: All types derive Serialize/Deserialize

use std::fmt;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::audio::AudioFormat;

/// Role of a message participant in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    /// System message providing instructions to the model.
    System,
    /// User message from the human.
    #[default]
    User,
    /// Assistant message from the model.
    Assistant,
    /// Tool/function response message.
    Tool,
    /// Developer message for OpenAI o1/o3 models.
    Developer,
}

impl Role {
    /// Returns the string representation.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Developer => "developer",
        }
    }

    /// Returns `true` if this is a system role.
    #[inline]
    #[must_use]
    pub const fn is_system(&self) -> bool {
        matches!(self, Self::System)
    }

    /// Returns `true` if this is a user role.
    #[inline]
    #[must_use]
    pub const fn is_user(&self) -> bool {
        matches!(self, Self::User)
    }

    /// Returns `true` if this is an assistant role.
    #[inline]
    #[must_use]
    pub const fn is_assistant(&self) -> bool {
        matches!(self, Self::Assistant)
    }

    /// Returns `true` if this is a tool role.
    #[inline]
    #[must_use]
    pub const fn is_tool(&self) -> bool {
        matches!(self, Self::Tool)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Supported MIME types for images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ImageMime {
    /// JPEG image.
    #[default]
    Jpeg,
    /// PNG image.
    Png,
    /// GIF image.
    Gif,
    /// WebP image.
    WebP,
}

impl ImageMime {
    /// Returns the MIME type string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::WebP => "image/webp",
        }
    }

    /// Detects MIME type from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "png" => Some(Self::Png),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::WebP),
            _ => None,
        }
    }

    /// Detects MIME type from magic bytes.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        match &data[..4] {
            [0xFF, 0xD8, 0xFF, ..] => Some(Self::Jpeg),
            [0x89, 0x50, 0x4E, 0x47] => Some(Self::Png),
            [0x47, 0x49, 0x46, 0x38] => Some(Self::Gif),
            [0x52, 0x49, 0x46, 0x46] if data.len() >= 12 && &data[8..12] == b"WEBP" => {
                Some(Self::WebP)
            }
            _ => None,
        }
    }
}

impl fmt::Display for ImageMime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Detail level for image processing in vision models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Low resolution processing (faster, cheaper).
    Low,
    /// High resolution processing (more accurate).
    High,
    /// Let the model decide based on image size.
    #[default]
    Auto,
}

/// Input audio data for audio messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data.
    pub data: String,
    /// Audio format.
    pub format: AudioFormat,
}

impl InputAudio {
    /// Creates a new input audio from base64-encoded data.
    #[must_use]
    pub fn new(data: impl Into<String>, format: AudioFormat) -> Self {
        Self {
            data: data.into(),
            format,
        }
    }

    /// Creates input audio from raw bytes.
    #[must_use]
    pub fn from_bytes(data: &[u8], format: AudioFormat) -> Self {
        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
        Self::new(encoded, format)
    }
}

/// A single content part within a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content.
    Text {
        /// The text string.
        text: String,
    },
    /// Image URL content (including data URLs).
    ImageUrl {
        /// Image URL details.
        image_url: ImageUrl,
    },
    /// Input audio content (for audio-capable models).
    InputAudio {
        /// Audio input details.
        input_audio: InputAudio,
    },
}

/// Image URL with optional detail level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The URL (http/https or data URL).
    pub url: String,
    /// Detail level for processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

impl ContentPart {
    /// Creates a text content part.
    #[inline]
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates an image URL content part.
    #[inline]
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Creates an image URL content part with detail level.
    #[inline]
    #[must_use]
    pub fn image_url_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(detail),
            },
        }
    }

    /// Creates an image content part from raw bytes.
    #[must_use]
    pub fn image_bytes(data: &[u8], mime: ImageMime) -> Self {
        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
        let data_url = format!("data:{};base64,{}", mime.as_str(), encoded);
        Self::image_url(data_url)
    }

    /// Creates an image content part from raw bytes with auto-detected MIME type.
    #[must_use]
    pub fn image_bytes_auto(data: &[u8]) -> Self {
        let mime = ImageMime::from_bytes(data).unwrap_or(ImageMime::Jpeg);
        Self::image_bytes(data, mime)
    }

    /// Creates an input audio content part.
    #[inline]
    #[must_use]
    pub fn input_audio(data: impl Into<String>, format: AudioFormat) -> Self {
        Self::InputAudio {
            input_audio: InputAudio::new(data, format),
        }
    }

    /// Creates an input audio content part from raw bytes.
    #[must_use]
    pub fn input_audio_bytes(data: &[u8], format: AudioFormat) -> Self {
        Self::InputAudio {
            input_audio: InputAudio::from_bytes(data, format),
        }
    }

    /// Returns the text content if this is a text part.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Returns `true` if this is a text content part.
    #[inline]
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Returns `true` if this is an image content part.
    #[inline]
    #[must_use]
    pub const fn is_image(&self) -> bool {
        matches!(self, Self::ImageUrl { .. })
    }

    /// Returns `true` if this is an audio content part.
    #[inline]
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::InputAudio { .. })
    }
}

/// Message content that can be either simple text or multipart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Simple text content (most common case).
    Text(String),
    /// Array of content parts (for multimodal messages).
    Parts(Vec<ContentPart>),
}

impl Content {
    /// Creates simple text content.
    #[inline]
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Creates multipart content from parts.
    #[inline]
    #[must_use]
    pub const fn parts(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }

    /// Extracts all text content, joining multiple text parts with newlines.
    #[must_use]
    pub fn as_text(&self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text.clone()),
            Self::Parts(parts) => {
                let texts: Vec<&str> = parts.iter().filter_map(ContentPart::as_text).collect();
                if texts.is_empty() {
                    None
                } else {
                    Some(texts.join("\n"))
                }
            }
        }
    }

    /// Returns `true` if the content contains any images.
    #[must_use]
    pub fn has_images(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(ContentPart::is_image),
        }
    }

    /// Returns `true` if the content is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.is_empty(),
        }
    }
}

impl Default for Content {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for Content {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

impl From<Vec<ContentPart>> for Content {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
}

/// A function call within a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call.
    pub name: String,
    /// Arguments as a JSON string.
    pub arguments: String,
}

impl FunctionCall {
    /// Creates a new function call.
    #[inline]
    #[must_use]
    pub fn new(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Parses arguments as a typed value.
    pub fn parse_arguments<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }

    /// Returns arguments as a JSON value.
    #[must_use]
    pub fn arguments_value(&self) -> Value {
        serde_json::from_str(&self.arguments).unwrap_or(Value::Null)
    }
}

/// A tool call made by the assistant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// Type of tool call (always "function" currently).
    #[serde(rename = "type", default = "default_tool_type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

fn default_tool_type() -> String {
    "function".to_owned()
}

impl ToolCall {
    /// Creates a new function tool call.
    #[must_use]
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            call_type: "function".to_owned(),
            function: FunctionCall::new(name, arguments),
        }
    }

    /// Returns the function name.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// Returns the function arguments.
    #[inline]
    #[must_use]
    pub fn arguments(&self) -> &str {
        &self.function.arguments
    }

    /// Parse arguments as a typed value.
    pub fn parse_arguments<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        self.function.parse_arguments()
    }
}

impl fmt::Display for ToolCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.function.name, self.function.arguments)
    }
}

/// A thinking block from Anthropic Claude models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingBlock {
    /// Standard thinking block with visible content.
    Thinking {
        /// The thinking/reasoning content.
        thinking: String,
        /// Signature for verification (if provided).
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Redacted thinking block (content hidden).
    RedactedThinking {
        /// Opaque data representing redacted content.
        data: String,
    },
}

impl ThinkingBlock {
    /// Creates a new thinking block.
    #[must_use]
    pub fn thinking(content: impl Into<String>) -> Self {
        Self::Thinking {
            thinking: content.into(),
            signature: None,
        }
    }

    /// Returns the thinking content if this is a standard thinking block.
    #[must_use]
    pub fn as_thinking(&self) -> Option<&str> {
        match self {
            Self::Thinking { thinking, .. } => Some(thinking),
            Self::RedactedThinking { .. } => None,
        }
    }
}

/// Annotation on an assistant message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// URL citation annotation.
    UrlCitation {
        /// Start index in the text.
        start_index: usize,
        /// End index in the text.
        end_index: usize,
        /// The cited URL.
        url: String,
        /// Title of the cited content.
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// File citation annotation.
    FileCitation {
        /// File ID reference.
        file_id: String,
        /// Quote from the file.
        #[serde(skip_serializing_if = "Option::is_none")]
        quote: Option<String>,
    },
}

/// A chat message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: Role,

    /// Content of the message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,

    /// Refusal message if the model declined to respond.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,

    /// Annotations on the message (citations, etc.).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<Annotation>,

    /// Tool calls made by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID this message responds to (for tool role).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Name associated with the message (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Reasoning content from OpenAI o1/o3 models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    /// Thinking blocks from Anthropic Claude models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<ThinkingBlock>>,
}

impl Message {
    /// Creates a new message with the given role and content.
    #[must_use]
    pub fn new(role: Role, content: impl Into<Content>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            refusal: None,
            annotations: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking_blocks: None,
        }
    }

    /// Creates a system message.
    #[inline]
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, Content::text(content))
    }

    /// Creates a user message.
    #[inline]
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, Content::text(content))
    }

    /// Creates an assistant message.
    #[inline]
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, Content::text(content))
    }

    /// Creates an assistant message with tool calls (no text content).
    #[must_use]
    pub const fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            refusal: None,
            annotations: Vec::new(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking_blocks: None,
        }
    }

    /// Creates a tool response message.
    #[must_use]
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(Content::text(content)),
            refusal: None,
            annotations: Vec::new(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
            reasoning_content: None,
            thinking_blocks: None,
        }
    }

    /// Creates a builder for constructing a message.
    #[inline]
    #[must_use]
    pub const fn builder(role: Role) -> MessageBuilder {
        MessageBuilder::new(role)
    }

    /// Returns the text content of the message.
    #[must_use]
    pub fn text(&self) -> Option<String> {
        self.content.as_ref().and_then(Content::as_text)
    }

    /// Returns `true` if the message has tool calls.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }

    /// Returns `true` if the message contains images.
    #[must_use]
    pub fn has_images(&self) -> bool {
        self.content.as_ref().is_some_and(Content::has_images)
    }

    /// Returns `true` if the message has no content and no tool calls.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let no_content = self.content.as_ref().is_none_or(Content::is_empty);
        let no_tools = !self.has_tool_calls();
        no_content && no_tools
    }

    /// Sets the name field.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl Default for Message {
    fn default() -> Self {
        Self {
            role: Role::User,
            content: None,
            refusal: None,
            annotations: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking_blocks: None,
        }
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] ", self.role)?;
        if let Some(text) = self.text() {
            write!(f, "{text}")?;
        }
        if let Some(tool_calls) = &self.tool_calls {
            for tc in tool_calls {
                write!(f, " [{tc}]")?;
            }
        }
        Ok(())
    }
}

/// Builder for constructing messages with a fluent API.
#[derive(Debug, Clone)]
pub struct MessageBuilder {
    role: Role,
    parts: Vec<ContentPart>,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
    name: Option<String>,
}

impl MessageBuilder {
    /// Creates a new builder with the specified role.
    #[inline]
    #[must_use]
    pub const fn new(role: Role) -> Self {
        Self {
            role,
            parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// Adds text content.
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(ContentPart::text(text));
        self
    }

    /// Adds an image URL.
    #[must_use]
    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.parts.push(ContentPart::image_url(url));
        self
    }

    /// Adds an image URL with detail level.
    #[must_use]
    pub fn image_url_with_detail(mut self, url: impl Into<String>, detail: ImageDetail) -> Self {
        self.parts
            .push(ContentPart::image_url_with_detail(url, detail));
        self
    }

    /// Adds an image from raw bytes.
    #[must_use]
    pub fn image_bytes(mut self, data: &[u8], mime: ImageMime) -> Self {
        self.parts.push(ContentPart::image_bytes(data, mime));
        self
    }

    /// Adds a tool call.
    #[must_use]
    pub fn tool_call(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        self.tool_calls
            .push(ToolCall::function(id, name, arguments));
        self
    }

    /// Sets the tool call ID (for tool response messages).
    #[must_use]
    pub fn tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Sets the name field.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builds the message.
    #[must_use]
    pub fn build(self) -> Message {
        let content = if self.parts.is_empty() {
            None
        } else if self.parts.len() == 1 && self.parts[0].is_text() {
            // Optimize single text to simple string
            self.parts.into_iter().next().and_then(|p| match p {
                ContentPart::Text { text } => Some(Content::Text(text)),
                ContentPart::ImageUrl { .. } | ContentPart::InputAudio { .. } => None,
            })
        } else {
            Some(Content::Parts(self.parts))
        };

        let tool_calls = if self.tool_calls.is_empty() {
            None
        } else {
            Some(self.tool_calls)
        };

        Message {
            role: self.role,
            content,
            refusal: None,
            annotations: Vec::new(),
            tool_calls,
            tool_call_id: self.tool_call_id,
            name: self.name,
            reasoning_content: None,
            thinking_blocks: None,
        }
    }
}

/// Delta update for streaming tool calls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index of the tool call being updated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Tool call ID (may arrive in chunks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool type.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// Function details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Delta update for function call in streaming.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name (may arrive in chunks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Arguments fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Delta update for streaming message content.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Incremental text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Incremental tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

impl MessageDelta {
    /// Creates a text-only delta.
    #[inline]
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            tool_calls: None,
        }
    }

    /// Returns `true` if this delta is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.content.is_none() && self.tool_calls.is_none()
    }
}

/// Aggregator for streaming message deltas.
#[derive(Debug, Clone, Default)]
pub struct MessageAggregator {
    content: String,
    tool_calls: std::collections::BTreeMap<usize, ToolCallBuilder>,
}

#[derive(Debug, Clone, Default)]
struct ToolCallBuilder {
    id: String,
    call_type: String,
    name: String,
    arguments: String,
}

impl MessageAggregator {
    /// Creates a new aggregator.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies a delta to the aggregator.
    pub fn apply(&mut self, delta: &MessageDelta) {
        if let Some(content) = &delta.content {
            self.content.push_str(content);
        }

        if let Some(tool_calls) = &delta.tool_calls {
            for tc_delta in tool_calls {
                let index = tc_delta.index.unwrap_or(0);
                let builder = self.tool_calls.entry(index).or_default();

                if let Some(id) = &tc_delta.id {
                    builder.id.clone_from(id);
                }
                if let Some(t) = &tc_delta.call_type {
                    builder.call_type.clone_from(t);
                }
                if let Some(func) = &tc_delta.function {
                    if let Some(name) = &func.name {
                        builder.name.clone_from(name);
                    }
                    if let Some(args) = &func.arguments {
                        builder.arguments.push_str(args);
                    }
                }
            }
        }
    }

    /// Builds the final message from accumulated deltas.
    #[must_use]
    pub fn build(self) -> Message {
        let content = if self.content.is_empty() {
            None
        } else {
            Some(Content::text(self.content))
        };

        let tool_calls = if self.tool_calls.is_empty() {
            None
        } else {
            Some(
                self.tool_calls
                    .into_values()
                    .map(|b| ToolCall {
                        id: b.id,
                        call_type: if b.call_type.is_empty() {
                            "function".to_owned()
                        } else {
                            b.call_type
                        },
                        function: FunctionCall {
                            name: b.name,
                            arguments: b.arguments,
                        },
                    })
                    .collect(),
            )
        };

        Message {
            role: Role::Assistant,
            content,
            refusal: None,
            annotations: Vec::new(),
            tool_calls,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking_blocks: None,
        }
    }

    /// Returns the current accumulated text.
    #[must_use]
    pub fn current_text(&self) -> &str {
        &self.content
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    mod role {
        use super::*;

        #[test]
        fn as_str_returns_correct_values() {
            assert_eq!(Role::System.as_str(), "system");
            assert_eq!(Role::User.as_str(), "user");
            assert_eq!(Role::Assistant.as_str(), "assistant");
            assert_eq!(Role::Tool.as_str(), "tool");
            assert_eq!(Role::Developer.as_str(), "developer");
        }

        #[test]
        fn display_matches_as_str() {
            assert_eq!(Role::System.to_string(), "system");
            assert_eq!(Role::User.to_string(), "user");
            assert_eq!(Role::Assistant.to_string(), "assistant");
            assert_eq!(Role::Tool.to_string(), "tool");
            assert_eq!(Role::Developer.to_string(), "developer");
        }

        #[test]
        fn default_is_user() {
            assert_eq!(Role::default(), Role::User);
        }

        #[test]
        fn is_system_returns_true_only_for_system() {
            assert!(Role::System.is_system());
            assert!(!Role::User.is_system());
            assert!(!Role::Assistant.is_system());
        }

        #[test]
        fn is_user_returns_true_only_for_user() {
            assert!(Role::User.is_user());
            assert!(!Role::System.is_user());
            assert!(!Role::Assistant.is_user());
        }

        #[test]
        fn is_assistant_returns_true_only_for_assistant() {
            assert!(Role::Assistant.is_assistant());
            assert!(!Role::User.is_assistant());
            assert!(!Role::Tool.is_assistant());
        }

        #[test]
        fn is_tool_returns_true_only_for_tool() {
            assert!(Role::Tool.is_tool());
            assert!(!Role::User.is_tool());
            assert!(!Role::Assistant.is_tool());
        }

        #[test]
        fn serde_roundtrip() {
            let role = Role::Assistant;
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, r#""assistant""#);
            let parsed: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, role);
        }

        #[test]
        fn serde_all_variants() {
            for (role, expected) in [
                (Role::System, r#""system""#),
                (Role::User, r#""user""#),
                (Role::Assistant, r#""assistant""#),
                (Role::Tool, r#""tool""#),
                (Role::Developer, r#""developer""#),
            ] {
                assert_eq!(serde_json::to_string(&role).unwrap(), expected);
            }
        }

        #[test]
        fn copy_trait() {
            let role = Role::User;
            let copy = role;
            assert_eq!(role, copy);
        }

        #[test]
        fn hash_trait() {
            use std::collections::HashSet;
            let mut set = HashSet::new();
            set.insert(Role::User);
            set.insert(Role::Assistant);
            assert!(set.contains(&Role::User));
            assert!(!set.contains(&Role::System));
        }
    }

    mod image_mime {
        use super::*;

        #[test]
        fn as_str_returns_correct_mime_types() {
            assert_eq!(ImageMime::Jpeg.as_str(), "image/jpeg");
            assert_eq!(ImageMime::Png.as_str(), "image/png");
            assert_eq!(ImageMime::Gif.as_str(), "image/gif");
            assert_eq!(ImageMime::WebP.as_str(), "image/webp");
        }

        #[test]
        fn display_matches_as_str() {
            assert_eq!(ImageMime::Jpeg.to_string(), "image/jpeg");
            assert_eq!(ImageMime::Png.to_string(), "image/png");
        }

        #[test]
        fn default_is_jpeg() {
            assert_eq!(ImageMime::default(), ImageMime::Jpeg);
        }

        #[test]
        fn from_extension_recognizes_jpeg() {
            assert_eq!(ImageMime::from_extension("jpg"), Some(ImageMime::Jpeg));
            assert_eq!(ImageMime::from_extension("jpeg"), Some(ImageMime::Jpeg));
            assert_eq!(ImageMime::from_extension("JPG"), Some(ImageMime::Jpeg));
            assert_eq!(ImageMime::from_extension("JPEG"), Some(ImageMime::Jpeg));
        }

        #[test]
        fn from_extension_recognizes_other_formats() {
            assert_eq!(ImageMime::from_extension("png"), Some(ImageMime::Png));
            assert_eq!(ImageMime::from_extension("gif"), Some(ImageMime::Gif));
            assert_eq!(ImageMime::from_extension("webp"), Some(ImageMime::WebP));
        }

        #[test]
        fn from_extension_returns_none_for_unknown() {
            assert_eq!(ImageMime::from_extension("bmp"), None);
            assert_eq!(ImageMime::from_extension("tiff"), None);
            assert_eq!(ImageMime::from_extension(""), None);
        }

        #[test]
        fn from_bytes_detects_jpeg() {
            let jpeg_magic = [0xFF, 0xD8, 0xFF, 0xE0];
            assert_eq!(ImageMime::from_bytes(&jpeg_magic), Some(ImageMime::Jpeg));
        }

        #[test]
        fn from_bytes_detects_png() {
            let png_magic = [0x89, 0x50, 0x4E, 0x47];
            assert_eq!(ImageMime::from_bytes(&png_magic), Some(ImageMime::Png));
        }

        #[test]
        fn from_bytes_detects_gif() {
            let gif_magic = [0x47, 0x49, 0x46, 0x38];
            assert_eq!(ImageMime::from_bytes(&gif_magic), Some(ImageMime::Gif));
        }

        #[test]
        fn from_bytes_detects_webp() {
            let webp_magic = [
                0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
            ];
            assert_eq!(ImageMime::from_bytes(&webp_magic), Some(ImageMime::WebP));
        }

        #[test]
        fn from_bytes_returns_none_for_short_data() {
            assert_eq!(ImageMime::from_bytes(&[0xFF, 0xD8]), None);
            assert_eq!(ImageMime::from_bytes(&[]), None);
        }

        #[test]
        fn from_bytes_returns_none_for_unknown() {
            assert_eq!(ImageMime::from_bytes(&[0x00, 0x00, 0x00, 0x00]), None);
        }

        #[test]
        fn serde_roundtrip() {
            let mime = ImageMime::Png;
            let json = serde_json::to_string(&mime).unwrap();
            let parsed: ImageMime = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, mime);
        }

        #[test]
        fn copy_trait() {
            let mime = ImageMime::Gif;
            let copy = mime;
            assert_eq!(mime, copy);
        }
    }

    mod image_detail {
        use super::*;

        #[test]
        fn default_is_auto() {
            assert_eq!(ImageDetail::default(), ImageDetail::Auto);
        }

        #[test]
        fn serde_roundtrip() {
            for detail in [ImageDetail::Low, ImageDetail::High, ImageDetail::Auto] {
                let json = serde_json::to_string(&detail).unwrap();
                let parsed: ImageDetail = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed, detail);
            }
        }

        #[test]
        fn serde_uses_lowercase() {
            assert_eq!(
                serde_json::to_string(&ImageDetail::Low).unwrap(),
                r#""low""#
            );
            assert_eq!(
                serde_json::to_string(&ImageDetail::High).unwrap(),
                r#""high""#
            );
            assert_eq!(
                serde_json::to_string(&ImageDetail::Auto).unwrap(),
                r#""auto""#
            );
        }

        #[test]
        fn copy_trait() {
            let detail = ImageDetail::High;
            let copy = detail;
            assert_eq!(detail, copy);
        }
    }

    mod input_audio {
        use super::*;

        #[test]
        fn new_creates_input_audio() {
            let audio = InputAudio::new("base64data", AudioFormat::Mp3);
            assert_eq!(audio.data, "base64data");
            assert_eq!(audio.format, AudioFormat::Mp3);
        }

        #[test]
        fn from_bytes_encodes_to_base64() {
            let data = b"raw audio bytes";
            let audio = InputAudio::from_bytes(data, AudioFormat::Wav);
            assert_eq!(audio.format, AudioFormat::Wav);
            // Verify base64 encoding
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(&audio.data)
                .unwrap();
            assert_eq!(decoded, data);
        }

        #[test]
        fn serde_roundtrip() {
            let audio = InputAudio::new("data123", AudioFormat::Flac);
            let json = serde_json::to_string(&audio).unwrap();
            let parsed: InputAudio = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, audio);
        }

        #[test]
        fn clone_trait() {
            let audio = InputAudio::new("data", AudioFormat::Opus);
            let cloned = audio.clone();
            assert_eq!(audio, cloned);
        }
    }

    mod content_part {
        use super::*;

        #[test]
        fn text_creates_text_part() {
            let part = ContentPart::text("Hello");
            assert!(part.is_text());
            assert!(!part.is_image());
            assert!(!part.is_audio());
            assert_eq!(part.as_text(), Some("Hello"));
        }

        #[test]
        fn image_url_creates_image_part() {
            let part = ContentPart::image_url("https://example.com/img.png");
            assert!(part.is_image());
            assert!(!part.is_text());
            assert!(!part.is_audio());
            assert_eq!(part.as_text(), None);
        }

        #[test]
        fn image_url_with_detail_sets_detail() {
            let part = ContentPart::image_url_with_detail(
                "https://example.com/img.png",
                ImageDetail::High,
            );
            if let ContentPart::ImageUrl { image_url } = part {
                assert_eq!(image_url.detail, Some(ImageDetail::High));
            } else {
                panic!("Expected ImageUrl variant");
            }
        }

        #[test]
        fn image_bytes_creates_data_url() {
            let data = [0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic bytes
            let part = ContentPart::image_bytes(&data, ImageMime::Jpeg);
            if let ContentPart::ImageUrl { image_url } = part {
                assert!(image_url.url.starts_with("data:image/jpeg;base64,"));
            } else {
                panic!("Expected ImageUrl variant");
            }
        }

        #[test]
        fn image_bytes_auto_detects_mime() {
            let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            let part = ContentPart::image_bytes_auto(&png_data);
            if let ContentPart::ImageUrl { image_url } = part {
                assert!(image_url.url.starts_with("data:image/png;base64,"));
            } else {
                panic!("Expected ImageUrl variant");
            }
        }

        #[test]
        fn input_audio_creates_audio_part() {
            let part = ContentPart::input_audio("base64data", AudioFormat::Mp3);
            assert!(part.is_audio());
            assert!(!part.is_text());
            assert!(!part.is_image());
        }

        #[test]
        fn input_audio_bytes_encodes_data() {
            let data = b"audio bytes";
            let part = ContentPart::input_audio_bytes(data, AudioFormat::Wav);
            if let ContentPart::InputAudio { input_audio } = part {
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(&input_audio.data)
                    .unwrap();
                assert_eq!(decoded, data);
            } else {
                panic!("Expected InputAudio variant");
            }
        }

        #[test]
        fn serde_text_roundtrip() {
            let part = ContentPart::text("Test content");
            let json = serde_json::to_string(&part).unwrap();
            assert!(json.contains(r#""type":"text""#));
            let parsed: ContentPart = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, part);
        }

        #[test]
        fn serde_image_url_roundtrip() {
            let part = ContentPart::image_url("https://example.com/img.png");
            let json = serde_json::to_string(&part).unwrap();
            assert!(json.contains(r#""type":"image_url""#));
            let parsed: ContentPart = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, part);
        }
    }

    mod content {
        use super::*;

        #[test]
        fn text_creates_text_content() {
            let content = Content::text("Hello");
            assert_eq!(content.as_text(), Some("Hello".to_owned()));
        }

        #[test]
        fn parts_creates_multipart_content() {
            let parts = vec![ContentPart::text("Part 1"), ContentPart::text("Part 2")];
            let content = Content::parts(parts);
            assert_eq!(content.as_text(), Some("Part 1\nPart 2".to_owned()));
        }

        #[test]
        fn as_text_joins_multiple_text_parts() {
            let parts = vec![
                ContentPart::text("Line 1"),
                ContentPart::image_url("https://example.com/img.png"),
                ContentPart::text("Line 2"),
            ];
            let content = Content::Parts(parts);
            assert_eq!(content.as_text(), Some("Line 1\nLine 2".to_owned()));
        }

        #[test]
        fn as_text_returns_none_for_no_text() {
            let parts = vec![ContentPart::image_url("https://example.com/img.png")];
            let content = Content::Parts(parts);
            assert_eq!(content.as_text(), None);
        }

        #[test]
        fn has_images_returns_true_for_image_content() {
            let parts = vec![
                ContentPart::text("Look at this:"),
                ContentPart::image_url("https://example.com/img.png"),
            ];
            let content = Content::Parts(parts);
            assert!(content.has_images());
        }

        #[test]
        fn has_images_returns_false_for_text_only() {
            let content = Content::text("No images here");
            assert!(!content.has_images());
        }

        #[test]
        fn is_empty_for_empty_text() {
            let content = Content::text("");
            assert!(content.is_empty());
        }

        #[test]
        fn is_empty_for_empty_parts() {
            let content = Content::Parts(vec![]);
            assert!(content.is_empty());
        }

        #[test]
        fn is_empty_false_for_non_empty() {
            let content = Content::text("Hello");
            assert!(!content.is_empty());
        }

        #[test]
        fn default_is_empty_text() {
            let content = Content::default();
            assert!(content.is_empty());
            assert_eq!(content.as_text(), Some(String::new()));
        }

        #[test]
        fn from_string() {
            let content: Content = String::from("Hello").into();
            assert_eq!(content.as_text(), Some("Hello".to_owned()));
        }

        #[test]
        fn from_str() {
            let content: Content = "Hello".into();
            assert_eq!(content.as_text(), Some("Hello".to_owned()));
        }

        #[test]
        fn from_vec_content_part() {
            let parts = vec![ContentPart::text("Test")];
            let content: Content = parts.into();
            if let Content::Parts(p) = content {
                assert_eq!(p.len(), 1);
            } else {
                panic!("Expected Parts variant");
            }
        }

        #[test]
        fn serde_text_roundtrip() {
            let content = Content::text("Hello world");
            let json = serde_json::to_string(&content).unwrap();
            let parsed: Content = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, content);
        }

        #[test]
        fn serde_parts_roundtrip() {
            let parts = vec![ContentPart::text("Part 1"), ContentPart::text("Part 2")];
            let content = Content::Parts(parts);
            let json = serde_json::to_string(&content).unwrap();
            let parsed: Content = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, content);
        }
    }

    mod function_call {
        use super::*;

        #[test]
        fn new_creates_function_call() {
            let fc = FunctionCall::new("get_weather", r#"{"city": "Tokyo"}"#);
            assert_eq!(fc.name, "get_weather");
            assert_eq!(fc.arguments, r#"{"city": "Tokyo"}"#);
        }

        #[test]
        fn parse_arguments_success() {
            #[derive(Debug, Deserialize, PartialEq)]
            struct Args {
                city: String,
            }
            let fc = FunctionCall::new("get_weather", r#"{"city": "Tokyo"}"#);
            let args: Args = fc.parse_arguments().unwrap();
            assert_eq!(args.city, "Tokyo");
        }

        #[test]
        fn parse_arguments_error() {
            let fc = FunctionCall::new("test", "invalid json");
            let result: Result<Value, _> = fc.parse_arguments();
            assert!(result.is_err());
        }

        #[test]
        fn arguments_value_success() {
            let fc = FunctionCall::new("test", r#"{"key": "value"}"#);
            let value = fc.arguments_value();
            assert_eq!(value["key"], "value");
        }

        #[test]
        fn arguments_value_returns_null_on_error() {
            let fc = FunctionCall::new("test", "not json");
            let value = fc.arguments_value();
            assert_eq!(value, Value::Null);
        }

        #[test]
        fn serde_roundtrip() {
            let fc = FunctionCall::new("my_func", r#"{"arg": 42}"#);
            let json = serde_json::to_string(&fc).unwrap();
            let parsed: FunctionCall = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, fc);
        }
    }

    mod tool_call {
        use super::*;

        #[test]
        fn function_creates_tool_call() {
            let tc = ToolCall::function("call_123", "get_weather", r#"{"city": "NYC"}"#);
            assert_eq!(tc.id, "call_123");
            assert_eq!(tc.call_type, "function");
            assert_eq!(tc.name(), "get_weather");
            assert_eq!(tc.arguments(), r#"{"city": "NYC"}"#);
        }

        #[test]
        fn parse_arguments_delegates() {
            #[derive(Debug, Deserialize, PartialEq)]
            struct Args {
                city: String,
            }
            let tc = ToolCall::function("id", "func", r#"{"city": "Paris"}"#);
            let args: Args = tc.parse_arguments().unwrap();
            assert_eq!(args.city, "Paris");
        }

        #[test]
        fn display_format() {
            let tc = ToolCall::function("id", "calculate", r#"{"x": 1, "y": 2}"#);
            let display = tc.to_string();
            assert_eq!(display, r#"calculate({"x": 1, "y": 2})"#);
        }

        #[test]
        fn serde_roundtrip() {
            let tc = ToolCall::function("call_456", "my_tool", "{}");
            let json = serde_json::to_string(&tc).unwrap();
            let parsed: ToolCall = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.id, tc.id);
            assert_eq!(parsed.name(), tc.name());
        }

        #[test]
        fn serde_default_type() {
            // Test that type defaults to "function" when missing
            let json = r#"{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}"#;
            let tc: ToolCall = serde_json::from_str(json).unwrap();
            assert_eq!(tc.call_type, "function");
        }
    }

    mod thinking_block {
        use super::*;

        #[test]
        fn thinking_creates_block() {
            let block = ThinkingBlock::thinking("My reasoning");
            if let ThinkingBlock::Thinking {
                thinking,
                signature,
            } = &block
            {
                assert_eq!(thinking, "My reasoning");
                assert!(signature.is_none());
            } else {
                panic!("Expected Thinking variant");
            }
        }

        #[test]
        fn as_thinking_returns_content() {
            let block = ThinkingBlock::thinking("Test thought");
            assert_eq!(block.as_thinking(), Some("Test thought"));
        }

        #[test]
        fn as_thinking_returns_none_for_redacted() {
            let block = ThinkingBlock::RedactedThinking {
                data: "opaque".to_owned(),
            };
            assert_eq!(block.as_thinking(), None);
        }

        #[test]
        fn serde_thinking_roundtrip() {
            let block = ThinkingBlock::thinking("Reasoning here");
            let json = serde_json::to_string(&block).unwrap();
            assert!(json.contains(r#""type":"thinking""#));
            let parsed: ThinkingBlock = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.as_thinking(), Some("Reasoning here"));
        }

        #[test]
        fn serde_redacted_roundtrip() {
            let block = ThinkingBlock::RedactedThinking {
                data: "hidden".to_owned(),
            };
            let json = serde_json::to_string(&block).unwrap();
            assert!(json.contains(r#""type":"redacted_thinking""#));
            let parsed: ThinkingBlock = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.as_thinking(), None);
        }
    }

    mod annotation {
        use super::*;

        #[test]
        fn url_citation_serde() {
            let annotation = Annotation::UrlCitation {
                start_index: 10,
                end_index: 20,
                url: "https://example.com".to_owned(),
                title: Some("Example".to_owned()),
            };
            let json = serde_json::to_string(&annotation).unwrap();
            assert!(json.contains(r#""type":"url_citation""#));
            let parsed: Annotation = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, annotation);
        }

        #[test]
        fn file_citation_serde() {
            let annotation = Annotation::FileCitation {
                file_id: "file_123".to_owned(),
                quote: Some("relevant text".to_owned()),
            };
            let json = serde_json::to_string(&annotation).unwrap();
            assert!(json.contains(r#""type":"file_citation""#));
            let parsed: Annotation = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, annotation);
        }
    }

    mod message {
        use super::*;

        #[test]
        fn new_creates_message() {
            let msg = Message::new(Role::User, "Hello");
            assert_eq!(msg.role, Role::User);
            assert_eq!(msg.text(), Some("Hello".to_owned()));
        }

        #[test]
        fn system_creates_system_message() {
            let msg = Message::system("You are helpful.");
            assert!(msg.role.is_system());
            assert_eq!(msg.text(), Some("You are helpful.".to_owned()));
        }

        #[test]
        fn user_creates_user_message() {
            let msg = Message::user("Hello!");
            assert!(msg.role.is_user());
            assert_eq!(msg.text(), Some("Hello!".to_owned()));
        }

        #[test]
        fn assistant_creates_assistant_message() {
            let msg = Message::assistant("Hi there!");
            assert!(msg.role.is_assistant());
            assert_eq!(msg.text(), Some("Hi there!".to_owned()));
        }

        #[test]
        fn assistant_tool_calls_creates_message() {
            let tool_calls = vec![ToolCall::function("id", "func", "{}")];
            let msg = Message::assistant_tool_calls(tool_calls);
            assert!(msg.role.is_assistant());
            assert!(msg.has_tool_calls());
            assert!(msg.content.is_none());
        }

        #[test]
        fn tool_creates_tool_message() {
            let msg = Message::tool("call_123", r#"{"result": 42}"#);
            assert!(msg.role.is_tool());
            assert_eq!(msg.tool_call_id, Some("call_123".to_owned()));
            assert_eq!(msg.text(), Some(r#"{"result": 42}"#.to_owned()));
        }

        #[test]
        fn has_tool_calls_true() {
            let tool_calls = vec![ToolCall::function("id", "func", "{}")];
            let msg = Message::assistant_tool_calls(tool_calls);
            assert!(msg.has_tool_calls());
        }

        #[test]
        fn has_tool_calls_false() {
            let msg = Message::user("No tools");
            assert!(!msg.has_tool_calls());
        }

        #[test]
        fn has_images_true() {
            let msg = Message::builder(Role::User)
                .text("Look at this:")
                .image_url("https://example.com/img.png")
                .build();
            assert!(msg.has_images());
        }

        #[test]
        fn has_images_false() {
            let msg = Message::user("No images");
            assert!(!msg.has_images());
        }

        #[test]
        fn is_empty_true() {
            let msg = Message::default();
            assert!(msg.is_empty());
        }

        #[test]
        fn is_empty_false_with_content() {
            let msg = Message::user("Hello");
            assert!(!msg.is_empty());
        }

        #[test]
        fn is_empty_false_with_tool_calls() {
            let tool_calls = vec![ToolCall::function("id", "func", "{}")];
            let msg = Message::assistant_tool_calls(tool_calls);
            assert!(!msg.is_empty());
        }

        #[test]
        fn with_name_sets_name() {
            let msg = Message::user("Hello").with_name("John");
            assert_eq!(msg.name, Some("John".to_owned()));
        }

        #[test]
        fn default_is_user_role() {
            let msg = Message::default();
            assert_eq!(msg.role, Role::User);
            assert!(msg.content.is_none());
        }

        #[test]
        fn display_format() {
            let msg = Message::user("Hello world");
            let display = msg.to_string();
            assert!(display.contains("[user]"));
            assert!(display.contains("Hello world"));
        }

        #[test]
        fn display_with_tool_calls() {
            let tool_calls = vec![ToolCall::function("id", "calculate", r#"{"x": 1}"#)];
            let msg = Message::assistant_tool_calls(tool_calls);
            let display = msg.to_string();
            assert!(display.contains("[assistant]"));
            assert!(display.contains("calculate"));
        }

        #[test]
        fn serde_simple_message() {
            let msg = Message::user("Hello");
            let json = serde_json::to_string(&msg).unwrap();
            let parsed: Message = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.role, msg.role);
            assert_eq!(parsed.text(), msg.text());
        }

        #[test]
        fn serde_skips_none_fields() {
            let msg = Message::user("Hello");
            let json = serde_json::to_string(&msg).unwrap();
            assert!(!json.contains("tool_calls"));
            assert!(!json.contains("refusal"));
            assert!(!json.contains("annotations"));
        }

        #[test]
        fn serde_with_tool_calls() {
            let tool_calls = vec![ToolCall::function("call_1", "func", "{}")];
            let msg = Message::assistant_tool_calls(tool_calls);
            let json = serde_json::to_string(&msg).unwrap();
            assert!(json.contains("tool_calls"));
            let parsed: Message = serde_json::from_str(&json).unwrap();
            assert!(parsed.has_tool_calls());
        }
    }

    mod message_builder {
        use super::*;

        #[test]
        fn builder_creates_empty_message() {
            let msg = MessageBuilder::new(Role::User).build();
            assert_eq!(msg.role, Role::User);
            assert!(msg.content.is_none());
        }

        #[test]
        fn text_adds_text_content() {
            let msg = Message::builder(Role::User).text("Hello").build();
            assert_eq!(msg.text(), Some("Hello".to_owned()));
        }

        #[test]
        fn single_text_optimized_to_string() {
            let msg = Message::builder(Role::User).text("Hello").build();
            // Single text should be Content::Text, not Content::Parts
            if let Some(Content::Text(text)) = &msg.content {
                assert_eq!(text, "Hello");
            } else {
                panic!("Expected Content::Text for single text");
            }
        }

        #[test]
        fn multiple_parts_creates_parts() {
            let msg = Message::builder(Role::User)
                .text("Look at this:")
                .image_url("https://example.com/img.png")
                .build();
            if let Some(Content::Parts(parts)) = &msg.content {
                assert_eq!(parts.len(), 2);
            } else {
                panic!("Expected Content::Parts for multimodal");
            }
        }

        #[test]
        fn image_url_adds_image() {
            let msg = Message::builder(Role::User)
                .image_url("https://example.com/img.png")
                .build();
            assert!(msg.has_images());
        }

        #[test]
        fn image_url_with_detail() {
            let msg = Message::builder(Role::User)
                .image_url_with_detail("https://example.com/img.png", ImageDetail::High)
                .build();
            if let Some(Content::Parts(parts)) = &msg.content {
                if let ContentPart::ImageUrl { image_url } = &parts[0] {
                    assert_eq!(image_url.detail, Some(ImageDetail::High));
                } else {
                    panic!("Expected ImageUrl");
                }
            }
        }

        #[test]
        fn image_bytes_adds_data_url() {
            let data = [0xFF, 0xD8, 0xFF, 0xE0];
            let msg = Message::builder(Role::User)
                .image_bytes(&data, ImageMime::Jpeg)
                .build();
            assert!(msg.has_images());
        }

        #[test]
        fn tool_call_adds_tool_calls() {
            let msg = Message::builder(Role::Assistant)
                .tool_call("call_1", "get_weather", r#"{"city": "Tokyo"}"#)
                .tool_call("call_2", "get_time", r#"{"timezone": "JST"}"#)
                .build();
            assert!(msg.has_tool_calls());
            assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 2);
        }

        #[test]
        fn tool_call_id_sets_id() {
            let msg = Message::builder(Role::Tool)
                .text("Result")
                .tool_call_id("call_123")
                .build();
            assert_eq!(msg.tool_call_id, Some("call_123".to_owned()));
        }

        #[test]
        fn name_sets_name() {
            let msg = Message::builder(Role::User)
                .text("Hello")
                .name("Alice")
                .build();
            assert_eq!(msg.name, Some("Alice".to_owned()));
        }

        #[test]
        fn clone_trait() {
            let builder = Message::builder(Role::User).text("Hello");
            let cloned = builder.clone();
            let msg1 = builder.build();
            let msg2 = cloned.build();
            assert_eq!(msg1.text(), msg2.text());
        }
    }

    mod message_delta {
        use super::*;

        #[test]
        fn text_creates_text_delta() {
            let delta = MessageDelta::text("Hello");
            assert_eq!(delta.content, Some("Hello".to_owned()));
            assert!(delta.tool_calls.is_none());
        }

        #[test]
        fn is_empty_true() {
            let delta = MessageDelta::default();
            assert!(delta.is_empty());
        }

        #[test]
        fn is_empty_false_with_content() {
            let delta = MessageDelta::text("Hi");
            assert!(!delta.is_empty());
        }

        #[test]
        fn is_empty_false_with_tool_calls() {
            let delta = MessageDelta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta::default()]),
            };
            assert!(!delta.is_empty());
        }

        #[test]
        fn serde_roundtrip() {
            let delta = MessageDelta::text("Test");
            let json = serde_json::to_string(&delta).unwrap();
            let parsed: MessageDelta = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.content, delta.content);
        }
    }

    mod message_aggregator {
        use super::*;

        #[test]
        fn new_creates_empty_aggregator() {
            let agg = MessageAggregator::new();
            assert_eq!(agg.current_text(), "");
        }

        #[test]
        fn apply_text_delta() {
            let mut agg = MessageAggregator::new();
            agg.apply(&MessageDelta::text("Hello"));
            agg.apply(&MessageDelta::text(" World"));
            assert_eq!(agg.current_text(), "Hello World");
        }

        #[test]
        fn build_creates_message() {
            let mut agg = MessageAggregator::new();
            agg.apply(&MessageDelta::text("Hello"));
            let msg = agg.build();
            assert_eq!(msg.role, Role::Assistant);
            assert_eq!(msg.text(), Some("Hello".to_owned()));
        }

        #[test]
        fn build_empty_content_is_none() {
            let agg = MessageAggregator::new();
            let msg = agg.build();
            assert!(msg.content.is_none());
        }

        #[test]
        fn apply_tool_call_delta() {
            let mut agg = MessageAggregator::new();

            // First delta with id and function name
            agg.apply(&MessageDelta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: Some(0),
                    id: Some("call_123".to_owned()),
                    call_type: Some("function".to_owned()),
                    function: Some(FunctionCallDelta {
                        name: Some("get_weather".to_owned()),
                        arguments: Some(r#"{"city":"#.to_owned()),
                    }),
                }]),
            });

            // Second delta with arguments continuation
            agg.apply(&MessageDelta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: Some(0),
                    id: None,
                    call_type: None,
                    function: Some(FunctionCallDelta {
                        name: None,
                        arguments: Some(r#""Tokyo"}"#.to_owned()),
                    }),
                }]),
            });

            let msg = agg.build();
            assert!(msg.has_tool_calls());
            let tool_calls = msg.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].id, "call_123");
            assert_eq!(tool_calls[0].name(), "get_weather");
            assert_eq!(tool_calls[0].arguments(), r#"{"city":"Tokyo"}"#);
        }

        #[test]
        fn apply_multiple_tool_calls() {
            let mut agg = MessageAggregator::new();

            agg.apply(&MessageDelta {
                content: None,
                tool_calls: Some(vec![
                    ToolCallDelta {
                        index: Some(0),
                        id: Some("call_1".to_owned()),
                        call_type: None,
                        function: Some(FunctionCallDelta {
                            name: Some("func1".to_owned()),
                            arguments: Some("{}".to_owned()),
                        }),
                    },
                    ToolCallDelta {
                        index: Some(1),
                        id: Some("call_2".to_owned()),
                        call_type: None,
                        function: Some(FunctionCallDelta {
                            name: Some("func2".to_owned()),
                            arguments: Some("{}".to_owned()),
                        }),
                    },
                ]),
            });

            let msg = agg.build();
            let tool_calls = msg.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 2);
        }

        #[test]
        fn default_trait() {
            let agg = MessageAggregator::default();
            assert_eq!(agg.current_text(), "");
        }

        #[test]
        fn clone_trait() {
            let mut agg = MessageAggregator::new();
            agg.apply(&MessageDelta::text("Hello"));
            let cloned = agg.clone();
            assert_eq!(agg.current_text(), cloned.current_text());
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn full_conversation_flow() {
            let messages = [
                Message::system("You are a helpful assistant."),
                Message::user("What's the weather in Tokyo?"),
                Message::assistant_tool_calls(vec![ToolCall::function(
                    "call_1",
                    "get_weather",
                    r#"{"city": "Tokyo"}"#,
                )]),
                Message::tool("call_1", r#"{"temp": 20, "condition": "sunny"}"#),
                Message::assistant("The weather in Tokyo is sunny with 20°C."),
            ];

            assert_eq!(messages.len(), 5);
            assert!(messages[0].role.is_system());
            assert!(messages[1].role.is_user());
            assert!(messages[2].has_tool_calls());
            assert!(messages[3].role.is_tool());
            assert!(messages[4].role.is_assistant());
        }

        #[test]
        fn multimodal_message_json() {
            let msg = Message::builder(Role::User)
                .text("What's in this image?")
                .image_url("https://example.com/cat.jpg")
                .build();

            let json = serde_json::to_string_pretty(&msg).unwrap();
            assert!(json.contains("user"));
            assert!(json.contains("What's in this image?"));
            assert!(json.contains("https://example.com/cat.jpg"));

            let parsed: Message = serde_json::from_str(&json).unwrap();
            assert!(parsed.has_images());
        }

        #[test]
        fn streaming_aggregation() {
            let deltas = [
                MessageDelta::text("The "),
                MessageDelta::text("weather "),
                MessageDelta::text("is "),
                MessageDelta::text("sunny."),
            ];

            let mut agg = MessageAggregator::new();
            for delta in &deltas {
                agg.apply(delta);
            }

            let msg = agg.build();
            assert_eq!(msg.text(), Some("The weather is sunny.".to_owned()));
        }

        #[test]
        fn message_with_all_optional_fields() {
            let msg = Message {
                role: Role::Assistant,
                content: Some(Content::text("Response")),
                refusal: Some("Cannot process".to_owned()),
                annotations: vec![Annotation::UrlCitation {
                    start_index: 0,
                    end_index: 8,
                    url: "https://example.com".to_owned(),
                    title: None,
                }],
                tool_calls: Some(vec![ToolCall::function("id", "func", "{}")]),
                tool_call_id: None,
                name: Some("assistant_name".to_owned()),
                reasoning_content: Some("My reasoning".to_owned()),
                thinking_blocks: Some(vec![ThinkingBlock::thinking("Thinking...")]),
            };

            let json = serde_json::to_string(&msg).unwrap();
            let parsed: Message = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.refusal, msg.refusal);
            assert_eq!(parsed.annotations.len(), 1);
            assert!(parsed.has_tool_calls());
            assert_eq!(parsed.name, msg.name);
            assert_eq!(parsed.reasoning_content, msg.reasoning_content);
            assert!(parsed.thinking_blocks.is_some());
        }
    }
}
