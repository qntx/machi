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
    /// Developer message for `OpenAI` o1/o3 models.
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
    /// `WebP` image.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the arguments string is not valid JSON or cannot be deserialized into `T`.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the arguments string is not valid JSON or cannot be deserialized into `T`.
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

    /// Reasoning content from `OpenAI` o1/o3 models.
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
