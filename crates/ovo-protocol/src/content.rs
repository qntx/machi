//! Model- and host-facing content blocks (text / image).

use serde::{Deserialize, Serialize};

/// MIME type for inline images.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImageMime(pub String);

impl ImageMime {
    /// Common PNG MIME.
    #[must_use]
    pub fn png() -> Self {
        Self("image/png".into())
    }

    /// Common JPEG MIME.
    #[must_use]
    pub fn jpeg() -> Self {
        Self("image/jpeg".into())
    }
}

/// Inline image payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageBlock {
    /// MIME type (e.g. `image/png`).
    pub mime_type: ImageMime,
    /// Base64-encoded bytes (or host-specific data URI payload without prefix).
    pub data: String,
    /// Optional stable media id for follow-up tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_id: Option<String>,
    /// Optional filename.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// Optional filesystem path when the host materializes the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Rich content unit for tool progress, tool results, and multimodal messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// Text body.
        text: String,
    },
    /// Inline image.
    Image(ImageBlock),
}

impl ContentBlock {
    /// Text block helper.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Flatten text blocks only (images become empty contribution).
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text.as_str()),
            Self::Image(_) => None,
        }
    }
}

/// Join text blocks with newlines; skip non-text.
#[must_use]
pub fn join_text_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(ContentBlock::as_text)
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip_text() {
        let block = ContentBlock::text("hello");
        let v = serde_json::to_value(&block).expect("ser");
        let back: ContentBlock = serde_json::from_value(v).expect("de");
        assert_eq!(back.as_text(), Some("hello"));
    }
}
