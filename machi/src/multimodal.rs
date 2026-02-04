//! Agent types for multimodal data (images, audio).
//!
//! This module provides types for handling multimodal content in agent interactions,
//! following the patterns established by smolagents while leveraging Rust's type system.
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::agent_types::{AgentImage, ImageFormat};
//!
//! // Create image from bytes
//! let image = AgentImage::from_bytes(png_bytes, ImageFormat::Png);
//!
//! // Convert to base64 for API calls
//! let base64 = image.to_base64();
//!
//! // Convert to data URL for vision models
//! let data_url = image.to_data_url();
//! ```

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::Path;

/// Supported image formats for multimodal content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ImageFormat {
    /// PNG format (default).
    #[default]
    Png,
    /// JPEG format.
    Jpeg,
    /// GIF format.
    Gif,
    /// WebP format.
    Webp,
    /// SVG format.
    Svg,
}

impl ImageFormat {
    /// Get the MIME type for this format.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
            Self::Svg => "image/svg+xml",
        }
    }

    /// Get the file extension for this format.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Gif => "gif",
            Self::Webp => "webp",
            Self::Svg => "svg",
        }
    }

    /// Detect format from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            "svg" => Some(Self::Svg),
            _ => None,
        }
    }

    /// Detect format from magic bytes (file signature).
    #[must_use]
    pub fn from_magic_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        match bytes {
            [0x89, 0x50, 0x4E, 0x47, ..] => Some(Self::Png),
            [0xFF, 0xD8, 0xFF, ..] => Some(Self::Jpeg),
            [0x47, 0x49, 0x46, 0x38, ..] => Some(Self::Gif),
            [0x52, 0x49, 0x46, 0x46, ..] if bytes.len() >= 12 && &bytes[8..12] == b"WEBP" => {
                Some(Self::Webp)
            }
            _ if bytes.starts_with(b"<svg") || bytes.starts_with(b"<?xml") => Some(Self::Svg),
            _ => None,
        }
    }
}

/// Internal representation of image data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "lowercase")]
enum ImageSource {
    /// Raw bytes data.
    Bytes {
        #[serde(with = "base64_serde")]
        data: Vec<u8>,
    },
    /// Base64 encoded string (already encoded).
    Base64 { data: String },
    /// URL reference.
    Url { url: String },
    /// File path reference.
    Path { path: String },
}

/// Image type for multimodal agent interactions.
///
/// `AgentImage` provides a unified interface for handling images from various sources
/// (bytes, files, URLs) and converting them to formats suitable for LLM APIs.
///
/// # Examples
///
/// ```rust,ignore
/// // From raw bytes
/// let img = AgentImage::from_bytes(bytes, ImageFormat::Png);
///
/// // From URL (lazy, no network request)
/// let img = AgentImage::from_url("https://example.com/image.png");
///
/// // Convert to data URL for vision APIs
/// let data_url = img.to_data_url();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentImage {
    source: ImageSource,
    format: ImageFormat,
}

impl AgentImage {
    /// Create an image from raw bytes.
    ///
    /// The format can be auto-detected if `None` is provided.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>, format: impl Into<Option<ImageFormat>>) -> Self {
        let format = format
            .into()
            .or_else(|| ImageFormat::from_magic_bytes(&bytes))
            .unwrap_or_default();
        Self {
            source: ImageSource::Bytes { data: bytes },
            format,
        }
    }

    /// Create an image from a base64 encoded string.
    #[must_use]
    pub fn from_base64(base64: impl Into<String>, format: ImageFormat) -> Self {
        Self {
            source: ImageSource::Base64 {
                data: base64.into(),
            },
            format,
        }
    }

    /// Create an image from a URL (lazy, no network request).
    ///
    /// The format is auto-detected from the URL extension if possible.
    #[must_use]
    pub fn from_url(url: impl Into<String>) -> Self {
        let url = url.into();
        let format = url
            .rsplit('.')
            .next()
            .and_then(|ext| ext.split('?').next())
            .and_then(ImageFormat::from_extension)
            .unwrap_or_default();
        Self {
            source: ImageSource::Url { url },
            format,
        }
    }

    /// Create an image from a file path (lazy, no file read).
    ///
    /// The format is auto-detected from the file extension if possible.
    #[must_use]
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let format = path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .and_then(ImageFormat::from_extension)
            .unwrap_or_default();
        Self {
            source: ImageSource::Path { path: path_str },
            format,
        }
    }

    /// Load image from file path asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub async fn load_from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path).await?;
        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(ImageFormat::from_extension)
            .or_else(|| ImageFormat::from_magic_bytes(&bytes))
            .unwrap_or_default();
        Ok(Self::from_bytes(bytes, format))
    }

    /// Get the image format.
    #[must_use]
    pub const fn format(&self) -> ImageFormat {
        self.format
    }

    /// Check if this image has embedded data (bytes or base64).
    #[must_use]
    pub const fn has_data(&self) -> bool {
        matches!(
            self.source,
            ImageSource::Bytes { .. } | ImageSource::Base64 { .. }
        )
    }

    /// Check if this image is a URL reference.
    #[must_use]
    pub const fn is_url(&self) -> bool {
        matches!(self.source, ImageSource::Url { .. })
    }

    /// Get the URL if this image is a URL reference.
    #[must_use]
    pub fn as_url(&self) -> Option<&str> {
        match &self.source {
            ImageSource::Url { url } => Some(url),
            _ => None,
        }
    }

    /// Convert to base64 encoded string.
    ///
    /// Returns `None` if the image is a URL or unloaded path.
    #[must_use]
    pub fn to_base64(&self) -> Option<Cow<'_, str>> {
        match &self.source {
            ImageSource::Bytes { data } => Some(Cow::Owned(BASE64.encode(data))),
            ImageSource::Base64 { data } => Some(Cow::Borrowed(data)),
            ImageSource::Url { .. } | ImageSource::Path { .. } => None,
        }
    }

    /// Convert to data URL format (data:image/png;base64,...).
    ///
    /// Returns `None` if the image is a URL or unloaded path.
    #[must_use]
    pub fn to_data_url(&self) -> Option<String> {
        self.to_base64()
            .map(|b64| format!("data:{};base64,{}", self.format.mime_type(), b64))
    }

    /// Get the URL for API calls.
    ///
    /// Returns the URL directly if it's a URL reference, or the data URL if data is available.
    #[must_use]
    pub fn to_api_url(&self) -> Option<String> {
        match &self.source {
            ImageSource::Url { url } => Some(url.clone()),
            _ => self.to_data_url(),
        }
    }

    /// Get raw bytes if available.
    ///
    /// Returns `None` if the image is a URL, unloaded path, or base64.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.source {
            ImageSource::Bytes { data } => Some(data),
            _ => None,
        }
    }

    /// Try to get raw bytes, decoding base64 if necessary.
    ///
    /// Returns `None` if the image is a URL or unloaded path.
    #[must_use]
    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        match &self.source {
            ImageSource::Bytes { data } => Some(data.clone()),
            ImageSource::Base64 { data } => BASE64.decode(data).ok(),
            ImageSource::Url { .. } | ImageSource::Path { .. } => None,
        }
    }
}

impl std::fmt::Display for AgentImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            ImageSource::Bytes { data } => {
                write!(
                    f,
                    "[Image: {} bytes, {}]",
                    data.len(),
                    self.format.mime_type()
                )
            }
            ImageSource::Base64 { data } => {
                write!(
                    f,
                    "[Image: ~{} bytes, {}]",
                    data.len() * 3 / 4,
                    self.format.mime_type()
                )
            }
            ImageSource::Url { url } => write!(f, "[Image: {url}]"),
            ImageSource::Path { path } => write!(f, "[Image: {path}]"),
        }
    }
}

/// Audio format for multimodal content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum AudioFormat {
    /// WAV format (default).
    #[default]
    Wav,
    /// MP3 format.
    Mp3,
    /// OGG format.
    Ogg,
    /// FLAC format.
    Flac,
    /// WebM format.
    Webm,
}

impl AudioFormat {
    /// Get the MIME type for this format.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Ogg => "audio/ogg",
            Self::Flac => "audio/flac",
            Self::Webm => "audio/webm",
        }
    }

    /// Get the file extension for this format.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Ogg => "ogg",
            Self::Flac => "flac",
            Self::Webm => "webm",
        }
    }

    /// Detect format from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Some(Self::Wav),
            "mp3" => Some(Self::Mp3),
            "ogg" => Some(Self::Ogg),
            "flac" => Some(Self::Flac),
            "webm" => Some(Self::Webm),
            _ => None,
        }
    }
}

/// Internal representation of audio data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "lowercase")]
enum AudioSource {
    /// Raw bytes data.
    Bytes {
        #[serde(with = "base64_serde")]
        data: Vec<u8>,
    },
    /// Base64 encoded string.
    Base64 { data: String },
    /// File path reference.
    Path { path: String },
}

/// Audio type for multimodal agent interactions.
///
/// `AgentAudio` provides a unified interface for handling audio content
/// from various sources (bytes, files) and converting them to formats
/// suitable for speech-to-text or audio processing APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAudio {
    source: AudioSource,
    format: AudioFormat,
    /// Sample rate in Hz.
    sample_rate: u32,
}

impl AgentAudio {
    /// Default sample rate (16kHz, common for speech).
    pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

    /// Create audio from raw bytes.
    #[must_use]
    pub const fn from_bytes(bytes: Vec<u8>, format: AudioFormat, sample_rate: u32) -> Self {
        Self {
            source: AudioSource::Bytes { data: bytes },
            format,
            sample_rate,
        }
    }

    /// Create audio from raw bytes with default sample rate.
    #[must_use]
    pub const fn from_bytes_default(bytes: Vec<u8>, format: AudioFormat) -> Self {
        Self::from_bytes(bytes, format, Self::DEFAULT_SAMPLE_RATE)
    }

    /// Create audio from base64 encoded string.
    #[must_use]
    pub fn from_base64(base64: impl Into<String>, format: AudioFormat, sample_rate: u32) -> Self {
        Self {
            source: AudioSource::Base64 {
                data: base64.into(),
            },
            format,
            sample_rate,
        }
    }

    /// Create audio from file path (lazy, no file read).
    #[must_use]
    pub fn from_path(path: impl AsRef<Path>, sample_rate: u32) -> Self {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        let format = path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .and_then(AudioFormat::from_extension)
            .unwrap_or_default();
        Self {
            source: AudioSource::Path { path: path_str },
            format,
            sample_rate,
        }
    }

    /// Load audio from file path asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub async fn load_from_path(path: impl AsRef<Path>, sample_rate: u32) -> std::io::Result<Self> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path).await?;
        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(AudioFormat::from_extension)
            .unwrap_or_default();
        Ok(Self::from_bytes(bytes, format, sample_rate))
    }

    /// Get the audio format.
    #[must_use]
    pub const fn format(&self) -> AudioFormat {
        self.format
    }

    /// Get the sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Convert to base64 encoded string.
    #[must_use]
    pub fn to_base64(&self) -> Option<Cow<'_, str>> {
        match &self.source {
            AudioSource::Bytes { data } => Some(Cow::Owned(BASE64.encode(data))),
            AudioSource::Base64 { data } => Some(Cow::Borrowed(data)),
            AudioSource::Path { .. } => None,
        }
    }

    /// Convert to data URL format.
    #[must_use]
    pub fn to_data_url(&self) -> Option<String> {
        self.to_base64()
            .map(|b64| format!("data:{};base64,{}", self.format.mime_type(), b64))
    }

    /// Get raw bytes if available.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.source {
            AudioSource::Bytes { data } => Some(data),
            _ => None,
        }
    }
}

impl std::fmt::Display for AgentAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            AudioSource::Bytes { data } => {
                write!(
                    f,
                    "[Audio: {} bytes, {}, {}Hz]",
                    data.len(),
                    self.format.mime_type(),
                    self.sample_rate
                )
            }
            AudioSource::Base64 { data } => {
                write!(
                    f,
                    "[Audio: ~{} bytes, {}, {}Hz]",
                    data.len() * 3 / 4,
                    self.format.mime_type(),
                    self.sample_rate
                )
            }
            AudioSource::Path { path } => {
                write!(f, "[Audio: {}, {}Hz]", path, self.sample_rate)
            }
        }
    }
}

/// Union type for different agent output types.
///
/// This enum represents the different types of content that can be
/// returned by tools or agents in a multimodal context.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum AgentOutput {
    /// Text content.
    Text {
        /// The text value.
        value: String,
    },
    /// Image content.
    Image {
        /// The image value.
        value: AgentImage,
    },
    /// Audio content.
    Audio {
        /// The audio value.
        value: AgentAudio,
    },
    /// Structured JSON object.
    Object {
        /// The JSON object value.
        value: serde_json::Value,
    },
}

impl AgentOutput {
    /// Create a text output.
    #[must_use]
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text {
            value: value.into(),
        }
    }

    /// Create an image output.
    #[must_use]
    pub const fn image(value: AgentImage) -> Self {
        Self::Image { value }
    }

    /// Create an audio output.
    #[must_use]
    pub const fn audio(value: AgentAudio) -> Self {
        Self::Audio { value }
    }

    /// Create an object output.
    #[must_use]
    pub const fn object(value: serde_json::Value) -> Self {
        Self::Object { value }
    }

    /// Get the type name as a string.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Image { .. } => "image",
            Self::Audio { .. } => "audio",
            Self::Object { .. } => "object",
        }
    }

    /// Try to get as text.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { value } => Some(value),
            _ => None,
        }
    }

    /// Try to get as image.
    #[must_use]
    pub const fn as_image(&self) -> Option<&AgentImage> {
        match self {
            Self::Image { value } => Some(value),
            _ => None,
        }
    }

    /// Try to get as audio.
    #[must_use]
    pub const fn as_audio(&self) -> Option<&AgentAudio> {
        match self {
            Self::Audio { value } => Some(value),
            _ => None,
        }
    }
}

impl std::fmt::Display for AgentOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text { value } => write!(f, "{value}"),
            Self::Image { value } => write!(f, "{value}"),
            Self::Audio { value } => write!(f, "{value}"),
            Self::Object { value } => write!(f, "{value}"),
        }
    }
}

impl From<String> for AgentOutput {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

impl From<&str> for AgentOutput {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

impl From<AgentImage> for AgentOutput {
    fn from(value: AgentImage) -> Self {
        Self::image(value)
    }
}

impl From<AgentAudio> for AgentOutput {
    fn from(value: AgentAudio) -> Self {
        Self::audio(value)
    }
}

/// Custom serde module for base64 encoding of byte vectors.
mod base64_serde {
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&BASE64.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        BASE64.decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_detection() {
        // PNG magic bytes
        let png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(ImageFormat::from_magic_bytes(&png), Some(ImageFormat::Png));

        // JPEG magic bytes
        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(
            ImageFormat::from_magic_bytes(&jpeg),
            Some(ImageFormat::Jpeg)
        );

        // Extension detection
        assert_eq!(ImageFormat::from_extension("png"), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::from_extension("JPG"), Some(ImageFormat::Jpeg));
    }

    #[test]
    fn test_agent_image_from_bytes() {
        let bytes = vec![0x89, 0x50, 0x4E, 0x47, 0x00, 0x00];
        let img = AgentImage::from_bytes(bytes.clone(), None);

        assert_eq!(img.format(), ImageFormat::Png);
        assert!(img.has_data());
        assert!(!img.is_url());
        assert_eq!(img.as_bytes(), Some(bytes.as_slice()));
    }

    #[test]
    fn test_agent_image_from_url() {
        let img = AgentImage::from_url("https://example.com/image.png");

        assert_eq!(img.format(), ImageFormat::Png);
        assert!(!img.has_data());
        assert!(img.is_url());
        assert_eq!(img.as_url(), Some("https://example.com/image.png"));
    }

    #[test]
    fn test_agent_image_to_base64() {
        let bytes = vec![1, 2, 3, 4, 5];
        let img = AgentImage::from_bytes(bytes, ImageFormat::Png);

        let base64 = img.to_base64().expect("should have base64 data");
        assert_eq!(base64.as_ref(), "AQIDBAU=");
    }

    #[test]
    fn test_agent_image_to_data_url() {
        let bytes = vec![1, 2, 3, 4, 5];
        let img = AgentImage::from_bytes(bytes, ImageFormat::Png);

        let data_url = img.to_data_url().expect("should have data URL");
        assert!(data_url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_agent_audio() {
        let audio = AgentAudio::from_bytes(vec![1, 2, 3], AudioFormat::Wav, 16000);

        assert_eq!(audio.format(), AudioFormat::Wav);
        assert_eq!(audio.sample_rate(), 16000);
        assert!(audio.to_base64().is_some());
    }

    #[test]
    fn test_agent_output() {
        let text = AgentOutput::text("hello");
        assert_eq!(text.type_name(), "text");
        assert_eq!(text.as_text(), Some("hello"));

        let img = AgentOutput::image(AgentImage::from_url("http://test.com/img.png"));
        assert_eq!(img.type_name(), "image");
        assert!(img.as_image().is_some());
    }
}
