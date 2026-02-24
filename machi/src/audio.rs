//! Audio processing types and provider traits.
//!
//! This module provides types and traits for audio-related operations:
//! - **Text-to-Speech (TTS)** — converting text into audio
//! - **Speech-to-Text (STT)** — converting audio into text (transcription)
//!
//! # Examples
//!
//! ```rust
//! use machi::audio::{AudioFormat, SpeechRequest, TranscriptionRequest};
//!
//! // Text-to-Speech request
//! let request = SpeechRequest::new("tts-1", "Hello, world!", "alloy")
//!     .format(AudioFormat::Mp3)
//!     .speed(1.0);
//!
//! // Speech-to-Text request
//! let request = TranscriptionRequest::new("whisper-1", vec![0u8; 16])
//!     .format(AudioFormat::Mp3)
//!     .language("en");
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Audio format for input/output operations.
///
/// Note: Not all formats are supported by all providers or operations.
/// - **TTS (`OpenAI`)**: mp3, opus, aac, flac, wav, pcm
/// - **STT (`OpenAI`)**: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// WAV format
    #[default]
    Wav,
    /// MP3 format
    Mp3,
    /// FLAC format
    Flac,
    /// OGG format
    Ogg,
    /// `WebM` format
    WebM,
    /// M4A format
    M4a,
    /// Opus format
    Opus,
    /// AAC format
    Aac,
    /// PCM format (raw audio)
    Pcm,
}

impl AudioFormat {
    /// Get the file extension for this format.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
            Self::WebM => "webm",
            Self::M4a => "m4a",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Pcm => "pcm",
        }
    }

    /// Get the MIME type for this format.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Flac => "audio/flac",
            Self::Ogg => "audio/ogg",
            Self::WebM => "audio/webm",
            Self::M4a => "audio/m4a",
            Self::Opus => "audio/opus",
            Self::Aac => "audio/aac",
            Self::Pcm => "audio/pcm",
        }
    }

    /// Get the format string for API requests.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        self.extension()
    }

    /// Detect format from file extension.
    #[must_use]
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "wav" => Some(Self::Wav),
            "mp3" => Some(Self::Mp3),
            "flac" => Some(Self::Flac),
            "ogg" => Some(Self::Ogg),
            "webm" => Some(Self::WebM),
            "m4a" => Some(Self::M4a),
            "opus" => Some(Self::Opus),
            "aac" => Some(Self::Aac),
            "pcm" => Some(Self::Pcm),
            _ => None,
        }
    }
}

/// Voice options for text-to-speech.
///
/// `OpenAI` built-in voices: `alloy`, `ash`, `ballad`, `coral`, `echo`,
/// `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse`, `marin`, `cedar`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Voice {
    /// Voice identifier (e.g., "alloy", "echo", "nova").
    pub id: String,
    /// Optional voice description (not sent to API, for display only).
    #[serde(skip)]
    pub description: Option<String>,
}

impl Voice {
    /// Create a new voice with the given ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: None,
        }
    }

    /// Set the voice description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

impl<S: Into<String>> From<S> for Voice {
    fn from(s: S) -> Self {
        Self::new(s)
    }
}

/// Request for generating speech from text.
///
/// # Models
/// - `tts-1`: Standard quality, lower latency
/// - `tts-1-hd`: Higher quality, higher latency
/// - `gpt-4o-mini-tts`: Supports instructions for voice control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// Model to use for TTS (e.g., "tts-1", "tts-1-hd", "gpt-4o-mini-tts").
    pub model: String,
    /// Text to convert to speech (max 4096 characters).
    pub input: String,
    /// Voice to use.
    pub voice: Voice,
    /// Output audio format.
    pub response_format: AudioFormat,
    /// Speaking speed (0.25 to 4.0, default 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Instructions for voice control (gpt-4o-mini-tts only).
    /// Example: "Speak in a cheerful and friendly tone."
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl SpeechRequest {
    /// Create a new speech request.
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        input: impl Into<String>,
        voice: impl Into<Voice>,
    ) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            voice: voice.into(),
            response_format: AudioFormat::Mp3,
            speed: None,
            instructions: None,
        }
    }

    /// Set the output format.
    #[must_use]
    pub const fn format(mut self, format: AudioFormat) -> Self {
        self.response_format = format;
        self
    }

    /// Set the speaking speed (0.25 to 4.0).
    #[must_use]
    pub const fn speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Set voice control instructions (gpt-4o-mini-tts only).
    ///
    /// Use this to control the tone, style, or emotion of the generated speech.
    /// Note: This does not work with `tts-1` or `tts-1-hd` models.
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Response from a speech synthesis request.
#[derive(Debug, Clone)]
pub struct SpeechResponse {
    /// The generated audio data.
    pub audio: Vec<u8>,
    /// The format of the audio data.
    pub format: AudioFormat,
}

impl SpeechResponse {
    /// Create a new speech response.
    #[must_use]
    pub const fn new(audio: Vec<u8>, format: AudioFormat) -> Self {
        Self { audio, format }
    }

    /// Save the audio to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, &self.audio)
    }

    /// Get the suggested file extension.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        self.format.extension()
    }
}

/// Output format for transcription responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionResponseFormat {
    /// JSON format with text only.
    #[default]
    Json,
    /// Plain text format.
    Text,
    /// SRT subtitle format.
    Srt,
    /// VTT subtitle format.
    Vtt,
    /// Verbose JSON with timestamps and metadata.
    VerboseJson,
}

impl TranscriptionResponseFormat {
    /// Get the format string for API requests.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Text => "text",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
            Self::VerboseJson => "verbose_json",
        }
    }
}

/// Timestamp granularity options for transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimestampGranularity {
    /// Word-level timestamps.
    Word,
    /// Segment-level timestamps.
    Segment,
}

impl TimestampGranularity {
    /// Get the granularity string for API requests.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Word => "word",
            Self::Segment => "segment",
        }
    }
}

/// Request for transcribing audio to text.
///
/// # Models
/// - `whisper-1`: `OpenAI` Whisper V2
/// - `gpt-4o-transcribe`: GPT-4o based transcription
/// - `gpt-4o-mini-transcribe`: Smaller, faster GPT-4o transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    /// Model to use for transcription (e.g., "whisper-1", "gpt-4o-transcribe").
    pub model: String,
    /// Audio data to transcribe.
    #[serde(skip)]
    pub audio: Vec<u8>,
    /// Audio format.
    #[serde(skip)]
    pub format: AudioFormat,
    /// Optional language hint (ISO 639-1 code, e.g., "en", "zh").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional prompt to guide the transcription.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Output format for the transcription.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<TranscriptionResponseFormat>,
    /// Sampling temperature (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Timestamp granularities to include (requires `verbose_json` format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,
}

impl TranscriptionRequest {
    /// Create a new transcription request.
    #[must_use]
    pub fn new(model: impl Into<String>, audio: Vec<u8>) -> Self {
        Self {
            model: model.into(),
            audio,
            format: AudioFormat::default(),
            language: None,
            prompt: None,
            response_format: None,
            temperature: None,
            timestamp_granularities: None,
        }
    }

    /// Set the audio format.
    #[must_use]
    pub const fn format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the language hint (ISO 639-1 code).
    #[must_use]
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set the prompt to guide transcription.
    #[must_use]
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the response format.
    #[must_use]
    pub const fn response_format(mut self, format: TranscriptionResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set the temperature (0.0 to 1.0).
    #[must_use]
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Enable word-level timestamps (requires `verbose_json` format).
    #[must_use]
    pub fn with_word_timestamps(mut self) -> Self {
        let mut granularities = self.timestamp_granularities.unwrap_or_default();
        if !granularities.contains(&TimestampGranularity::Word) {
            granularities.push(TimestampGranularity::Word);
        }
        self.timestamp_granularities = Some(granularities);
        self.response_format = Some(TranscriptionResponseFormat::VerboseJson);
        self
    }

    /// Enable segment-level timestamps (requires `verbose_json` format).
    #[must_use]
    pub fn with_segment_timestamps(mut self) -> Self {
        let mut granularities = self.timestamp_granularities.unwrap_or_default();
        if !granularities.contains(&TimestampGranularity::Segment) {
            granularities.push(TimestampGranularity::Segment);
        }
        self.timestamp_granularities = Some(granularities);
        self.response_format = Some(TranscriptionResponseFormat::VerboseJson);
        self
    }
}

/// A word with timestamp information from transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionWord {
    /// The transcribed word.
    pub word: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
}

/// A segment of transcribed text with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment ID.
    pub id: usize,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// The transcribed text for this segment.
    pub text: String,
}

/// Response from a transcription request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
    /// Optional detected language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional duration of the audio in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
    /// Word-level timestamps (when requested with `verbose_json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionWord>>,
    /// Segment-level timestamps (when requested with `verbose_json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,
}

impl TranscriptionResponse {
    /// Create a new transcription response with just text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    /// Set the detected language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the audio duration.
    #[must_use]
    pub const fn with_duration(mut self, duration: f32) -> Self {
        self.duration = Some(duration);
        self
    }
}

/// Trait for providers that support text-to-speech synthesis.
///
/// This trait defines the interface for converting text into audio.
/// Providers like `OpenAI` implement this trait.
#[async_trait]
pub trait TextToSpeechProvider: Send + Sync {
    /// Generate speech from text.
    ///
    /// # Arguments
    ///
    /// * `request` - The speech request containing text and parameters
    ///
    /// # Returns
    ///
    /// A `SpeechResponse` containing the audio data, or an error.
    async fn speech(&self, request: &SpeechRequest) -> Result<SpeechResponse>;

    /// Generate speech and save to a file.
    ///
    /// This is a convenience method that calls `speech` and saves the result.
    ///
    /// # Arguments
    ///
    /// * `request` - The speech request
    /// * `path` - Path to save the audio file
    ///
    /// # Returns
    ///
    /// The `SpeechResponse` on success, or an error.
    async fn speech_to_file(
        &self,
        request: &SpeechRequest,
        path: impl AsRef<std::path::Path> + Send,
    ) -> Result<SpeechResponse> {
        use crate::error::LlmError;

        let response = self.speech(request).await?;
        response
            .save(&path)
            .map_err(|e| LlmError::internal(format!("Failed to save audio file: {e}")))?;
        Ok(response)
    }

    /// List available voices for this provider.
    ///
    /// The default implementation returns an empty list.
    fn available_voices(&self) -> Vec<Voice> {
        Vec::new()
    }
}

/// Trait for providers that support speech-to-text transcription.
///
/// This trait defines the interface for converting audio data into text.
/// Providers like `OpenAI` (Whisper) implement this trait.
#[async_trait]
pub trait SpeechToTextProvider: Send + Sync {
    /// Transcribe audio data to text.
    ///
    /// # Arguments
    ///
    /// * `request` - The transcription request containing audio and parameters
    ///
    /// # Returns
    ///
    /// A `TranscriptionResponse` containing the transcribed text, or an error.
    async fn transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse>;

    /// Transcribe audio from a file path.
    ///
    /// This is a convenience method that reads the file and calls `transcribe`.
    /// The default implementation reads the file synchronously.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for transcription
    /// * `file_path` - Path to the audio file
    ///
    /// # Returns
    ///
    /// A `TranscriptionResponse` containing the transcribed text, or an error.
    async fn transcribe_file(&self, model: &str, file_path: &str) -> Result<TranscriptionResponse> {
        use crate::error::LlmError;

        let audio = std::fs::read(file_path)
            .map_err(|e| LlmError::internal(format!("Failed to read audio file: {e}")))?;

        // Detect format from extension
        let format = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(AudioFormat::from_extension)
            .unwrap_or_default();

        let request = TranscriptionRequest::new(model, audio).format(format);
        self.transcribe(&request).await
    }
}
