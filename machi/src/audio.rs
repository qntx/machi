//! Audio processing types and provider traits.
//!
//! This module provides types and traits for audio-related operations:
//! - **Text-to-Speech (TTS)**: Converting text into audio
//! - **Speech-to-Text (STT)**: Converting audio into text (transcription)
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! // Text-to-Speech
//! let request = SpeechRequest::new("tts-1", "Hello, world!", "alloy")
//!     .format(AudioFormat::Mp3)
//!     .speed(1.0);
//! let response = provider.speech(&request).await?;
//! response.save("output.mp3")?;
//!
//! // Speech-to-Text
//! let audio = std::fs::read("input.mp3")?;
//! let request = TranscriptionRequest::new("whisper-1", audio)
//!     .format(AudioFormat::Mp3)
//!     .language("en");
//! let response = provider.transcribe(&request).await?;
//! println!("Transcribed: {}", response.text);
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Audio format for input/output operations.
///
/// Note: Not all formats are supported by all providers or operations.
/// - **TTS (OpenAI)**: mp3, opus, aac, flac, wav, pcm
/// - **STT (OpenAI)**: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
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
    /// WebM format
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
/// OpenAI built-in voices: `alloy`, `ash`, `ballad`, `coral`, `echo`,
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
/// - `whisper-1`: OpenAI Whisper V2
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
    /// Timestamp granularities to include (requires verbose_json format).
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

    /// Enable word-level timestamps (requires verbose_json format).
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

    /// Enable segment-level timestamps (requires verbose_json format).
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
    /// Word-level timestamps (when requested with verbose_json).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionWord>>,
    /// Segment-level timestamps (when requested with verbose_json).
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
/// Providers like OpenAI implement this trait.
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
/// Providers like OpenAI (Whisper) implement this trait.
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

/// Combined trait for providers that support both TTS and STT.
///
/// This is a convenience trait for providers that implement both directions
/// of audio-text conversion.
pub trait AudioProvider: TextToSpeechProvider + SpeechToTextProvider {}

// Blanket implementation for any type that implements both traits
impl<T: TextToSpeechProvider + SpeechToTextProvider> AudioProvider for T {}

#[cfg(test)]
mod tests {
    use super::*;

    mod audio_format {
        use super::*;

        #[test]
        fn default_is_wav() {
            assert_eq!(AudioFormat::default(), AudioFormat::Wav);
        }

        #[test]
        fn extension_returns_correct_values() {
            let cases = [
                (AudioFormat::Wav, "wav"),
                (AudioFormat::Mp3, "mp3"),
                (AudioFormat::Flac, "flac"),
                (AudioFormat::Ogg, "ogg"),
                (AudioFormat::WebM, "webm"),
                (AudioFormat::M4a, "m4a"),
                (AudioFormat::Opus, "opus"),
                (AudioFormat::Aac, "aac"),
                (AudioFormat::Pcm, "pcm"),
            ];

            for (format, expected) in cases {
                assert_eq!(format.extension(), expected, "format: {format:?}");
            }
        }

        #[test]
        fn mime_type_returns_correct_values() {
            let cases = [
                (AudioFormat::Wav, "audio/wav"),
                (AudioFormat::Mp3, "audio/mpeg"),
                (AudioFormat::Flac, "audio/flac"),
                (AudioFormat::Ogg, "audio/ogg"),
                (AudioFormat::WebM, "audio/webm"),
                (AudioFormat::M4a, "audio/m4a"),
                (AudioFormat::Opus, "audio/opus"),
                (AudioFormat::Aac, "audio/aac"),
                (AudioFormat::Pcm, "audio/pcm"),
            ];

            for (format, expected) in cases {
                assert_eq!(format.mime_type(), expected, "format: {format:?}");
            }
        }

        #[test]
        fn as_str_equals_extension() {
            for format in [
                AudioFormat::Wav,
                AudioFormat::Mp3,
                AudioFormat::Flac,
                AudioFormat::Ogg,
                AudioFormat::WebM,
                AudioFormat::M4a,
                AudioFormat::Opus,
                AudioFormat::Aac,
                AudioFormat::Pcm,
            ] {
                assert_eq!(format.as_str(), format.extension());
            }
        }

        #[test]
        fn from_extension_parses_valid_extensions() {
            let cases = [
                ("wav", AudioFormat::Wav),
                ("mp3", AudioFormat::Mp3),
                ("flac", AudioFormat::Flac),
                ("ogg", AudioFormat::Ogg),
                ("webm", AudioFormat::WebM),
                ("m4a", AudioFormat::M4a),
                ("opus", AudioFormat::Opus),
                ("aac", AudioFormat::Aac),
                ("pcm", AudioFormat::Pcm),
            ];

            for (ext, expected) in cases {
                assert_eq!(
                    AudioFormat::from_extension(ext),
                    Some(expected),
                    "ext: {ext}"
                );
            }
        }

        #[test]
        fn from_extension_is_case_insensitive() {
            assert_eq!(AudioFormat::from_extension("MP3"), Some(AudioFormat::Mp3));
            assert_eq!(AudioFormat::from_extension("Wav"), Some(AudioFormat::Wav));
            assert_eq!(AudioFormat::from_extension("FLAC"), Some(AudioFormat::Flac));
        }

        #[test]
        fn from_extension_returns_none_for_unknown() {
            assert_eq!(AudioFormat::from_extension("unknown"), None);
            assert_eq!(AudioFormat::from_extension(""), None);
            assert_eq!(AudioFormat::from_extension("mp4"), None);
        }

        #[test]
        fn serde_roundtrip() {
            for format in [
                AudioFormat::Wav,
                AudioFormat::Mp3,
                AudioFormat::Flac,
                AudioFormat::Ogg,
            ] {
                let json = serde_json::to_string(&format).unwrap();
                let parsed: AudioFormat = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed, format);
            }
        }

        #[test]
        fn serde_uses_lowercase() {
            assert_eq!(
                serde_json::to_string(&AudioFormat::Mp3).unwrap(),
                r#""mp3""#
            );
            assert_eq!(
                serde_json::to_string(&AudioFormat::WebM).unwrap(),
                r#""webm""#
            );
        }
    }

    mod voice {
        use super::*;

        #[test]
        fn new_creates_voice_with_id() {
            let voice = Voice::new("alloy");
            assert_eq!(voice.id, "alloy");
            assert!(voice.description.is_none());
        }

        #[test]
        fn description_sets_value() {
            let voice = Voice::new("nova").description("A warm and friendly voice");
            assert_eq!(voice.id, "nova");
            assert_eq!(
                voice.description.as_deref(),
                Some("A warm and friendly voice")
            );
        }

        #[test]
        fn from_string_creates_voice() {
            let voice: Voice = "echo".into();
            assert_eq!(voice.id, "echo");
        }

        #[test]
        fn from_owned_string_creates_voice() {
            let voice: Voice = String::from("fable").into();
            assert_eq!(voice.id, "fable");
        }

        #[test]
        fn serde_excludes_description() {
            let voice = Voice::new("alloy").description("Test description");
            let json = serde_json::to_string(&voice).unwrap();

            // Description should be skipped in serialization
            assert!(!json.contains("description"));
            assert!(json.contains("alloy"));
        }

        #[test]
        fn serde_roundtrip() {
            let voice = Voice::new("shimmer");
            let json = serde_json::to_string(&voice).unwrap();
            let parsed: Voice = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.id, voice.id);
            // Description is not serialized, so it will be None after deserialization
            assert!(parsed.description.is_none());
        }

        #[test]
        fn equality_ignores_description() {
            let v1 = Voice::new("alloy").description("desc1");
            let v2 = Voice::new("alloy").description("desc2");
            let v3 = Voice::new("alloy");

            // Note: description is part of PartialEq, so these are NOT equal
            assert_ne!(v1, v2);
            assert_ne!(v1, v3);

            // Same description means equal
            let v4 = Voice::new("alloy").description("desc1");
            assert_eq!(v1, v4);
        }
    }

    mod speech_request {
        use super::*;

        #[test]
        fn new_sets_defaults() {
            let req = SpeechRequest::new("tts-1", "Hello world", "alloy");

            assert_eq!(req.model, "tts-1");
            assert_eq!(req.input, "Hello world");
            assert_eq!(req.voice.id, "alloy");
            assert_eq!(req.response_format, AudioFormat::Mp3);
            assert!(req.speed.is_none());
            assert!(req.instructions.is_none());
        }

        #[test]
        fn format_sets_response_format() {
            let req = SpeechRequest::new("tts-1", "test", "alloy").format(AudioFormat::Wav);
            assert_eq!(req.response_format, AudioFormat::Wav);
        }

        #[test]
        fn speed_sets_value() {
            let req = SpeechRequest::new("tts-1", "test", "alloy").speed(1.5);
            assert_eq!(req.speed, Some(1.5));
        }

        #[test]
        fn instructions_sets_value() {
            let req = SpeechRequest::new("gpt-4o-mini-tts", "test", "alloy")
                .instructions("Speak cheerfully");
            assert_eq!(req.instructions.as_deref(), Some("Speak cheerfully"));
        }

        #[test]
        fn builder_chain() {
            let req = SpeechRequest::new("tts-1-hd", "Hello", "nova")
                .format(AudioFormat::Opus)
                .speed(0.8)
                .instructions("Be calm");

            assert_eq!(req.model, "tts-1-hd");
            assert_eq!(req.response_format, AudioFormat::Opus);
            assert_eq!(req.speed, Some(0.8));
            assert_eq!(req.instructions.as_deref(), Some("Be calm"));
        }

        #[test]
        fn accepts_voice_struct() {
            let voice = Voice::new("coral").description("Calm voice");
            let req = SpeechRequest::new("tts-1", "test", voice);

            assert_eq!(req.voice.id, "coral");
        }

        #[test]
        fn serde_skips_none_values() {
            let req = SpeechRequest::new("tts-1", "test", "alloy");
            let json = serde_json::to_string(&req).unwrap();

            assert!(!json.contains("speed"));
            assert!(!json.contains("instructions"));
        }
    }

    mod speech_response {
        use super::*;

        #[test]
        fn new_creates_response() {
            let audio = vec![0u8, 1, 2, 3];
            let resp = SpeechResponse::new(audio.clone(), AudioFormat::Mp3);

            assert_eq!(resp.audio, audio);
            assert_eq!(resp.format, AudioFormat::Mp3);
        }

        #[test]
        fn extension_returns_format_extension() {
            let resp = SpeechResponse::new(vec![], AudioFormat::Wav);
            assert_eq!(resp.extension(), "wav");

            let resp = SpeechResponse::new(vec![], AudioFormat::Opus);
            assert_eq!(resp.extension(), "opus");
        }

        #[test]
        fn save_writes_file() {
            let audio = b"fake audio data".to_vec();
            let resp = SpeechResponse::new(audio.clone(), AudioFormat::Mp3);

            let temp_dir = std::env::temp_dir();
            let path = temp_dir.join("test_audio_save.mp3");

            resp.save(&path).unwrap();

            let read_back = std::fs::read(&path).unwrap();
            assert_eq!(read_back, audio);

            // Cleanup
            std::fs::remove_file(path).ok();
        }
    }

    mod transcription_response_format {
        use super::*;

        #[test]
        fn default_is_json() {
            assert_eq!(
                TranscriptionResponseFormat::default(),
                TranscriptionResponseFormat::Json
            );
        }

        #[test]
        fn as_str_returns_correct_values() {
            let cases = [
                (TranscriptionResponseFormat::Json, "json"),
                (TranscriptionResponseFormat::Text, "text"),
                (TranscriptionResponseFormat::Srt, "srt"),
                (TranscriptionResponseFormat::Vtt, "vtt"),
                (TranscriptionResponseFormat::VerboseJson, "verbose_json"),
            ];

            for (format, expected) in cases {
                assert_eq!(format.as_str(), expected);
            }
        }

        #[test]
        fn serde_uses_snake_case() {
            assert_eq!(
                serde_json::to_string(&TranscriptionResponseFormat::VerboseJson).unwrap(),
                r#""verbose_json""#
            );
        }
    }

    mod timestamp_granularity {
        use super::*;

        #[test]
        fn as_str_returns_correct_values() {
            assert_eq!(TimestampGranularity::Word.as_str(), "word");
            assert_eq!(TimestampGranularity::Segment.as_str(), "segment");
        }

        #[test]
        fn serde_uses_lowercase() {
            assert_eq!(
                serde_json::to_string(&TimestampGranularity::Word).unwrap(),
                r#""word""#
            );
            assert_eq!(
                serde_json::to_string(&TimestampGranularity::Segment).unwrap(),
                r#""segment""#
            );
        }
    }

    mod transcription_request {
        use super::*;

        #[test]
        fn new_sets_defaults() {
            let audio = vec![1, 2, 3];
            let req = TranscriptionRequest::new("whisper-1", audio.clone());

            assert_eq!(req.model, "whisper-1");
            assert_eq!(req.audio, audio);
            assert_eq!(req.format, AudioFormat::default());
            assert!(req.language.is_none());
            assert!(req.prompt.is_none());
            assert!(req.response_format.is_none());
            assert!(req.temperature.is_none());
            assert!(req.timestamp_granularities.is_none());
        }

        #[test]
        fn format_sets_audio_format() {
            let req = TranscriptionRequest::new("whisper-1", vec![]).format(AudioFormat::Mp3);
            assert_eq!(req.format, AudioFormat::Mp3);
        }

        #[test]
        fn language_sets_value() {
            let req = TranscriptionRequest::new("whisper-1", vec![]).language("en");
            assert_eq!(req.language.as_deref(), Some("en"));
        }

        #[test]
        fn prompt_sets_value() {
            let req = TranscriptionRequest::new("whisper-1", vec![]).prompt("Technical terms: API");
            assert_eq!(req.prompt.as_deref(), Some("Technical terms: API"));
        }

        #[test]
        fn response_format_sets_value() {
            let req = TranscriptionRequest::new("whisper-1", vec![])
                .response_format(TranscriptionResponseFormat::Vtt);
            assert_eq!(req.response_format, Some(TranscriptionResponseFormat::Vtt));
        }

        #[test]
        fn temperature_sets_value() {
            let req = TranscriptionRequest::new("whisper-1", vec![]).temperature(0.5);
            assert_eq!(req.temperature, Some(0.5));
        }

        #[test]
        fn with_word_timestamps_adds_granularity() {
            let req = TranscriptionRequest::new("whisper-1", vec![]).with_word_timestamps();

            let granularities = req.timestamp_granularities.as_ref().unwrap();
            assert!(granularities.contains(&TimestampGranularity::Word));
            assert_eq!(
                req.response_format,
                Some(TranscriptionResponseFormat::VerboseJson)
            );
        }

        #[test]
        fn with_segment_timestamps_adds_granularity() {
            let req = TranscriptionRequest::new("whisper-1", vec![]).with_segment_timestamps();

            let granularities = req.timestamp_granularities.as_ref().unwrap();
            assert!(granularities.contains(&TimestampGranularity::Segment));
            assert_eq!(
                req.response_format,
                Some(TranscriptionResponseFormat::VerboseJson)
            );
        }

        #[test]
        fn both_timestamps_combined() {
            let req = TranscriptionRequest::new("whisper-1", vec![])
                .with_word_timestamps()
                .with_segment_timestamps();

            let granularities = req.timestamp_granularities.as_ref().unwrap();
            assert!(granularities.contains(&TimestampGranularity::Word));
            assert!(granularities.contains(&TimestampGranularity::Segment));
        }

        #[test]
        fn timestamps_not_duplicated() {
            let req = TranscriptionRequest::new("whisper-1", vec![])
                .with_word_timestamps()
                .with_word_timestamps();

            let granularities = req.timestamp_granularities.as_ref().unwrap();
            let word_count = granularities
                .iter()
                .filter(|g| **g == TimestampGranularity::Word)
                .count();
            assert_eq!(word_count, 1);
        }

        #[test]
        fn builder_chain() {
            let req = TranscriptionRequest::new("gpt-4o-transcribe", vec![1, 2, 3])
                .format(AudioFormat::Flac)
                .language("zh")
                .prompt("Chinese speech")
                .temperature(0.3)
                .with_word_timestamps();

            assert_eq!(req.model, "gpt-4o-transcribe");
            assert_eq!(req.format, AudioFormat::Flac);
            assert_eq!(req.language.as_deref(), Some("zh"));
            assert_eq!(req.prompt.as_deref(), Some("Chinese speech"));
            assert_eq!(req.temperature, Some(0.3));
            assert!(req.timestamp_granularities.is_some());
        }
    }

    mod transcription_word {
        use super::*;

        #[test]
        fn serde_roundtrip() {
            let word = TranscriptionWord {
                word: "hello".into(),
                start: 0.5,
                end: 1.2,
            };

            let json = serde_json::to_string(&word).unwrap();
            let parsed: TranscriptionWord = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.word, "hello");
            assert!((parsed.start - 0.5).abs() < f32::EPSILON);
            assert!((parsed.end - 1.2).abs() < f32::EPSILON);
        }
    }

    mod transcription_segment {
        use super::*;

        #[test]
        fn serde_roundtrip() {
            let segment = TranscriptionSegment {
                id: 0,
                start: 0.0,
                end: 5.5,
                text: "Hello, world!".into(),
            };

            let json = serde_json::to_string(&segment).unwrap();
            let parsed: TranscriptionSegment = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.id, 0);
            assert_eq!(parsed.text, "Hello, world!");
        }
    }

    mod transcription_response {
        use super::*;

        #[test]
        fn default_is_empty() {
            let resp = TranscriptionResponse::default();

            assert!(resp.text.is_empty());
            assert!(resp.language.is_none());
            assert!(resp.duration.is_none());
            assert!(resp.words.is_none());
            assert!(resp.segments.is_none());
        }

        #[test]
        fn new_creates_with_text() {
            let resp = TranscriptionResponse::new("Hello world");
            assert_eq!(resp.text, "Hello world");
        }

        #[test]
        fn with_language_sets_value() {
            let resp = TranscriptionResponse::new("test").with_language("en");
            assert_eq!(resp.language.as_deref(), Some("en"));
        }

        #[test]
        fn with_duration_sets_value() {
            let resp = TranscriptionResponse::new("test").with_duration(10.5);
            assert_eq!(resp.duration, Some(10.5));
        }

        #[test]
        fn builder_chain() {
            let resp = TranscriptionResponse::new("Hello")
                .with_language("en")
                .with_duration(2.5);

            assert_eq!(resp.text, "Hello");
            assert_eq!(resp.language.as_deref(), Some("en"));
            assert_eq!(resp.duration, Some(2.5));
        }

        #[test]
        fn serde_roundtrip() {
            let resp = TranscriptionResponse {
                text: "Hello world".into(),
                language: Some("en".into()),
                duration: Some(5.0),
                words: Some(vec![TranscriptionWord {
                    word: "Hello".into(),
                    start: 0.0,
                    end: 0.5,
                }]),
                segments: Some(vec![TranscriptionSegment {
                    id: 0,
                    start: 0.0,
                    end: 5.0,
                    text: "Hello world".into(),
                }]),
            };

            let json = serde_json::to_string(&resp).unwrap();
            let parsed: TranscriptionResponse = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.text, resp.text);
            assert_eq!(parsed.language, resp.language);
            assert_eq!(parsed.words.as_ref().unwrap().len(), 1);
            assert_eq!(parsed.segments.as_ref().unwrap().len(), 1);
        }

        #[test]
        fn serde_skips_none_values() {
            let resp = TranscriptionResponse::new("test");
            let json = serde_json::to_string(&resp).unwrap();

            assert!(!json.contains("language"));
            assert!(!json.contains("duration"));
            assert!(!json.contains("words"));
            assert!(!json.contains("segments"));
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn speech_request_json_structure() {
            let req = SpeechRequest::new("tts-1", "Hello", "alloy")
                .format(AudioFormat::Mp3)
                .speed(1.0);

            let json: serde_json::Value = serde_json::to_value(&req).unwrap();

            assert_eq!(json["model"], "tts-1");
            assert_eq!(json["input"], "Hello");
            assert_eq!(json["voice"]["id"], "alloy");
            assert_eq!(json["response_format"], "mp3");
            assert_eq!(json["speed"], 1.0);
        }

        #[test]
        fn audio_format_extension_consistency() {
            // Verify that from_extension and extension are inverse operations
            for format in [
                AudioFormat::Wav,
                AudioFormat::Mp3,
                AudioFormat::Flac,
                AudioFormat::Ogg,
                AudioFormat::WebM,
                AudioFormat::M4a,
                AudioFormat::Opus,
                AudioFormat::Aac,
                AudioFormat::Pcm,
            ] {
                let ext = format.extension();
                let recovered = AudioFormat::from_extension(ext).unwrap();
                assert_eq!(recovered, format, "Roundtrip failed for {format:?}");
            }
        }
    }
}
