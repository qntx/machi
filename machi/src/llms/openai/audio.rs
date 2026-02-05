//! OpenAI Audio API implementation (TTS & STT).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::audio::{
    SpeechRequest, SpeechResponse, SpeechToTextProvider, TextToSpeechProvider,
    TranscriptionRequest, TranscriptionResponse, Voice,
};
use crate::error::{LlmError, Result};

use super::client::OpenAI;

/// OpenAI text-to-speech request.
#[derive(Debug, Clone, Serialize)]
struct OpenAISpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// OpenAI transcription response (verbose JSON format).
#[derive(Debug, Clone, Deserialize)]
struct OpenAITranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub duration: Option<f32>,
}

#[async_trait]
impl TextToSpeechProvider for OpenAI {
    async fn speech(&self, request: &SpeechRequest) -> Result<SpeechResponse> {
        let url = self.speech_url();

        let body = OpenAISpeechRequest {
            model: request.model.clone(),
            input: request.input.clone(),
            voice: request.voice.id.clone(),
            response_format: Some(request.response_format.as_str().to_owned()),
            speed: request.speed,
            instructions: request.instructions.clone(),
        };

        let response = self
            .build_request(&url)
            .json(&body)
            .send()
            .await
            .map_err(LlmError::from)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let audio = response.bytes().await.map_err(LlmError::from)?.to_vec();

        Ok(SpeechResponse::new(audio, request.response_format))
    }

    fn available_voices(&self) -> Vec<Voice> {
        vec![
            Voice::new("alloy").description("A neutral, balanced voice"),
            Voice::new("ash").description("A warm, gentle voice"),
            Voice::new("ballad").description("A soft, melodic voice"),
            Voice::new("coral").description("A clear, professional voice"),
            Voice::new("echo").description("A crisp, energetic voice"),
            Voice::new("fable").description("An expressive, storytelling voice"),
            Voice::new("onyx").description("A deep, authoritative voice"),
            Voice::new("nova").description("A friendly, conversational voice"),
            Voice::new("sage").description("A calm, wise voice"),
            Voice::new("shimmer").description("A bright, optimistic voice"),
        ]
    }
}

#[async_trait]
impl SpeechToTextProvider for OpenAI {
    async fn transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        let url = self.transcriptions_url();

        // Build filename with correct extension
        let filename = format!("audio.{}", request.format.extension());

        // Create multipart form
        let file_part = reqwest::multipart::Part::bytes(request.audio.clone())
            .file_name(filename)
            .mime_str(request.format.mime_type())
            .map_err(|e| LlmError::internal(format!("Invalid MIME type: {e}")))?;

        let mut form = reqwest::multipart::Form::new()
            .text("model", request.model.clone())
            .part("file", file_part);

        // Add optional parameters
        if let Some(ref lang) = request.language {
            form = form.text("language", lang.clone());
        }
        if let Some(ref prompt) = request.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(format) = request.response_format {
            form = form.text("response_format", format.as_str());
        } else {
            // Default to verbose_json to get language and duration
            form = form.text("response_format", "verbose_json");
        }
        if let Some(temp) = request.temperature {
            form = form.text("temperature", temp.to_string());
        }

        let response = self
            .build_multipart_request(&url)
            .multipart(form)
            .send()
            .await
            .map_err(LlmError::from)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let response_text = response.text().await.map_err(LlmError::from)?;

        // Try parsing as verbose JSON first, fall back to plain text
        if let Ok(parsed) = serde_json::from_str::<OpenAITranscriptionResponse>(&response_text) {
            Ok(TranscriptionResponse {
                text: parsed.text,
                language: parsed.language,
                duration: parsed.duration,
                words: None,    // TODO: Parse words from verbose response
                segments: None, // TODO: Parse segments from verbose response
            })
        } else {
            // Plain text response
            Ok(TranscriptionResponse::new(response_text))
        }
    }
}
