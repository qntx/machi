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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::audio::TextToSpeechProvider;
    use crate::llms::openai::OpenAIConfig;

    mod openai_speech_request {
        use super::*;

        #[test]
        fn serializes_required_fields() {
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: "Hello world".to_owned(),
                voice: "alloy".to_owned(),
                response_format: None,
                speed: None,
                instructions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["model"], "tts-1");
            assert_eq!(json["input"], "Hello world");
            assert_eq!(json["voice"], "alloy");
        }

        #[test]
        fn skips_none_optional_fields() {
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: "Test".to_owned(),
                voice: "nova".to_owned(),
                response_format: None,
                speed: None,
                instructions: None,
            };

            let json = serde_json::to_string(&req).unwrap();

            assert!(!json.contains("response_format"));
            assert!(!json.contains("speed"));
            assert!(!json.contains("instructions"));
        }

        #[test]
        fn includes_response_format_when_set() {
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: "Test".to_owned(),
                voice: "alloy".to_owned(),
                response_format: Some("mp3".to_owned()),
                speed: None,
                instructions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["response_format"], "mp3");
        }

        #[test]
        fn includes_speed_when_set() {
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: "Test".to_owned(),
                voice: "alloy".to_owned(),
                response_format: None,
                speed: Some(1.5),
                instructions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["speed"], 1.5);
        }

        #[test]
        fn includes_instructions_when_set() {
            let req = OpenAISpeechRequest {
                model: "tts-1-hd".to_owned(),
                input: "Test".to_owned(),
                voice: "alloy".to_owned(),
                response_format: None,
                speed: None,
                instructions: Some("Speak slowly and clearly".to_owned()),
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["instructions"], "Speak slowly and clearly");
        }

        #[test]
        fn handles_all_tts_models() {
            for model in ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"] {
                let req = OpenAISpeechRequest {
                    model: model.to_owned(),
                    input: "Test".to_owned(),
                    voice: "alloy".to_owned(),
                    response_format: None,
                    speed: None,
                    instructions: None,
                };
                let json = serde_json::to_value(&req).unwrap();
                assert_eq!(json["model"], model);
            }
        }

        #[test]
        fn handles_all_voices() {
            let voices = [
                "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage",
                "shimmer",
            ];
            for voice in voices {
                let req = OpenAISpeechRequest {
                    model: "tts-1".to_owned(),
                    input: "Test".to_owned(),
                    voice: voice.to_owned(),
                    response_format: None,
                    speed: None,
                    instructions: None,
                };
                let json = serde_json::to_value(&req).unwrap();
                assert_eq!(json["voice"], voice);
            }
        }

        #[test]
        fn handles_all_response_formats() {
            for format in ["mp3", "opus", "aac", "flac", "wav", "pcm"] {
                let req = OpenAISpeechRequest {
                    model: "tts-1".to_owned(),
                    input: "Test".to_owned(),
                    voice: "alloy".to_owned(),
                    response_format: Some(format.to_owned()),
                    speed: None,
                    instructions: None,
                };
                let json = serde_json::to_value(&req).unwrap();
                assert_eq!(json["response_format"], format);
            }
        }

        #[test]
        fn speed_range() {
            // Valid speed range is 0.25 to 4.0
            for speed in [0.25_f32, 0.5, 1.0, 2.0, 4.0] {
                let req = OpenAISpeechRequest {
                    model: "tts-1".to_owned(),
                    input: "Test".to_owned(),
                    voice: "alloy".to_owned(),
                    response_format: None,
                    speed: Some(speed),
                    instructions: None,
                };
                let json = serde_json::to_value(&req).unwrap();
                assert!((json["speed"].as_f64().unwrap() as f32 - speed).abs() < 0.001);
            }
        }

        #[test]
        fn handles_unicode_input() {
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: "你好世界！こんにちは！🌍".to_owned(),
                voice: "alloy".to_owned(),
                response_format: None,
                speed: None,
                instructions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["input"], "你好世界！こんにちは！🌍");
        }

        #[test]
        fn handles_long_input() {
            let long_text = "Hello world. ".repeat(100);
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: long_text.clone(),
                voice: "alloy".to_owned(),
                response_format: None,
                speed: None,
                instructions: None,
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["input"], long_text);
        }
    }

    mod openai_transcription_response {
        use super::*;

        #[test]
        fn deserializes_minimal_response() {
            let json = r#"{"text": "Hello world"}"#;
            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.text, "Hello world");
            assert!(response.language.is_none());
            assert!(response.duration.is_none());
        }

        #[test]
        fn deserializes_full_response() {
            let json = r#"{
                "text": "Hello world",
                "language": "en",
                "duration": 2.5
            }"#;

            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.text, "Hello world");
            assert_eq!(response.language, Some("en".to_owned()));
            assert!((response.duration.unwrap() - 2.5).abs() < 0.001);
        }

        #[test]
        fn handles_various_languages() {
            let languages = ["en", "ja", "zh", "es", "fr", "de", "ko", "ru"];
            for lang in languages {
                let json = format!(r#"{{"text": "test", "language": "{lang}"}}"#);
                let response: OpenAITranscriptionResponse = serde_json::from_str(&json).unwrap();
                assert_eq!(response.language, Some(lang.to_owned()));
            }
        }

        #[test]
        fn handles_long_duration() {
            let json = r#"{"text": "Long audio", "duration": 3600.5}"#;
            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert!((response.duration.unwrap() - 3600.5).abs() < 0.001);
        }

        #[test]
        fn handles_unicode_text() {
            let json = r#"{"text": "你好世界 🌍"}"#;
            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.text, "你好世界 🌍");
        }

        #[test]
        fn handles_multiline_text() {
            let json = r#"{"text": "Line 1.\nLine 2.\nLine 3."}"#;
            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert!(response.text.contains('\n'));
        }
    }

    mod available_voices {
        use super::*;

        fn test_client() -> OpenAI {
            OpenAI::new(OpenAIConfig::new("test-key")).unwrap()
        }

        #[test]
        fn returns_all_openai_voices() {
            let client = test_client();
            let voices = client.available_voices();

            assert_eq!(voices.len(), 10);
        }

        #[test]
        fn contains_alloy() {
            let client = test_client();
            let voices = client.available_voices();

            assert!(voices.iter().any(|v| v.id == "alloy"));
        }

        #[test]
        fn contains_all_expected_voices() {
            let client = test_client();
            let voices = client.available_voices();

            let expected_ids = [
                "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage",
                "shimmer",
            ];

            for id in expected_ids {
                assert!(voices.iter().any(|v| v.id == id), "Missing voice: {id}");
            }
        }

        #[test]
        fn all_voices_have_descriptions() {
            let client = test_client();
            let voices = client.available_voices();

            for voice in &voices {
                assert!(
                    voice.description.is_some(),
                    "Voice {} missing description",
                    voice.id
                );
            }
        }

        #[test]
        fn voice_descriptions_are_meaningful() {
            let client = test_client();
            let voices = client.available_voices();

            for voice in &voices {
                let desc = voice.description.as_ref().unwrap();
                assert!(
                    desc.len() > 10,
                    "Voice {} has too short description: {}",
                    voice.id,
                    desc
                );
            }
        }
    }

    mod transcription_response_conversion {
        use super::*;

        #[test]
        fn converts_verbose_json_response() {
            let openai_response = OpenAITranscriptionResponse {
                text: "Hello world".to_owned(),
                language: Some("en".to_owned()),
                duration: Some(2.5),
            };

            let response = TranscriptionResponse {
                text: openai_response.text,
                language: openai_response.language,
                duration: openai_response.duration,
                words: None,
                segments: None,
            };

            assert_eq!(response.text, "Hello world");
            assert_eq!(response.language, Some("en".to_owned()));
            assert!((response.duration.unwrap() - 2.5).abs() < 0.001);
        }

        #[test]
        fn converts_plain_text_response() {
            let text = "Plain transcription text";
            let response = TranscriptionResponse::new(text);

            assert_eq!(response.text, text);
            assert!(response.language.is_none());
            assert!(response.duration.is_none());
        }
    }

    mod realistic_scenarios {
        use super::*;

        #[test]
        fn verbose_json_transcription() {
            // Actual OpenAI verbose_json response format
            let json = r#"{
                "task": "transcribe",
                "language": "english",
                "duration": 8.470000267028809,
                "text": "The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing volleyball.",
                "words": [
                    {
                        "word": "The",
                        "start": 0.0,
                        "end": 0.5
                    }
                ]
            }"#;

            // Our struct only parses what we need
            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert!(response.text.contains("beach"));
            assert!(response.duration.is_some());
        }

        #[test]
        fn speech_request_full_options() {
            let req = OpenAISpeechRequest {
                model: "tts-1-hd".to_owned(),
                input: "Welcome to our service. How may I assist you today?".to_owned(),
                voice: "nova".to_owned(),
                response_format: Some("opus".to_owned()),
                speed: Some(1.1),
                instructions: Some("Speak in a friendly, professional tone.".to_owned()),
            };

            let json = serde_json::to_value(&req).unwrap();

            assert_eq!(json["model"], "tts-1-hd");
            assert_eq!(json["voice"], "nova");
            assert_eq!(json["response_format"], "opus");
            // f32 precision: 1.1 becomes ~1.100000023841858
            assert!((json["speed"].as_f64().unwrap() - 1.1).abs() < 0.001);
            assert!(json["instructions"].as_str().unwrap().contains("friendly"));
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn handles_empty_transcription_text() {
            let json = r#"{"text": ""}"#;
            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert!(response.text.is_empty());
        }

        #[test]
        fn handles_zero_duration() {
            let json = r#"{"text": "test", "duration": 0.0}"#;
            let response: OpenAITranscriptionResponse = serde_json::from_str(json).unwrap();

            assert!((response.duration.unwrap() - 0.0).abs() < 0.001);
        }

        #[test]
        fn speech_request_empty_input() {
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: String::new(),
                voice: "alloy".to_owned(),
                response_format: None,
                speed: None,
                instructions: None,
            };

            let json = serde_json::to_value(&req).unwrap();
            assert_eq!(json["input"], "");
        }

        #[test]
        fn speech_request_special_characters() {
            let req = OpenAISpeechRequest {
                model: "tts-1".to_owned(),
                input: "Hello! How are you? <test> & \"quotes\"".to_owned(),
                voice: "alloy".to_owned(),
                response_format: None,
                speed: None,
                instructions: None,
            };

            let json = serde_json::to_value(&req).unwrap();
            assert!(json["input"].as_str().unwrap().contains("<test>"));
            assert!(json["input"].as_str().unwrap().contains('&'));
        }
    }
}
