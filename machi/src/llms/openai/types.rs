//! OpenAI API request and response types.
//!
//! This module contains types that map directly to OpenAI's Chat Completions API.
//! These are internal types used for serialization/deserialization with the API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::usage::Usage;

/// OpenAI chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    /// Deprecated: use max_completion_tokens instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Max tokens including visible output and reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    /// Reasoning effort for o1/o3 models (low, medium, high).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Seed for deterministic sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

/// Stream options for OpenAI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// OpenAI message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// OpenAI message content variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    Text(String),
    Array(Vec<OpenAIContentPart>),
}

/// OpenAI content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAIImageUrl },
    InputAudio { input_audio: OpenAIInputAudio },
}

/// OpenAI image URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// OpenAI input audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIInputAudio {
    /// Base64-encoded audio data.
    pub data: String,
    /// Audio format (wav or mp3).
    pub format: String,
}

/// OpenAI tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunction,
}

/// OpenAI function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    /// Enable strict schema validation (Structured Outputs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// OpenAI tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

/// OpenAI function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// OpenAI response format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Value },
}

impl OpenAIResponseFormat {
    /// Creates from our ResponseFormat type.
    pub fn from_response_format(format: &crate::chat::ResponseFormat) -> Self {
        match format {
            crate::chat::ResponseFormat::Text => Self::Text,
            crate::chat::ResponseFormat::JsonObject => Self::JsonObject,
            crate::chat::ResponseFormat::JsonSchema { json_schema } => Self::JsonSchema {
                json_schema: serde_json::json!({
                    "name": json_schema.name,
                    "schema": json_schema.schema,
                    "strict": json_schema.strict,
                }),
            },
        }
    }
}

/// OpenAI chat completion response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Service tier used for processing.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Backend configuration fingerprint.
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

/// OpenAI response choice.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIChoice {
    pub index: usize,
    pub message: OpenAIResponseMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI response message.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: Option<String>,
    /// Refusal message if the model declined to respond.
    #[serde(default)]
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    /// Annotations on the message (citations, etc.).
    #[serde(default)]
    pub annotations: Option<Vec<Value>>,
}

/// OpenAI streaming chunk.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIStreamChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// OpenAI stream choice.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIStreamChoice {
    pub index: usize,
    pub delta: OpenAIStreamDelta,
    pub finish_reason: Option<String>,
}

/// OpenAI stream delta.
#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIStreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    /// Refusal message delta.
    #[serde(default)]
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

/// OpenAI stream tool call delta.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIStreamToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<OpenAIStreamFunctionCall>,
}

/// OpenAI stream function call delta.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIStreamFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// OpenAI error response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

/// OpenAI error details.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// OpenAI embedding request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIEmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// OpenAI embedding data.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIEmbeddingData {
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// OpenAI embedding response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIEmbeddingResponse {
    pub data: Vec<OpenAIEmbeddingData>,
    pub model: String,
    pub usage: Option<OpenAIEmbeddingUsage>,
}

/// OpenAI embedding usage statistics.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OpenAIEmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI text-to-speech request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAISpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Instructions for the TTS model (gpt-4o-mini-tts only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// OpenAI transcription response (verbose JSON format).
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAITranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub duration: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = OpenAIMessage {
            role: "user".to_owned(),
            content: Some(OpenAIContent::Text("Hello".to_owned())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        let json = serde_json::to_string(&msg).expect("serialization should succeed");
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: OpenAIChatResponse =
            serde_json::from_str(json).expect("deserialization should succeed");
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
    }
}
