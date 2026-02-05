//! Ollama API request and response types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Ollama chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(default)]
    pub stream: bool,
}

/// Ollama generation options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Ollama message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Ollama tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OllamaFunction,
}

/// Ollama function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Ollama tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    pub function: OllamaFunctionCall,
}

/// Ollama function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    pub name: String,
    pub arguments: Value,
}

/// Ollama chat completion response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OllamaChatResponse {
    pub model: String,
    pub message: OllamaResponseMessage,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

/// Ollama response message.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OllamaResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Ollama streaming response chunk.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OllamaStreamChunk {
    pub model: String,
    pub message: OllamaStreamMessage,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
}

/// Ollama stream message.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OllamaStreamMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Ollama error response.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaErrorResponse {
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = OllamaMessage {
            role: "user".to_owned(),
            content: "Hello".to_owned(),
            images: None,
            tool_calls: None,
        };

        let json = serde_json::to_string(&msg).expect("serialization should succeed");
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5
        }"#;

        let response: OllamaChatResponse =
            serde_json::from_str(json).expect("deserialization should succeed");
        assert_eq!(response.model, "llama3.2");
        assert!(response.done);
    }
}
