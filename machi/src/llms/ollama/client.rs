//! Ollama API client implementation.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::ChatRequest;
use crate::error::{LlmError, Result};
use crate::message::{Content, ContentPart, Message, Role};
use crate::tool::ToolDefinition;

use super::config::OllamaConfig;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
}

/// Ollama generation options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
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

/// Ollama error response.
#[derive(Debug, Clone, Deserialize)]
struct OllamaErrorResponse {
    pub error: String,
}

/// Ollama API client.
#[derive(Debug, Clone)]
pub struct Ollama {
    pub(crate) config: Arc<OllamaConfig>,
    pub(crate) http_client: Client,
}

impl Ollama {
    /// Create a new Ollama client with the given configuration.
    pub fn new(config: OllamaConfig) -> Result<Self> {
        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout_secs {
            builder = builder.timeout(Duration::from_secs(timeout));
        }

        let http_client = builder
            .build()
            .map_err(|e| LlmError::internal(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            config: Arc::new(config),
            http_client,
        })
    }

    /// Create a client with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(OllamaConfig::default())
    }

    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self> {
        Self::new(OllamaConfig::from_env())
    }

    /// Get the base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Get the default model.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Get a reference to the HTTP client.
    #[must_use]
    pub(crate) const fn client(&self) -> &Client {
        &self.http_client
    }

    /// Build the chat API URL.
    pub(crate) fn chat_url(&self) -> String {
        format!("{}/api/chat", self.config.base_url)
    }

    /// Build the embeddings API URL.
    pub(crate) fn embeddings_url(&self) -> String {
        format!("{}/api/embed", self.config.base_url)
    }

    /// Convert Message to Ollama format (async version for URL image support).
    pub(crate) async fn convert_message_async(
        client: &Client,
        msg: &Message,
    ) -> Result<OllamaMessage> {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            // System and Developer both map to "system" in Ollama
            Role::System | Role::Developer => "system",
        };

        let (content, images) = Self::extract_content_async(client, msg).await?;

        Ok(OllamaMessage {
            role: role.to_owned(),
            content,
            images,
            tool_calls: None,
        })
    }

    /// Extract text content and images from a message (async for URL download).
    async fn extract_content_async(
        client: &Client,
        msg: &Message,
    ) -> Result<(String, Option<Vec<String>>)> {
        let Some(content) = &msg.content else {
            return Ok((String::new(), None));
        };

        match content {
            Content::Text(text) => Ok((text.clone(), None)),
            Content::Parts(parts) => {
                let mut text_parts = Vec::new();
                let mut images = Vec::new();

                for part in parts {
                    match part {
                        ContentPart::Text { text } => text_parts.push(text.clone()),
                        ContentPart::ImageUrl { image_url } => {
                            let url = &image_url.url;
                            // Handle data URL (base64 encoded)
                            if let Some(data) = url.strip_prefix("data:")
                                && let Some(base64_start) = data.find(";base64,")
                            {
                                let base64_data = &data[base64_start + 8..];
                                images.push(base64_data.to_owned());
                            }
                            // Handle http/https URL - download and convert to base64
                            else if url.starts_with("http://") || url.starts_with("https://") {
                                let base64_data =
                                    Self::download_image_as_base64(client, url).await?;
                                images.push(base64_data);
                            }
                        }
                        ContentPart::InputAudio { .. } => {
                            // Ollama doesn't support audio input, skip
                        }
                    }
                }

                let images = if images.is_empty() {
                    None
                } else {
                    Some(images)
                };

                Ok((text_parts.join("\n"), images))
            }
        }
    }

    /// Download an image from URL and convert to base64.
    async fn download_image_as_base64(client: &Client, url: &str) -> Result<String> {
        let response = client
            .get(url)
            .header("User-Agent", "machi/0.5")
            .send()
            .await
            .map_err(|e| LlmError::internal(format!("Failed to download image: {e}")))?;

        if !response.status().is_success() {
            return Err(LlmError::internal(format!(
                "Failed to download image: HTTP {}",
                response.status()
            ))
            .into());
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| LlmError::internal(format!("Failed to read image bytes: {e}")))?;

        Ok(base64::engine::general_purpose::STANDARD.encode(&bytes))
    }

    /// Convert ToolDefinition to Ollama format.
    pub(crate) fn convert_tool(tool: &ToolDefinition) -> OllamaTool {
        OllamaTool {
            tool_type: "function".to_owned(),
            function: OllamaFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            },
        }
    }

    /// Build the request body (async for URL image support).
    pub(crate) async fn build_body(&self, request: &ChatRequest) -> Result<OllamaChatRequest> {
        let mut messages = Vec::with_capacity(request.messages.len());
        for msg in &request.messages {
            let converted = Self::convert_message_async(&self.http_client, msg).await?;
            messages.push(converted);
        }

        let tools = request
            .tools
            .as_ref()
            .map(|t| t.iter().map(Self::convert_tool).collect());

        let model = if request.model.is_empty() {
            self.config.model.clone()
        } else {
            request.model.clone()
        };

        let options = if request.temperature.is_some()
            || request.top_p.is_some()
            || request.max_tokens.is_some()
            || request.stop.is_some()
            || request.seed.is_some()
        {
            #[allow(clippy::cast_possible_wrap)]
            Some(OllamaOptions {
                temperature: request.temperature,
                top_p: request.top_p,
                num_predict: request.max_tokens.map(|t| t as i32),
                seed: request.seed,
                stop: request.stop.clone(),
                ..Default::default()
            })
        } else {
            None
        };

        let format = request.response_format.as_ref().and_then(|f| match f {
            crate::chat::ResponseFormat::JsonObject => Some(serde_json::json!("json")),
            crate::chat::ResponseFormat::JsonSchema { json_schema } => {
                Some(json_schema.schema.clone())
            }
            crate::chat::ResponseFormat::Text => None,
        });

        Ok(OllamaChatRequest {
            model,
            messages,
            tools,
            format,
            options,
            stream: request.stream,
            keep_alive: self.config.keep_alive.clone(),
            think: None, // Enable via model-specific configuration if needed
        })
    }

    /// Parse an error response from Ollama.
    pub(crate) fn parse_error(status: u16, body: &str) -> LlmError {
        if let Ok(error_response) = serde_json::from_str::<OllamaErrorResponse>(body) {
            return LlmError::provider("ollama", error_response.error);
        }
        LlmError::http_status(status, body.to_owned())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    mod ollama_client {
        use super::*;

        #[test]
        fn new_creates_client_with_config() {
            let config = OllamaConfig::default();
            let client = Ollama::new(config).unwrap();

            assert_eq!(client.base_url(), OllamaConfig::DEFAULT_BASE_URL);
            assert_eq!(client.model(), OllamaConfig::DEFAULT_MODEL);
        }

        #[test]
        fn new_with_custom_config() {
            let config = OllamaConfig::new()
                .base_url("http://custom:11434")
                .model("llama3");
            let client = Ollama::new(config).unwrap();

            assert_eq!(client.base_url(), "http://custom:11434");
            assert_eq!(client.model(), "llama3");
        }

        #[test]
        fn new_with_timeout() {
            let config = OllamaConfig::new().timeout(60);
            let client = Ollama::new(config);

            assert!(client.is_ok());
        }

        #[test]
        fn with_defaults_creates_client() {
            let client = Ollama::with_defaults().unwrap();

            assert_eq!(client.base_url(), OllamaConfig::DEFAULT_BASE_URL);
        }

        #[test]
        fn from_env_creates_client() {
            let client = Ollama::from_env().unwrap();

            // Should have valid base URL
            assert!(!client.base_url().is_empty());
        }

        #[test]
        fn client_is_clone() {
            let client = Ollama::with_defaults().unwrap();
            let cloned = client.clone();

            assert_eq!(client.base_url(), cloned.base_url());
            assert_eq!(client.model(), cloned.model());
        }

        #[test]
        fn client_is_debug() {
            let client = Ollama::with_defaults().unwrap();
            let debug_str = format!("{client:?}");

            assert!(debug_str.contains("Ollama"));
        }
    }

    mod url_building {
        use super::*;

        #[test]
        fn chat_url_format() {
            let client = Ollama::with_defaults().unwrap();
            let url = client.chat_url();

            assert_eq!(url, "http://localhost:11434/api/chat");
        }

        #[test]
        fn embeddings_url_format() {
            let client = Ollama::with_defaults().unwrap();
            let url = client.embeddings_url();

            assert_eq!(url, "http://localhost:11434/api/embed");
        }

        #[test]
        fn urls_with_custom_base() {
            let config = OllamaConfig::new().base_url("http://gpu-server:11434");
            let client = Ollama::new(config).unwrap();

            assert_eq!(client.chat_url(), "http://gpu-server:11434/api/chat");
            assert_eq!(client.embeddings_url(), "http://gpu-server:11434/api/embed");
        }

        #[test]
        fn urls_without_trailing_slash() {
            let config = OllamaConfig::new().base_url("http://server:11434/");
            let client = Ollama::new(config).unwrap();

            // Note: Current implementation doesn't strip trailing slash
            assert!(client.chat_url().contains("/api/chat"));
        }
    }

    mod message_conversion {
        use super::*;

        #[tokio::test]
        async fn converts_user_message() {
            let client = Client::new();
            let msg = Message::user("Hello!");
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.role, "user");
            assert_eq!(converted.content, "Hello!");
            assert!(converted.images.is_none());
            assert!(converted.tool_calls.is_none());
        }

        #[tokio::test]
        async fn converts_assistant_message() {
            let client = Client::new();
            let msg = Message::assistant("I can help with that.");
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.role, "assistant");
            assert_eq!(converted.content, "I can help with that.");
        }

        #[tokio::test]
        async fn converts_system_message() {
            let client = Client::new();
            let msg = Message::system("You are a helpful assistant.");
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.role, "system");
            assert_eq!(converted.content, "You are a helpful assistant.");
        }

        #[tokio::test]
        async fn converts_tool_message() {
            let client = Client::new();
            let msg = Message::tool("call_123", r#"{"result": "success"}"#);
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.role, "tool");
        }

        #[tokio::test]
        async fn handles_empty_content() {
            let client = Client::new();
            let msg = Message {
                role: Role::User,
                content: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                annotations: Vec::new(),
                reasoning_content: None,
                thinking_blocks: None,
            };
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.content, "");
            assert!(converted.images.is_none());
        }

        #[tokio::test]
        async fn handles_unicode_content() {
            let client = Client::new();
            let msg = Message::user("你好世界 🌍 مرحبا");
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.content, "你好世界 🌍 مرحبا");
        }

        #[tokio::test]
        async fn handles_multiline_content() {
            let client = Client::new();
            let msg = Message::user("Line 1\nLine 2\nLine 3");
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.content, "Line 1\nLine 2\nLine 3");
        }
    }

    mod content_extraction {
        use super::*;
        use crate::message::ImageUrl;

        #[tokio::test]
        async fn extracts_text_content() {
            let client = Client::new();
            let msg = Message::user("Simple text");
            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.content, "Simple text");
            assert!(converted.images.is_none());
        }

        #[tokio::test]
        async fn extracts_base64_image_from_data_url() {
            let client = Client::new();
            let data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

            let msg = Message {
                role: Role::User,
                content: Some(Content::Parts(vec![
                    ContentPart::Text {
                        text: "What's in this image?".to_owned(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: data_url.to_owned(),
                            detail: None,
                        },
                    },
                ])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                annotations: Vec::new(),
                reasoning_content: None,
                thinking_blocks: None,
            };

            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.content, "What's in this image?");
            assert!(converted.images.is_some());
            let images = converted.images.unwrap();
            assert_eq!(images.len(), 1);
            // Should extract only the base64 part, not the data: prefix
            assert!(!images[0].starts_with("data:"));
        }

        #[tokio::test]
        async fn combines_multiple_text_parts() {
            let client = Client::new();
            let msg = Message {
                role: Role::User,
                content: Some(Content::Parts(vec![
                    ContentPart::Text {
                        text: "First part".to_owned(),
                    },
                    ContentPart::Text {
                        text: "Second part".to_owned(),
                    },
                ])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                annotations: Vec::new(),
                reasoning_content: None,
                thinking_blocks: None,
            };

            let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

            assert_eq!(converted.content, "First part\nSecond part");
        }
    }

    mod tool_conversion {
        use super::*;

        #[test]
        fn converts_basic_tool() {
            let tool = ToolDefinition::new(
                "get_weather",
                "Get current weather",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
            );

            let converted = Ollama::convert_tool(&tool);

            assert_eq!(converted.tool_type, "function");
            assert_eq!(converted.function.name, "get_weather");
            assert_eq!(converted.function.description, "Get current weather");
        }

        #[test]
        fn preserves_parameters_schema() {
            let params = serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            });

            let tool = ToolDefinition::new("search", "Search for items", params.clone());
            let converted = Ollama::convert_tool(&tool);

            assert_eq!(converted.function.parameters, params);
        }

        #[test]
        fn handles_empty_parameters() {
            let tool = ToolDefinition::new(
                "no_params",
                "A tool with no parameters",
                serde_json::json!({}),
            );

            let converted = Ollama::convert_tool(&tool);

            assert_eq!(converted.function.parameters, serde_json::json!({}));
        }

        #[test]
        fn converted_tool_is_serializable() {
            let tool =
                ToolDefinition::new("test", "Test tool", serde_json::json!({"type": "object"}));

            let converted = Ollama::convert_tool(&tool);
            let json = serde_json::to_string(&converted).unwrap();

            assert!(json.contains("\"type\":\"function\""));
            assert!(json.contains("\"name\":\"test\""));
        }
    }

    mod error_parsing {
        use super::*;

        #[test]
        fn parses_ollama_error_response() {
            let body = r#"{"error":"model not found"}"#;
            let error = Ollama::parse_error(404, body);

            let error_str = error.to_string();
            assert!(error_str.contains("model not found"));
        }

        #[test]
        fn parses_ollama_error_with_details() {
            let body = r#"{"error":"invalid request: temperature must be between 0 and 1"}"#;
            let error = Ollama::parse_error(400, body);

            let error_str = error.to_string();
            assert!(error_str.contains("temperature"));
        }

        #[test]
        fn handles_non_json_error_body() {
            let body = "Internal Server Error";
            let error = Ollama::parse_error(500, body);

            let error_str = error.to_string();
            assert!(error_str.contains("500") || error_str.contains("Internal Server Error"));
        }

        #[test]
        fn handles_empty_error_body() {
            let error = Ollama::parse_error(502, "");

            let error_str = error.to_string();
            assert!(error_str.contains("502"));
        }

        #[test]
        fn handles_malformed_json_error() {
            let body = r#"{"error": incomplete"#;
            let error = Ollama::parse_error(400, body);

            // Should fall back to HTTP status error
            let error_str = error.to_string();
            assert!(!error_str.is_empty());
        }
    }

    mod request_body_building {
        use super::*;
        use crate::chat::ChatRequest;

        #[tokio::test]
        async fn builds_basic_request() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3").message(Message::user("Hello"));

            let body = client.build_body(&request).await.unwrap();

            assert_eq!(body.model, "llama3");
            assert_eq!(body.messages.len(), 1);
            assert!(!body.stream);
        }

        #[tokio::test]
        async fn uses_default_model_when_empty() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("").message(Message::user("Hello"));

            let body = client.build_body(&request).await.unwrap();

            assert_eq!(body.model, OllamaConfig::DEFAULT_MODEL);
        }

        #[tokio::test]
        async fn includes_temperature_option() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .temperature(0.7);

            let body = client.build_body(&request).await.unwrap();

            assert!(body.options.is_some());
            let options = body.options.unwrap();
            assert_eq!(options.temperature, Some(0.7));
        }

        #[tokio::test]
        async fn includes_max_tokens_as_num_predict() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .max_tokens(100);

            let body = client.build_body(&request).await.unwrap();

            assert!(body.options.is_some());
            let options = body.options.unwrap();
            assert_eq!(options.num_predict, Some(100));
        }

        #[tokio::test]
        async fn includes_stop_sequences() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .stop(vec!["END".to_owned(), "STOP".to_owned()]);

            let body = client.build_body(&request).await.unwrap();

            assert!(body.options.is_some());
            let options = body.options.unwrap();
            assert!(options.stop.is_some());
            let stop = options.stop.unwrap();
            assert_eq!(stop.len(), 2);
        }

        #[tokio::test]
        async fn includes_seed() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .seed(42);

            let body = client.build_body(&request).await.unwrap();

            assert!(body.options.is_some());
            let options = body.options.unwrap();
            assert_eq!(options.seed, Some(42));
        }

        #[tokio::test]
        async fn no_options_when_defaults() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3").message(Message::user("Hello"));

            let body = client.build_body(&request).await.unwrap();

            assert!(body.options.is_none());
        }

        #[tokio::test]
        async fn includes_tools() {
            let client = Ollama::with_defaults().unwrap();
            let tool = ToolDefinition::new(
                "test_tool",
                "A test tool",
                serde_json::json!({"type": "object"}),
            );
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .tools(vec![tool]);

            let body = client.build_body(&request).await.unwrap();

            assert!(body.tools.is_some());
            let tools = body.tools.unwrap();
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].function.name, "test_tool");
        }

        #[tokio::test]
        async fn sets_stream_flag() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .stream();

            let body = client.build_body(&request).await.unwrap();

            assert!(body.stream);
        }

        #[tokio::test]
        async fn includes_keep_alive_from_config() {
            let config = OllamaConfig::new().keep_alive("5m");
            let client = Ollama::new(config).unwrap();
            let request = ChatRequest::new("llama3").message(Message::user("Hello"));

            let body = client.build_body(&request).await.unwrap();

            assert_eq!(body.keep_alive, Some("5m".to_owned()));
        }

        #[tokio::test]
        async fn handles_json_response_format() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .response_format(crate::chat::ResponseFormat::JsonObject);

            let body = client.build_body(&request).await.unwrap();

            assert!(body.format.is_some());
            assert_eq!(body.format.unwrap(), serde_json::json!("json"));
        }

        #[tokio::test]
        async fn handles_text_response_format() {
            let client = Ollama::with_defaults().unwrap();
            let request = ChatRequest::new("llama3")
                .message(Message::user("Hello"))
                .response_format(crate::chat::ResponseFormat::Text);

            let body = client.build_body(&request).await.unwrap();

            assert!(body.format.is_none());
        }
    }

    mod serialization {
        use super::*;

        #[test]
        fn ollama_options_serializes_only_present_fields() {
            let options = OllamaOptions {
                temperature: Some(0.5),
                ..Default::default()
            };

            let json = serde_json::to_string(&options).unwrap();

            assert!(json.contains("temperature"));
            assert!(!json.contains("top_p"));
            assert!(!json.contains("top_k"));
        }

        #[test]
        fn ollama_message_serializes_correctly() {
            let msg = OllamaMessage {
                role: "user".to_owned(),
                content: "Hello".to_owned(),
                images: None,
                tool_calls: None,
            };

            let json = serde_json::to_string(&msg).unwrap();

            assert!(json.contains("\"role\":\"user\""));
            assert!(json.contains("\"content\":\"Hello\""));
            assert!(!json.contains("images"));
            assert!(!json.contains("tool_calls"));
        }

        #[test]
        fn ollama_message_with_images() {
            let msg = OllamaMessage {
                role: "user".to_owned(),
                content: "What's this?".to_owned(),
                images: Some(vec!["base64data".to_owned()]),
                tool_calls: None,
            };

            let json = serde_json::to_string(&msg).unwrap();

            assert!(json.contains("images"));
            assert!(json.contains("base64data"));
        }

        #[test]
        fn ollama_tool_uses_type_rename() {
            let tool = OllamaTool {
                tool_type: "function".to_owned(),
                function: OllamaFunction {
                    name: "test".to_owned(),
                    description: "Test".to_owned(),
                    parameters: serde_json::json!({}),
                },
            };

            let json = serde_json::to_string(&tool).unwrap();

            // Should serialize as "type", not "tool_type"
            assert!(json.contains("\"type\":\"function\""));
            assert!(!json.contains("tool_type"));
        }

        #[test]
        fn ollama_chat_request_stream_defaults_false() {
            let request = OllamaChatRequest {
                model: "llama3".to_owned(),
                messages: vec![],
                tools: None,
                format: None,
                options: None,
                stream: false,
                keep_alive: None,
                think: None,
            };

            let json = serde_json::to_string(&request).unwrap();

            assert!(json.contains("\"stream\":false"));
        }
    }

    mod deserialization {
        use super::*;

        #[test]
        fn ollama_options_deserializes() {
            let json = r#"{"temperature":0.7,"top_p":0.9}"#;
            let options: OllamaOptions = serde_json::from_str(json).unwrap();

            assert_eq!(options.temperature, Some(0.7));
            assert_eq!(options.top_p, Some(0.9));
            assert!(options.top_k.is_none());
        }

        #[test]
        fn ollama_message_deserializes() {
            let json = r#"{"role":"assistant","content":"Hello!"}"#;
            let msg: OllamaMessage = serde_json::from_str(json).unwrap();

            assert_eq!(msg.role, "assistant");
            assert_eq!(msg.content, "Hello!");
        }

        #[test]
        fn ollama_tool_call_deserializes() {
            let json = r#"{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}"#;
            let call: OllamaToolCall = serde_json::from_str(json).unwrap();

            assert_eq!(call.function.name, "get_weather");
            assert_eq!(call.function.arguments["city"], "Tokyo");
        }
    }

    mod traits {
        use super::*;

        #[test]
        fn ollama_options_default() {
            let options = OllamaOptions::default();

            assert!(options.temperature.is_none());
            assert!(options.top_p.is_none());
            assert!(options.top_k.is_none());
            assert!(options.num_predict.is_none());
            assert!(options.seed.is_none());
        }

        #[test]
        fn ollama_options_clone() {
            let options = OllamaOptions {
                temperature: Some(0.5),
                seed: Some(42),
                ..Default::default()
            };

            let cloned = options;

            assert_eq!(cloned.temperature, Some(0.5));
            assert_eq!(cloned.seed, Some(42));
        }

        #[test]
        fn ollama_message_clone() {
            let msg = OllamaMessage {
                role: "user".to_owned(),
                content: "Test".to_owned(),
                images: Some(vec!["img".to_owned()]),
                tool_calls: None,
            };

            let cloned = msg.clone();

            assert_eq!(cloned.role, msg.role);
            assert_eq!(cloned.images, msg.images);
        }

        #[test]
        fn ollama_tool_clone() {
            let tool = OllamaTool {
                tool_type: "function".to_owned(),
                function: OllamaFunction {
                    name: "test".to_owned(),
                    description: "Test".to_owned(),
                    parameters: serde_json::json!({}),
                },
            };

            let cloned = tool;

            assert_eq!(cloned.function.name, "test");
        }

        #[test]
        fn all_types_are_debug() {
            let options = OllamaOptions::default();
            let msg = OllamaMessage {
                role: "user".to_owned(),
                content: String::new(),
                images: None,
                tool_calls: None,
            };
            let tool = OllamaTool {
                tool_type: "function".to_owned(),
                function: OllamaFunction {
                    name: String::new(),
                    description: String::new(),
                    parameters: serde_json::json!({}),
                },
            };

            // These should compile and not panic
            let _ = format!("{options:?}");
            let _ = format!("{msg:?}");
            let _ = format!("{tool:?}");
        }
    }
}
