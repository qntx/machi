//! OpenAI API client implementation.

use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::ChatRequest;
use crate::error::{LlmError, Result};
use crate::message::{Content, ContentPart, Message, Role};
use crate::tool::ToolDefinition;

use super::config::OpenAIConfig;

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
    /// Whether to enable parallel function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    /// Reasoning effort for o-series models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Service tier for processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// User identifier for abuse detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
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
    pub data: String,
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

/// OpenAI error response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

/// OpenAI error details.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// OpenAI API client.
#[derive(Debug, Clone)]
pub struct OpenAI {
    pub(crate) config: Arc<OpenAIConfig>,
    pub(crate) client: Client,
}

impl OpenAI {
    /// Create a new OpenAI client with the given configuration.
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(LlmError::auth("openai", "API key is required").into());
        }

        let mut builder = Client::builder();
        if let Some(timeout) = config.timeout_secs {
            builder = builder.timeout(Duration::from_secs(timeout));
        }

        let client = builder
            .build()
            .map_err(|e| LlmError::internal(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            config: Arc::new(config),
            client,
        })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self> {
        let config = OpenAIConfig::from_env()?;
        Self::new(config)
    }

    /// Get the API key.
    #[must_use]
    pub fn api_key(&self) -> &str {
        &self.config.api_key
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

    /// Build the chat completions URL.
    pub(crate) fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.config.base_url)
    }

    /// Build the audio speech URL.
    pub(crate) fn speech_url(&self) -> String {
        format!("{}/audio/speech", self.config.base_url)
    }

    /// Build the audio transcriptions URL.
    pub(crate) fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.config.base_url)
    }

    /// Build the embeddings URL.
    pub(crate) fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.config.base_url)
    }

    /// Build request headers for JSON requests.
    pub(crate) fn build_request(&self, url: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");

        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        req
    }

    /// Build request headers for multipart requests.
    pub(crate) fn build_multipart_request(&self, url: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.api_key));

        if let Some(org) = &self.config.organization {
            req = req.header("OpenAI-Organization", org);
        }

        req
    }

    /// Convert Message to OpenAI format.
    pub(crate) fn convert_message(msg: &Message) -> OpenAIMessage {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Developer => "developer",
        };

        let content = msg.content.as_ref().map(|c| match c {
            Content::Text(text) => OpenAIContent::Text(text.clone()),
            Content::Parts(parts) => {
                let openai_parts: Vec<OpenAIContentPart> = parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => {
                            OpenAIContentPart::Text { text: text.clone() }
                        }
                        ContentPart::ImageUrl { image_url } => OpenAIContentPart::ImageUrl {
                            image_url: OpenAIImageUrl {
                                url: image_url.url.clone(),
                                detail: image_url.detail.map(|d| format!("{d:?}").to_lowercase()),
                            },
                        },
                        ContentPart::InputAudio { input_audio } => OpenAIContentPart::InputAudio {
                            input_audio: OpenAIInputAudio {
                                data: input_audio.data.clone(),
                                format: input_audio.format.as_str().to_owned(),
                            },
                        },
                    })
                    .collect();
                OpenAIContent::Array(openai_parts)
            }
        });

        let tool_calls = msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| OpenAIToolCall {
                    id: tc.id.clone(),
                    call_type: "function".to_owned(),
                    function: OpenAIFunctionCall {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    },
                })
                .collect()
        });

        OpenAIMessage {
            role: role.to_owned(),
            content,
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
            name: msg.name.clone(),
        }
    }

    /// Convert ToolDefinition to OpenAI format.
    pub(crate) fn convert_tool(tool: &ToolDefinition) -> OpenAITool {
        OpenAITool {
            tool_type: "function".to_owned(),
            function: OpenAIFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            },
        }
    }

    /// Build the request body.
    pub(crate) fn build_body(&self, request: &ChatRequest) -> OpenAIChatRequest {
        let messages: Vec<OpenAIMessage> =
            request.messages.iter().map(Self::convert_message).collect();

        let tools = request
            .tools
            .as_ref()
            .map(|t| t.iter().map(Self::convert_tool).collect());

        let model = if request.model.is_empty() {
            self.config.model.clone()
        } else {
            request.model.clone()
        };

        // Prefer max_completion_tokens over deprecated max_tokens
        let (max_tokens, max_completion_tokens) = match request.max_completion_tokens {
            Some(tokens) => (None, Some(tokens)),
            None => (request.max_tokens, None),
        };

        OpenAIChatRequest {
            model,
            messages,
            max_tokens,
            max_completion_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop.clone(),
            tools,
            tool_choice: request.tool_choice.clone(),
            parallel_tool_calls: request.parallel_tool_calls,
            stream: request.stream,
            stream_options: if request.stream {
                Some(StreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            response_format: request
                .response_format
                .as_ref()
                .map(OpenAIResponseFormat::from_response_format),
            reasoning_effort: request.reasoning_effort.map(|e| e.as_str().to_owned()),
            service_tier: request.service_tier.clone(),
            user: request.user.clone(),
        }
    }

    /// Parse an error response from OpenAI.
    pub(crate) fn parse_error(status: u16, body: &str) -> LlmError {
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(body) {
            let error = error_response.error;
            let code = error.code.unwrap_or_else(|| error.error_type.clone());

            return match status {
                401 => LlmError::auth("openai", error.message),
                429 => LlmError::rate_limited("openai"),
                400 if error.message.contains("context_length") => LlmError::context_exceeded(0, 0),
                _ => LlmError::provider_code("openai", code, error.message),
            };
        }

        LlmError::http_status(status, body.to_owned())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::message::{ContentPart, ImageDetail, ImageMime, ToolCall};

    mod openai_client {
        use super::*;

        #[test]
        fn new_requires_api_key() {
            let config = OpenAIConfig::new("");
            let result = OpenAI::new(config);
            assert!(result.is_err());
        }

        #[test]
        fn new_with_valid_key_succeeds() {
            let config = OpenAIConfig::new("sk-test-key");
            let result = OpenAI::new(config);
            assert!(result.is_ok());
        }

        #[test]
        fn api_key_returns_configured_key() {
            let client = OpenAI::new(OpenAIConfig::new("sk-abc123")).unwrap();
            assert_eq!(client.api_key(), "sk-abc123");
        }

        #[test]
        fn base_url_returns_configured_url() {
            let config = OpenAIConfig::new("key").with_base_url("https://custom.api.com/v1");
            let client = OpenAI::new(config).unwrap();
            assert_eq!(client.base_url(), "https://custom.api.com/v1");
        }

        #[test]
        fn model_returns_configured_model() {
            let config = OpenAIConfig::new("key").with_model("gpt-4-turbo");
            let client = OpenAI::new(config).unwrap();
            assert_eq!(client.model(), "gpt-4-turbo");
        }

        #[test]
        fn chat_url_builds_correctly() {
            let client = OpenAI::new(OpenAIConfig::new("key")).unwrap();
            assert_eq!(
                client.chat_url(),
                "https://api.openai.com/v1/chat/completions"
            );
        }

        #[test]
        fn speech_url_builds_correctly() {
            let client = OpenAI::new(OpenAIConfig::new("key")).unwrap();
            assert_eq!(
                client.speech_url(),
                "https://api.openai.com/v1/audio/speech"
            );
        }

        #[test]
        fn transcriptions_url_builds_correctly() {
            let client = OpenAI::new(OpenAIConfig::new("key")).unwrap();
            assert_eq!(
                client.transcriptions_url(),
                "https://api.openai.com/v1/audio/transcriptions"
            );
        }

        #[test]
        fn embeddings_url_builds_correctly() {
            let client = OpenAI::new(OpenAIConfig::new("key")).unwrap();
            assert_eq!(
                client.embeddings_url(),
                "https://api.openai.com/v1/embeddings"
            );
        }

        #[test]
        fn custom_base_url_affects_all_endpoints() {
            let config = OpenAIConfig::new("key").with_base_url("https://azure.openai.com");
            let client = OpenAI::new(config).unwrap();
            assert!(client.chat_url().starts_with("https://azure.openai.com"));
            assert!(
                client
                    .embeddings_url()
                    .starts_with("https://azure.openai.com")
            );
        }
    }

    mod message_conversion {
        use super::*;

        #[test]
        fn converts_user_message() {
            let msg = Message::user("Hello!");
            let converted = OpenAI::convert_message(&msg);

            assert_eq!(converted.role, "user");
            assert!(matches!(converted.content, Some(OpenAIContent::Text(ref t)) if t == "Hello!"));
            assert!(converted.tool_calls.is_none());
            assert!(converted.tool_call_id.is_none());
        }

        #[test]
        fn converts_system_message() {
            let msg = Message::system("You are helpful.");
            let converted = OpenAI::convert_message(&msg);
            assert_eq!(converted.role, "system");
        }

        #[test]
        fn converts_assistant_message() {
            let msg = Message::assistant("I can help!");
            let converted = OpenAI::convert_message(&msg);
            assert_eq!(converted.role, "assistant");
        }

        #[test]
        fn converts_tool_message() {
            let msg = Message::tool("call_123", r#"{"result": "success"}"#);
            let converted = OpenAI::convert_message(&msg);

            assert_eq!(converted.role, "tool");
            assert_eq!(converted.tool_call_id, Some("call_123".to_owned()));
        }

        #[test]
        fn converts_developer_role() {
            let msg = Message::new(Role::Developer, Content::text("Developer message"));
            let converted = OpenAI::convert_message(&msg);
            assert_eq!(converted.role, "developer");
        }

        #[test]
        fn converts_multipart_content() {
            let parts = vec![
                ContentPart::text("Look at this image:"),
                ContentPart::image_url("https://example.com/image.jpg"),
            ];
            let msg = Message::new(Role::User, Content::Parts(parts));
            let converted = OpenAI::convert_message(&msg);

            if let Some(OpenAIContent::Array(parts)) = converted.content {
                assert_eq!(parts.len(), 2);
                assert!(
                    matches!(&parts[0], OpenAIContentPart::Text { text } if text == "Look at this image:")
                );
                assert!(matches!(&parts[1], OpenAIContentPart::ImageUrl { .. }));
            } else {
                panic!("Expected Array content");
            }
        }

        #[test]
        fn converts_image_with_detail() {
            let parts = vec![ContentPart::image_url_with_detail(
                "https://example.com/img.png",
                ImageDetail::High,
            )];
            let msg = Message::new(Role::User, Content::Parts(parts));
            let converted = OpenAI::convert_message(&msg);

            if let Some(OpenAIContent::Array(parts)) = converted.content {
                if let OpenAIContentPart::ImageUrl { image_url } = &parts[0] {
                    assert_eq!(image_url.detail, Some("high".to_owned()));
                } else {
                    panic!("Expected ImageUrl part");
                }
            }
        }

        #[test]
        fn converts_image_bytes_to_data_url() {
            // PNG magic bytes
            let png_bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            let parts = vec![ContentPart::image_bytes(&png_bytes, ImageMime::Png)];
            let msg = Message::new(Role::User, Content::Parts(parts));
            let converted = OpenAI::convert_message(&msg);

            if let Some(OpenAIContent::Array(parts)) = converted.content
                && let OpenAIContentPart::ImageUrl { image_url } = &parts[0] {
                    assert!(image_url.url.starts_with("data:image/png;base64,"));
                }
        }

        #[test]
        fn converts_assistant_with_tool_calls() {
            let tool_calls = vec![ToolCall::function(
                "call_abc",
                "get_weather",
                r#"{"city": "Tokyo"}"#,
            )];
            let msg = Message::assistant_tool_calls(tool_calls);
            let converted = OpenAI::convert_message(&msg);

            assert_eq!(converted.role, "assistant");
            let tc = converted.tool_calls.expect("Should have tool_calls");
            assert_eq!(tc.len(), 1);
            assert_eq!(tc[0].id, "call_abc");
            assert_eq!(tc[0].call_type, "function");
            assert_eq!(tc[0].function.name, "get_weather");
            assert_eq!(tc[0].function.arguments, r#"{"city": "Tokyo"}"#);
        }

        #[test]
        fn converts_message_with_name() {
            let msg = Message::user("Hello").with_name("Alice");
            let converted = OpenAI::convert_message(&msg);
            assert_eq!(converted.name, Some("Alice".to_owned()));
        }

        #[test]
        fn handles_message_without_content() {
            let msg = Message::assistant_tool_calls(vec![ToolCall::function("id", "fn", "{}")]);
            let converted = OpenAI::convert_message(&msg);
            assert!(converted.content.is_none());
        }
    }

    mod tool_conversion {
        use super::*;

        #[test]
        fn converts_basic_tool() {
            let tool = ToolDefinition::new(
                "test_tool",
                "A test tool",
                serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            );

            let converted = OpenAI::convert_tool(&tool);
            assert_eq!(converted.tool_type, "function");
            assert_eq!(converted.function.name, "test_tool");
            assert_eq!(converted.function.description, "A test tool");
        }

        #[test]
        fn converts_tool_with_parameters() {
            let params = serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["city"]
            });
            let tool = ToolDefinition::new("get_weather", "Get weather for city", params.clone());
            let converted = OpenAI::convert_tool(&tool);

            assert_eq!(converted.function.parameters, params);
        }

        #[test]
        fn converts_strict_tool() {
            let tool = ToolDefinition::new_strict(
                "strict_tool",
                "Strict mode tool",
                serde_json::json!({"type": "object"}),
            );
            let converted = OpenAI::convert_tool(&tool);
            assert_eq!(converted.function.strict, Some(true));
        }

        #[test]
        fn non_strict_tool_has_none_strict() {
            let tool = ToolDefinition::new("tool", "desc", serde_json::json!({}));
            let converted = OpenAI::convert_tool(&tool);
            assert!(converted.function.strict.is_none());
        }
    }

    mod build_body {
        use super::*;

        fn test_client() -> OpenAI {
            OpenAI::new(OpenAIConfig::new("test-key")).unwrap()
        }

        #[test]
        fn uses_request_model_when_provided() {
            let client = test_client();
            let request = ChatRequest::new("gpt-4-turbo").user("Hi");
            let body = client.build_body(&request);
            assert_eq!(body.model, "gpt-4-turbo");
        }

        #[test]
        fn falls_back_to_config_model_when_empty() {
            let config = OpenAIConfig::new("key").with_model("gpt-4o-mini");
            let client = OpenAI::new(config).unwrap();
            let request = ChatRequest::new("").user("Hi");
            let body = client.build_body(&request);
            assert_eq!(body.model, "gpt-4o-mini");
        }

        #[test]
        fn converts_all_messages() {
            let client = test_client();
            let request = ChatRequest::new("model")
                .system("Be helpful")
                .user("Hello")
                .assistant("Hi there");
            let body = client.build_body(&request);

            assert_eq!(body.messages.len(), 3);
            assert_eq!(body.messages[0].role, "system");
            assert_eq!(body.messages[1].role, "user");
            assert_eq!(body.messages[2].role, "assistant");
        }

        #[test]
        fn sets_temperature() {
            let client = test_client();
            let request = ChatRequest::new("model").temperature(0.7);
            let body = client.build_body(&request);
            assert_eq!(body.temperature, Some(0.7));
        }

        #[test]
        fn sets_top_p() {
            let client = test_client();
            let request = ChatRequest::new("model").top_p(0.9);
            let body = client.build_body(&request);
            assert_eq!(body.top_p, Some(0.9));
        }

        #[test]
        fn prefers_max_completion_tokens_over_max_tokens() {
            let client = test_client();
            let request = ChatRequest::new("model")
                .max_tokens(100)
                .max_completion_tokens(200);
            let body = client.build_body(&request);

            // max_completion_tokens should be used, max_tokens should be None
            assert!(body.max_tokens.is_none());
            assert_eq!(body.max_completion_tokens, Some(200));
        }

        #[test]
        fn uses_max_tokens_when_no_completion_tokens() {
            let client = test_client();
            let request = ChatRequest::new("model").max_tokens(100);
            let body = client.build_body(&request);

            assert_eq!(body.max_tokens, Some(100));
            assert!(body.max_completion_tokens.is_none());
        }

        #[test]
        fn sets_stop_sequences() {
            let client = test_client();
            let request = ChatRequest::new("model").stop(vec!["END".into(), "STOP".into()]);
            let body = client.build_body(&request);

            let stop = body.stop.unwrap();
            assert_eq!(stop.len(), 2);
            assert!(stop.contains(&"END".to_owned()));
        }

        #[test]
        fn sets_frequency_and_presence_penalty() {
            let client = test_client();
            let request = ChatRequest::new("model")
                .frequency_penalty(0.5)
                .presence_penalty(-0.5);
            let body = client.build_body(&request);

            assert_eq!(body.frequency_penalty, Some(0.5));
            assert_eq!(body.presence_penalty, Some(-0.5));
        }

        #[test]
        fn converts_tools() {
            let client = test_client();
            let tools = vec![ToolDefinition::new(
                "search",
                "Search the web",
                serde_json::json!({"type": "object"}),
            )];
            let request = ChatRequest::new("model").tools(tools);
            let body = client.build_body(&request);

            let tools = body.tools.unwrap();
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].function.name, "search");
        }

        #[test]
        fn sets_tool_choice() {
            let client = test_client();
            let request = ChatRequest::new("model").tool_choice("required");
            let body = client.build_body(&request);

            assert!(body.tool_choice.is_some());
        }

        #[test]
        fn sets_parallel_tool_calls() {
            let client = test_client();
            let request = ChatRequest::new("model").parallel_tool_calls(false);
            let body = client.build_body(&request);
            assert_eq!(body.parallel_tool_calls, Some(false));
        }

        #[test]
        fn stream_enables_stream_options() {
            let client = test_client();
            let request = ChatRequest::new("model").stream();
            let body = client.build_body(&request);

            assert!(body.stream);
            let opts = body.stream_options.unwrap();
            assert!(opts.include_usage);
        }

        #[test]
        fn non_stream_has_no_stream_options() {
            let client = test_client();
            let request = ChatRequest::new("model");
            let body = client.build_body(&request);

            assert!(!body.stream);
            assert!(body.stream_options.is_none());
        }

        #[test]
        fn sets_reasoning_effort() {
            let client = test_client();
            let request =
                ChatRequest::new("o1").reasoning_effort(crate::chat::ReasoningEffort::High);
            let body = client.build_body(&request);
            assert_eq!(body.reasoning_effort, Some("high".to_owned()));
        }

        #[test]
        fn sets_service_tier() {
            let client = test_client();
            let request = ChatRequest::new("model").service_tier("flex");
            let body = client.build_body(&request);
            assert_eq!(body.service_tier, Some("flex".to_owned()));
        }

        #[test]
        fn sets_user_id() {
            let client = test_client();
            let request = ChatRequest::new("model").user_id("user-123");
            let body = client.build_body(&request);
            assert_eq!(body.user, Some("user-123".to_owned()));
        }
    }

    mod response_format {
        use super::*;

        #[test]
        fn text_format_serializes_correctly() {
            let fmt = OpenAIResponseFormat::Text;
            let json = serde_json::to_value(&fmt).unwrap();
            assert_eq!(json["type"], "text");
        }

        #[test]
        fn json_object_format_serializes_correctly() {
            let fmt = OpenAIResponseFormat::JsonObject;
            let json = serde_json::to_value(&fmt).unwrap();
            assert_eq!(json["type"], "json_object");
        }

        #[test]
        fn json_schema_format_serializes_correctly() {
            let schema = serde_json::json!({
                "type": "object",
                "properties": {"name": {"type": "string"}}
            });
            let fmt = OpenAIResponseFormat::JsonSchema {
                json_schema: serde_json::json!({
                    "name": "Person",
                    "schema": schema,
                    "strict": true
                }),
            };
            let json = serde_json::to_value(&fmt).unwrap();
            assert_eq!(json["type"], "json_schema");
            assert!(json["json_schema"].is_object());
        }

        #[test]
        fn from_response_format_converts_text() {
            let fmt = crate::chat::ResponseFormat::Text;
            let openai_fmt = OpenAIResponseFormat::from_response_format(&fmt);
            assert!(matches!(openai_fmt, OpenAIResponseFormat::Text));
        }

        #[test]
        fn from_response_format_converts_json_object() {
            let fmt = crate::chat::ResponseFormat::JsonObject;
            let openai_fmt = OpenAIResponseFormat::from_response_format(&fmt);
            assert!(matches!(openai_fmt, OpenAIResponseFormat::JsonObject));
        }

        #[test]
        fn from_response_format_converts_json_schema() {
            let schema = serde_json::json!({"type": "object"});
            let fmt = crate::chat::ResponseFormat::json_schema("TestSchema", schema);
            let openai_fmt = OpenAIResponseFormat::from_response_format(&fmt);

            if let OpenAIResponseFormat::JsonSchema { json_schema } = openai_fmt {
                assert_eq!(json_schema["name"], "TestSchema");
                assert!(json_schema["strict"].as_bool().unwrap_or(false));
            } else {
                panic!("Expected JsonSchema variant");
            }
        }
    }

    mod error_parsing {
        use super::*;

        #[test]
        fn parses_401_as_auth_error() {
            let body = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
            let error = OpenAI::parse_error(401, body);

            // Should be an auth error
            let msg = error.to_string();
            assert!(msg.contains("Invalid API key") || msg.contains("auth"));
        }

        #[test]
        fn parses_429_as_rate_limit_error() {
            let body = r#"{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
            let error = OpenAI::parse_error(429, body);

            let msg = error.to_string();
            assert!(msg.to_lowercase().contains("rate") || msg.contains("limit"));
        }

        #[test]
        fn parses_context_length_error() {
            let body = r#"{"error":{"message":"This model's maximum context_length is 128000 tokens","type":"invalid_request_error","code":"context_length_exceeded"}}"#;
            let error = OpenAI::parse_error(400, body);

            let msg = error.to_string();
            // LlmError::context_exceeded produces "Context length exceeded: used X, max Y"
            assert!(msg.to_lowercase().contains("context"));
        }

        #[test]
        fn parses_generic_error_with_code() {
            let body = r#"{"error":{"message":"Something went wrong","type":"server_error","code":"internal_error"}}"#;
            let error = OpenAI::parse_error(500, body);

            let msg = error.to_string();
            assert!(msg.contains("Something went wrong") || msg.contains("internal"));
        }

        #[test]
        fn handles_malformed_error_response() {
            let body = "not valid json";
            let error = OpenAI::parse_error(500, body);

            // Should fall back to HTTP status error
            let msg = error.to_string();
            assert!(msg.contains("500") || msg.contains("not valid json"));
        }

        #[test]
        fn handles_empty_error_body() {
            let error = OpenAI::parse_error(500, "");
            // Should not panic and return some error
            assert!(!error.to_string().is_empty());
        }

        #[test]
        fn handles_error_without_code() {
            let body = r#"{"error":{"message":"Error message","type":"error_type"}}"#;
            let error = OpenAI::parse_error(400, body);

            // Should use type as fallback for code
            let msg = error.to_string();
            assert!(msg.contains("Error message") || msg.contains("error_type"));
        }
    }

    mod serialization {
        use super::*;

        #[test]
        fn chat_request_skips_none_fields() {
            let body = OpenAIChatRequest {
                model: "gpt-4o".to_owned(),
                messages: vec![],
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                frequency_penalty: None,
                presence_penalty: None,
                stop: None,
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                response_format: None,
                reasoning_effort: None,
                stream: false,
                stream_options: None,
                service_tier: None,
                user: None,
            };

            let json = serde_json::to_string(&body).unwrap();

            // Should not contain null/none optional fields
            assert!(!json.contains("max_tokens"));
            assert!(!json.contains("temperature"));
            assert!(!json.contains("tools"));
            assert!(json.contains("model"));
            assert!(json.contains("messages"));
        }

        #[test]
        fn message_content_text_serializes_as_string() {
            let msg = OpenAIMessage {
                role: "user".to_owned(),
                content: Some(OpenAIContent::Text("Hello".to_owned())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            };

            let json = serde_json::to_value(&msg).unwrap();
            assert_eq!(json["content"], "Hello");
        }

        #[test]
        fn message_content_array_serializes_as_array() {
            let msg = OpenAIMessage {
                role: "user".to_owned(),
                content: Some(OpenAIContent::Array(vec![OpenAIContentPart::Text {
                    text: "Hi".to_owned(),
                }])),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            };

            let json = serde_json::to_value(&msg).unwrap();
            assert!(json["content"].is_array());
        }

        #[test]
        fn tool_serializes_to_openai_format() {
            let tool = OpenAITool {
                tool_type: "function".to_owned(),
                function: OpenAIFunction {
                    name: "test".to_owned(),
                    description: "Test tool".to_owned(),
                    parameters: serde_json::json!({}),
                    strict: Some(true),
                },
            };

            let json = serde_json::to_value(&tool).unwrap();
            assert_eq!(json["type"], "function");
            assert_eq!(json["function"]["name"], "test");
            assert_eq!(json["function"]["strict"], true);
        }

        #[test]
        fn tool_call_serializes_correctly() {
            let tc = OpenAIToolCall {
                id: "call_123".to_owned(),
                call_type: "function".to_owned(),
                function: OpenAIFunctionCall {
                    name: "search".to_owned(),
                    arguments: r#"{"q":"test"}"#.to_owned(),
                },
            };

            let json = serde_json::to_value(&tc).unwrap();
            assert_eq!(json["id"], "call_123");
            assert_eq!(json["type"], "function");
            assert_eq!(json["function"]["name"], "search");
        }
    }

    mod deserialization {
        use super::*;

        #[test]
        fn deserializes_text_content() {
            let json = r#"{"role": "user", "content": "Hello"}"#;
            let msg: OpenAIMessage = serde_json::from_str(json).unwrap();

            assert_eq!(msg.role, "user");
            assert!(matches!(msg.content, Some(OpenAIContent::Text(ref t)) if t == "Hello"));
        }

        #[test]
        fn deserializes_array_content() {
            let json = r#"{"role": "user", "content": [{"type": "text", "text": "Hi"}]}"#;
            let msg: OpenAIMessage = serde_json::from_str(json).unwrap();

            if let Some(OpenAIContent::Array(parts)) = msg.content {
                assert_eq!(parts.len(), 1);
            } else {
                panic!("Expected Array content");
            }
        }

        #[test]
        fn deserializes_tool_call() {
            let json = r#"{"id": "call_abc", "type": "function", "function": {"name": "test", "arguments": "{}"}}"#;
            let tc: OpenAIToolCall = serde_json::from_str(json).unwrap();

            assert_eq!(tc.id, "call_abc");
            assert_eq!(tc.call_type, "function");
            assert_eq!(tc.function.name, "test");
        }

        #[test]
        fn deserializes_image_url_part() {
            let json = r#"{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg", "detail": "high"}}"#;
            let part: OpenAIContentPart = serde_json::from_str(json).unwrap();

            if let OpenAIContentPart::ImageUrl { image_url } = part {
                assert_eq!(image_url.url, "https://example.com/img.jpg");
                assert_eq!(image_url.detail, Some("high".to_owned()));
            } else {
                panic!("Expected ImageUrl part");
            }
        }

        #[test]
        fn deserializes_input_audio_part() {
            let json = r#"{"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}}"#;
            let part: OpenAIContentPart = serde_json::from_str(json).unwrap();

            if let OpenAIContentPart::InputAudio { input_audio } = part {
                assert_eq!(input_audio.data, "base64data");
                assert_eq!(input_audio.format, "wav");
            } else {
                panic!("Expected InputAudio part");
            }
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn full_request_body_serialization() {
            let client = OpenAI::new(OpenAIConfig::new("key")).unwrap();

            let tools = vec![ToolDefinition::new(
                "get_weather",
                "Get current weather",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }),
            )];

            let request = ChatRequest::new("gpt-4o")
                .system("You are a helpful assistant")
                .user("What's the weather in Tokyo?")
                .tools(tools)
                .temperature(0.7)
                .max_completion_tokens(1000)
                .stream();

            let body = client.build_body(&request);
            let json = serde_json::to_string(&body).unwrap();

            // Verify JSON is valid and contains expected fields
            let parsed: Value = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed["model"], "gpt-4o");
            assert!(parsed["messages"].is_array());
            assert_eq!(parsed["messages"].as_array().unwrap().len(), 2);
            assert_eq!(parsed["temperature"], 0.7);
            assert_eq!(parsed["max_completion_tokens"], 1000);
            assert!(parsed["stream"].as_bool().unwrap());
            assert!(parsed["tools"].is_array());
            assert!(parsed["stream_options"]["include_usage"].as_bool().unwrap());
        }

        #[test]
        fn tool_call_conversation_round_trip() {
            let client = OpenAI::new(OpenAIConfig::new("key")).unwrap();

            // User asks, assistant calls tool, tool responds, assistant answers
            let request = ChatRequest::new("gpt-4o")
                .user("What's the weather?")
                .message(Message::assistant_tool_calls(vec![ToolCall::function(
                    "call_123",
                    "get_weather",
                    r#"{"city":"Tokyo"}"#,
                )]))
                .message(Message::tool(
                    "call_123",
                    r#"{"temp":22,"condition":"sunny"}"#,
                ))
                .assistant("The weather in Tokyo is sunny at 22°C.");

            let body = client.build_body(&request);

            assert_eq!(body.messages.len(), 4);
            assert_eq!(body.messages[0].role, "user");
            assert_eq!(body.messages[1].role, "assistant");
            assert!(body.messages[1].tool_calls.is_some());
            assert_eq!(body.messages[2].role, "tool");
            assert_eq!(body.messages[2].tool_call_id, Some("call_123".to_owned()));
            assert_eq!(body.messages[3].role, "assistant");
        }
    }
}
