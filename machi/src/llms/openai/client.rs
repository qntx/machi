//! `OpenAI` API client implementation.

use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::config::OpenAIConfig;
use crate::chat::ChatRequest;
use crate::error::Result;
use crate::llms::LlmError;
use crate::message::{Content, ContentPart, Message, Role};
use crate::tool::ToolDefinition;

/// `OpenAI` chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    /// Deprecated: use `max_completion_tokens` instead.
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
    /// Number of completions to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Random seed for reproducibility (deprecated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Whether to return log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top log probabilities to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// Whether to store the completion for later retrieval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Metadata key-value pairs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// Stream options for `OpenAI`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// `OpenAI` message format.
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

/// `OpenAI` message content variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    Text(String),
    Array(Vec<OpenAIContentPart>),
}

/// `OpenAI` content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAIImageUrl },
    InputAudio { input_audio: OpenAIInputAudio },
}

/// `OpenAI` image URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// `OpenAI` input audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIInputAudio {
    pub data: String,
    pub format: String,
}

/// `OpenAI` tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunction,
}

/// `OpenAI` function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// `OpenAI` tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

/// `OpenAI` function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// `OpenAI` response format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Value },
}

impl OpenAIResponseFormat {
    /// Creates from our `ResponseFormat` type.
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

/// `OpenAI` error response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

/// `OpenAI` error details.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// `OpenAI` API client.
#[derive(Debug, Clone)]
pub struct OpenAI {
    pub(crate) config: Arc<OpenAIConfig>,
    pub(crate) client: Client,
}

impl OpenAI {
    /// Create a new `OpenAI` client with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is empty or the HTTP client fails to build.
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
    ///
    /// # Errors
    ///
    /// Returns an error if required environment variables are missing or the client fails to build.
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

    /// Convert Message to `OpenAI` format.
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

    /// Convert `ToolDefinition` to `OpenAI` format.
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
        let (max_tokens, max_completion_tokens) = request
            .max_completion_tokens
            .map_or((request.max_tokens, None), |tokens| (None, Some(tokens)));

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
            n: request.n,
            seed: request.seed,
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            store: request.store,
            metadata: request.metadata.clone(),
        }
    }

    /// Parse an error response from `OpenAI`.
    pub(crate) fn parse_error(status: u16, body: &str) -> LlmError {
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(body) {
            let error = error_response.error;
            let code = error.code.unwrap_or_else(|| error.error_type.clone());

            return match status {
                401 => LlmError::auth("openai", error.message),
                429 => LlmError::rate_limited("openai"),
                400 if error.message.contains("context_length") => {
                    // Attempt to extract token counts from the error message.
                    let (used, max) = parse_context_length_tokens(&error.message);
                    LlmError::context_exceeded(used, max)
                }
                _ => LlmError::provider_code("openai", code, error.message),
            };
        }

        LlmError::http_status(status, body.to_owned())
    }
}

/// Extract token counts from an `OpenAI` context-length error message.
///
/// `OpenAI` error messages typically look like:
/// "This model's maximum context length is 8192 tokens. However, your messages resulted in 9500 tokens."
/// Returns `(used, max)`, defaulting to `(0, 0)` if parsing fails.
fn parse_context_length_tokens(message: &str) -> (usize, usize) {
    let mut max = 0usize;
    let mut used = 0usize;

    // Look for "maximum context length is <N> tokens"
    if let Some(pos) = message.find("maximum context length is ") {
        let after = &message[pos + "maximum context length is ".len()..];
        if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
            max = after[..end].parse().unwrap_or(0);
        }
    }

    // Look for "resulted in <N> tokens" or "your messages resulted in <N> tokens"
    if let Some(pos) = message.find("resulted in ") {
        let after = &message[pos + "resulted in ".len()..];
        if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
            used = after[..end].parse().unwrap_or(0);
        }
    }

    (used, max)
}
