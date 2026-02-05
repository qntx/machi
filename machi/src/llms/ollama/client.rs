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

    #[tokio::test]
    async fn test_message_conversion() {
        let client = Client::new();
        let msg = Message::user("Hello!");
        let converted = Ollama::convert_message_async(&client, &msg).await.unwrap();

        assert_eq!(converted.role, "user");
        assert_eq!(converted.content, "Hello!");
        assert!(converted.images.is_none());
    }

    #[test]
    fn test_tool_conversion() {
        let tool = ToolDefinition::new(
            "test_tool",
            "A test tool",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        );

        let converted = Ollama::convert_tool(&tool);
        assert_eq!(converted.function.name, "test_tool");
        assert_eq!(converted.tool_type, "function");
    }
}
