//! Ollama ChatProvider implementation.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::Deserialize;

use tracing::{Instrument, debug, error, info, info_span};

use crate::chat::ChatProvider;
use crate::chat::{ChatRequest, ChatResponse};
use crate::error::{LlmError, Result};
use crate::message::{Content, Message, Role, ToolCall};
use crate::stream::{StopReason, StreamChunk};
use crate::usage::Usage;

use super::client::{Ollama, OllamaToolCall};
use super::stream::parse_stream_line;

/// Ollama chat completion response.
#[derive(Debug, Clone, Deserialize)]
struct OllamaChatResponse {
    pub model: String,
    pub message: OllamaResponseMessage,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
}

/// Ollama response message.
#[derive(Debug, Clone, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default)]
    pub thinking: Option<String>,
}

impl Ollama {
    /// Parse the response into ChatResponse.
    fn parse_response(response: OllamaChatResponse) -> ChatResponse {
        let stop_reason = match response.done_reason.as_deref() {
            Some("length") => StopReason::Length,
            // "stop", None, and any other value defaults to Stop
            _ => StopReason::Stop,
        };

        let tool_calls = response.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| {
                    let args = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                    ToolCall::function(
                        format!("call_{}", uuid::Uuid::new_v4()),
                        tc.function.name,
                        args,
                    )
                })
                .collect()
        });

        let content = if response.message.content.is_empty() {
            None
        } else {
            Some(Content::Text(response.message.content))
        };

        // Extract thinking content from reasoning models
        let reasoning_content = response.message.thinking.filter(|t| !t.is_empty());

        let message = Message {
            role: Role::Assistant,
            content,
            refusal: None,
            annotations: Vec::new(),
            tool_calls,
            tool_call_id: None,
            name: None,
            reasoning_content,
            thinking_blocks: None,
        };

        let usage = match (response.prompt_eval_count, response.eval_count) {
            (Some(input), Some(output)) => Some(Usage::new(input, output)),
            _ => None,
        };

        ChatResponse {
            message,
            stop_reason,
            usage,
            model: Some(response.model),
            id: None,
            service_tier: None,
            raw: None,
        }
    }
}

#[async_trait]
impl ChatProvider for Ollama {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let span = info_span!(
            "gen_ai.chat",
            gen_ai.system = "ollama",
            gen_ai.request.model = %request.model,
            gen_ai.request.temperature = request.temperature.unwrap_or(-1.0),
            gen_ai.request.max_tokens = request.max_completion_tokens.or(request.max_tokens).unwrap_or(0),
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.response.finish_reason = tracing::field::Empty,
            error = tracing::field::Empty,
        );

        async {
            let url = self.chat_url();
            let mut body = self.build_body(request).await?;
            body.stream = false;

            debug!(model = %request.model, messages = request.messages.len(), "Sending Ollama chat request");

            let response = self.client().post(&url).json(&body).send().await?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                let err = Self::parse_error(status.as_u16(), &error_text);
                error!(error = %err, status = status.as_u16(), "Ollama API error");
                tracing::Span::current().record("error", tracing::field::display(&err));
                return Err(err.into());
            }

            let response_text = response.text().await?;
            let parsed: OllamaChatResponse = serde_json::from_str(&response_text).map_err(|e| {
                let err = LlmError::response_format(
                    "valid Ollama response",
                    format!("parse error: {e}, response: {response_text}"),
                );
                error!(error = %err, "Ollama response parse error");
                tracing::Span::current().record("error", tracing::field::display(&err));
                err
            })?;

            let result = Self::parse_response(parsed);

            // Record usage, model, and finish_reason in the span.
            let current = tracing::Span::current();
            if let Some(ref usage) = result.usage {
                current.record("gen_ai.usage.input_tokens", usage.input_tokens);
                current.record("gen_ai.usage.output_tokens", usage.output_tokens);
            }
            if let Some(ref model) = result.model {
                current.record("gen_ai.response.model", model.as_str());
            }
            current.record("gen_ai.response.finish_reason", result.stop_reason.as_str());

            info!(
                model = result.model.as_deref().unwrap_or(&request.model),
                finish_reason = result.stop_reason.as_str(),
                "Ollama chat completed",
            );

            Ok(result)
        }
        .instrument(span)
        .await
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        debug!(
            gen_ai.system = "ollama",
            model = %request.model,
            messages = request.messages.len(),
            "Starting Ollama chat stream",
        );

        let url = self.chat_url();
        let mut body = self.build_body(request).await?;
        body.stream = true;

        let response = self.client().post(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status.as_u16(), &error_text).into());
        }

        let stream = response.bytes_stream();
        let parsed_stream = stream.flat_map(move |chunk_result| {
            let chunks: Vec<Result<StreamChunk>> = match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    text.lines().filter_map(parse_stream_line).collect()
                }
                Err(e) => vec![Err(LlmError::stream(e.to_string()).into())],
            };
            futures::stream::iter(chunks)
        });

        Ok(Box::pin(parsed_stream))
    }

    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    fn default_model(&self) -> &str {
        self.model()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    mod ollama_chat_response {
        use super::*;

        #[test]
        fn deserializes_basic_response() {
            let json = r#"{
                "model": "llama3",
                "message": {
                    "content": "Hello!"
                },
                "done_reason": "stop"
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.model, "llama3");
            assert_eq!(response.message.content, "Hello!");
            assert_eq!(response.done_reason, Some("stop".to_owned()));
        }

        #[test]
        fn deserializes_with_usage_info() {
            let json = r#"{
                "model": "llama3",
                "message": {"content": "Test"},
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 50
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.prompt_eval_count, Some(100));
            assert_eq!(response.eval_count, Some(50));
        }

        #[test]
        fn deserializes_with_tool_calls() {
            let json = r#"{
                "model": "llama3",
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"}
                        }
                    }]
                },
                "done_reason": "stop"
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();

            assert!(response.message.tool_calls.is_some());
            let tool_calls = response.message.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].function.name, "get_weather");
        }

        #[test]
        fn deserializes_with_thinking() {
            let json = r#"{
                "model": "qwen3",
                "message": {
                    "content": "The answer is 42.",
                    "thinking": "Let me calculate this..."
                },
                "done_reason": "stop"
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();

            assert_eq!(
                response.message.thinking,
                Some("Let me calculate this...".to_owned())
            );
        }

        #[test]
        fn deserializes_with_length_done_reason() {
            let json = r#"{
                "model": "llama3",
                "message": {"content": "Truncated..."},
                "done_reason": "length"
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.done_reason, Some("length".to_owned()));
        }

        #[test]
        fn deserializes_without_optional_fields() {
            let json = r#"{
                "model": "llama3",
                "message": {"content": "Hello"}
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();

            assert!(response.done_reason.is_none());
            assert!(response.prompt_eval_count.is_none());
            assert!(response.eval_count.is_none());
            assert!(response.message.tool_calls.is_none());
            assert!(response.message.thinking.is_none());
        }

        #[test]
        fn deserializes_empty_content() {
            let json = r#"{
                "model": "llama3",
                "message": {"content": ""}
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();

            assert!(response.message.content.is_empty());
        }
    }

    mod parse_response {
        use super::*;
        use crate::llms::ollama::client::OllamaFunctionCall;

        fn make_response(content: &str, done_reason: Option<&str>) -> OllamaChatResponse {
            OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: content.to_owned(),
                    tool_calls: None,
                    thinking: None,
                },
                done_reason: done_reason.map(String::from),
                prompt_eval_count: None,
                eval_count: None,
            }
        }

        #[test]
        fn parses_basic_text_response() {
            let response = make_response("Hello, world!", Some("stop"));
            let parsed = Ollama::parse_response(response);

            assert_eq!(parsed.message.role, Role::Assistant);
            assert!(parsed.message.content.is_some());
            if let Some(Content::Text(text)) = &parsed.message.content {
                assert_eq!(text, "Hello, world!");
            } else {
                panic!("Expected text content");
            }
        }

        #[test]
        fn parses_stop_reason_stop() {
            let response = make_response("Done", Some("stop"));
            let parsed = Ollama::parse_response(response);

            assert_eq!(parsed.stop_reason, StopReason::Stop);
        }

        #[test]
        fn parses_stop_reason_length() {
            let response = make_response("Truncated", Some("length"));
            let parsed = Ollama::parse_response(response);

            assert_eq!(parsed.stop_reason, StopReason::Length);
        }

        #[test]
        fn parses_stop_reason_none_defaults_to_stop() {
            let response = make_response("Done", None);
            let parsed = Ollama::parse_response(response);

            assert_eq!(parsed.stop_reason, StopReason::Stop);
        }

        #[test]
        fn parses_stop_reason_unknown_defaults_to_stop() {
            let response = make_response("Done", Some("unknown_reason"));
            let parsed = Ollama::parse_response(response);

            assert_eq!(parsed.stop_reason, StopReason::Stop);
        }

        #[test]
        fn parses_empty_content_as_none() {
            let response = make_response("", Some("stop"));
            let parsed = Ollama::parse_response(response);

            assert!(parsed.message.content.is_none());
        }

        #[test]
        fn parses_usage_info() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: "Test".to_owned(),
                    tool_calls: None,
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: Some(100),
                eval_count: Some(50),
            };

            let parsed = Ollama::parse_response(response);

            assert!(parsed.usage.is_some());
            let usage = parsed.usage.unwrap();
            assert_eq!(usage.input_tokens, 100);
            assert_eq!(usage.output_tokens, 50);
            assert_eq!(usage.total_tokens, 150);
        }

        #[test]
        fn parses_partial_usage_as_none() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: "Test".to_owned(),
                    tool_calls: None,
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: Some(100),
                eval_count: None, // Missing eval_count
            };

            let parsed = Ollama::parse_response(response);

            assert!(parsed.usage.is_none());
        }

        #[test]
        fn parses_tool_calls() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: String::new(),
                    tool_calls: Some(vec![OllamaToolCall {
                        function: OllamaFunctionCall {
                            name: "get_weather".to_owned(),
                            arguments: serde_json::json!({"city": "Tokyo"}),
                        },
                    }]),
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);

            assert!(parsed.message.tool_calls.is_some());
            let tool_calls = parsed.message.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].function.name, "get_weather");
            assert!(tool_calls[0].id.starts_with("call_"));
        }

        #[test]
        fn parses_multiple_tool_calls() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: String::new(),
                    tool_calls: Some(vec![
                        OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: "get_weather".to_owned(),
                                arguments: serde_json::json!({"city": "Tokyo"}),
                            },
                        },
                        OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: "get_time".to_owned(),
                                arguments: serde_json::json!({"timezone": "JST"}),
                            },
                        },
                    ]),
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);

            let tool_calls = parsed.message.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 2);
            assert_eq!(tool_calls[0].function.name, "get_weather");
            assert_eq!(tool_calls[1].function.name, "get_time");
        }

        #[test]
        fn parses_thinking_content() {
            let response = OllamaChatResponse {
                model: "qwen3".to_owned(),
                message: OllamaResponseMessage {
                    content: "The answer is 42.".to_owned(),
                    tool_calls: None,
                    thinking: Some("Let me think about this...".to_owned()),
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);

            assert_eq!(
                parsed.message.reasoning_content,
                Some("Let me think about this...".to_owned())
            );
        }

        #[test]
        fn parses_empty_thinking_as_none() {
            let response = OllamaChatResponse {
                model: "qwen3".to_owned(),
                message: OllamaResponseMessage {
                    content: "Answer".to_owned(),
                    tool_calls: None,
                    thinking: Some(String::new()),
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);

            assert!(parsed.message.reasoning_content.is_none());
        }

        #[test]
        fn includes_model_in_response() {
            let response = make_response("Test", Some("stop"));
            let parsed = Ollama::parse_response(response);

            assert_eq!(parsed.model, Some("llama3".to_owned()));
        }

        #[test]
        fn id_is_none() {
            let response = make_response("Test", Some("stop"));
            let parsed = Ollama::parse_response(response);

            // Ollama doesn't provide a response ID
            assert!(parsed.id.is_none());
        }

        #[test]
        fn service_tier_is_none() {
            let response = make_response("Test", Some("stop"));
            let parsed = Ollama::parse_response(response);

            assert!(parsed.service_tier.is_none());
        }

        #[test]
        fn raw_is_none() {
            let response = make_response("Test", Some("stop"));
            let parsed = Ollama::parse_response(response);

            assert!(parsed.raw.is_none());
        }
    }

    mod chat_provider_impl {
        use super::*;

        #[test]
        fn provider_name_is_ollama() {
            let client = Ollama::with_defaults().unwrap();
            assert_eq!(client.provider_name(), "ollama");
        }

        #[test]
        fn default_model_returns_config_model() {
            let client = Ollama::with_defaults().unwrap();
            assert_eq!(client.default_model(), client.model());
        }

        #[test]
        fn supports_streaming() {
            let client = Ollama::with_defaults().unwrap();
            assert!(client.supports_streaming());
        }

        #[test]
        fn supports_tools() {
            let client = Ollama::with_defaults().unwrap();
            assert!(client.supports_tools());
        }

        #[test]
        fn supports_vision() {
            let client = Ollama::with_defaults().unwrap();
            assert!(client.supports_vision());
        }

        #[test]
        fn supports_json_mode() {
            let client = Ollama::with_defaults().unwrap();
            assert!(client.supports_json_mode());
        }
    }

    mod tool_call_id_generation {
        use super::*;
        use crate::llms::ollama::client::OllamaFunctionCall;

        #[test]
        fn generates_unique_tool_call_ids() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: String::new(),
                    tool_calls: Some(vec![
                        OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: "tool1".to_owned(),
                                arguments: serde_json::json!({}),
                            },
                        },
                        OllamaToolCall {
                            function: OllamaFunctionCall {
                                name: "tool2".to_owned(),
                                arguments: serde_json::json!({}),
                            },
                        },
                    ]),
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);
            let tool_calls = parsed.message.tool_calls.unwrap();

            // Each tool call should have a unique ID
            assert_ne!(tool_calls[0].id, tool_calls[1].id);
        }

        #[test]
        fn tool_call_ids_have_call_prefix() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: String::new(),
                    tool_calls: Some(vec![OllamaToolCall {
                        function: OllamaFunctionCall {
                            name: "test".to_owned(),
                            arguments: serde_json::json!({}),
                        },
                    }]),
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);
            let tool_calls = parsed.message.tool_calls.unwrap();

            assert!(tool_calls[0].id.starts_with("call_"));
        }
    }

    mod tool_call_arguments {
        use super::*;
        use crate::llms::ollama::client::OllamaFunctionCall;

        #[test]
        fn serializes_object_arguments() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: String::new(),
                    tool_calls: Some(vec![OllamaToolCall {
                        function: OllamaFunctionCall {
                            name: "get_weather".to_owned(),
                            arguments: serde_json::json!({
                                "city": "Tokyo",
                                "units": "celsius"
                            }),
                        },
                    }]),
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);
            let tool_calls = parsed.message.tool_calls.unwrap();
            let args = &tool_calls[0].function.arguments;

            // Arguments should be serialized to JSON string
            assert!(args.contains("Tokyo"));
            assert!(args.contains("celsius"));
        }

        #[test]
        fn handles_empty_arguments() {
            let response = OllamaChatResponse {
                model: "llama3".to_owned(),
                message: OllamaResponseMessage {
                    content: String::new(),
                    tool_calls: Some(vec![OllamaToolCall {
                        function: OllamaFunctionCall {
                            name: "no_params_tool".to_owned(),
                            arguments: serde_json::json!({}),
                        },
                    }]),
                    thinking: None,
                },
                done_reason: Some("stop".to_owned()),
                prompt_eval_count: None,
                eval_count: None,
            };

            let parsed = Ollama::parse_response(response);
            let tool_calls = parsed.message.tool_calls.unwrap();
            let args = &tool_calls[0].function.arguments;

            assert_eq!(args, "{}");
        }
    }

    mod realistic_responses {
        use super::*;

        #[test]
        fn parses_typical_chat_response() {
            let json = r#"{
                "model": "llama3.2:latest",
                "created_at": "2024-01-15T10:30:00Z",
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris. It is known for the Eiffel Tower."
                },
                "done": true,
                "done_reason": "stop",
                "total_duration": 1234567890,
                "load_duration": 123456789,
                "prompt_eval_count": 15,
                "prompt_eval_duration": 12345678,
                "eval_count": 25,
                "eval_duration": 123456789
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
            let parsed = Ollama::parse_response(response);

            assert!(parsed.message.content.is_some());
            assert_eq!(parsed.stop_reason, StopReason::Stop);
            assert!(parsed.usage.is_some());
            let usage = parsed.usage.unwrap();
            assert_eq!(usage.input_tokens, 15);
            assert_eq!(usage.output_tokens, 25);
        }

        #[test]
        fn parses_tool_call_response() {
            let json = r#"{
                "model": "llama3.2:latest",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_current_weather",
                                "arguments": {
                                    "location": "San Francisco, CA",
                                    "format": "fahrenheit"
                                }
                            }
                        }
                    ]
                },
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 20
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
            let parsed = Ollama::parse_response(response);

            assert!(parsed.message.content.is_none());
            assert!(parsed.message.tool_calls.is_some());
            let tool_calls = parsed.message.tool_calls.unwrap();
            assert_eq!(tool_calls[0].function.name, "get_current_weather");
        }

        #[test]
        fn parses_reasoning_model_response() {
            let json = r#"{
                "model": "qwen3:thinking",
                "message": {
                    "role": "assistant",
                    "content": "Based on my analysis, the answer is 42.",
                    "thinking": "Let me break this down step by step:\n1. First, I need to consider...\n2. Then, applying the formula..."
                },
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 50,
                "eval_count": 150
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
            let parsed = Ollama::parse_response(response);

            assert!(parsed.message.content.is_some());
            assert!(parsed.message.reasoning_content.is_some());
            assert!(
                parsed
                    .message
                    .reasoning_content
                    .unwrap()
                    .contains("step by step")
            );
        }

        #[test]
        fn parses_truncated_response() {
            let json = r#"{
                "model": "llama3.2:latest",
                "message": {
                    "role": "assistant",
                    "content": "This is a very long response that was truncated because it reached the maximum token limit. The response continues to explain in great detail about the topic but unfortunately the..."
                },
                "done": true,
                "done_reason": "length",
                "prompt_eval_count": 20,
                "eval_count": 4096
            }"#;

            let response: OllamaChatResponse = serde_json::from_str(json).unwrap();
            let parsed = Ollama::parse_response(response);

            assert_eq!(parsed.stop_reason, StopReason::Length);
            assert!(parsed.usage.is_some());
            assert_eq!(parsed.usage.unwrap().output_tokens, 4096);
        }
    }
}
