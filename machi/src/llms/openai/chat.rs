//! OpenAI ChatProvider implementation.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::Deserialize;

use tracing::{Instrument, debug, error, info, info_span};

use crate::chat::ChatProvider;
use crate::chat::{ChatRequest, ChatResponse};
use crate::error::Result;
use crate::llms::LlmError;
use crate::message::{Content, Role, ToolCall as MsgToolCall};
use crate::stream::{StopReason, StreamChunk};
use crate::usage::Usage;

use super::client::{OpenAI, OpenAIToolCall, StreamOptions};
use super::stream::parse_sse_events;

/// OpenAI chat completion response.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Service tier used for processing.
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// OpenAI response choice.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIChoice {
    pub message: OpenAIResponseMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI response message.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIResponseMessage {
    pub content: Option<String>,
    /// Refusal message if the model declined to respond.
    #[serde(default)]
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

impl OpenAI {
    /// Parse the response into ChatResponse.
    fn parse_response(response: OpenAIChatResponse) -> Result<ChatResponse> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::response_format("at least one choice", "empty choices"))?;

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("length") => StopReason::Length,
            Some("tool_calls") => StopReason::ToolCalls,
            Some("content_filter") => StopReason::ContentFilter,
            // "stop", None, and any other value defaults to Stop
            _ => StopReason::Stop,
        };

        let tool_calls = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|tc| MsgToolCall::function(tc.id, tc.function.name, tc.function.arguments))
                .collect()
        });

        let content = choice.message.content.map(Content::Text);

        let message = crate::message::Message {
            role: Role::Assistant,
            content,
            refusal: choice.message.refusal,
            annotations: Vec::new(),
            tool_calls,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            thinking_blocks: None,
        };

        Ok(ChatResponse {
            message,
            stop_reason,
            usage: response.usage,
            model: Some(response.model),
            id: Some(response.id),
            service_tier: response.service_tier,
            raw: None,
        })
    }
}

#[async_trait]
impl ChatProvider for OpenAI {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let span = info_span!(
            "gen_ai.chat",
            gen_ai.system = "openai",
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
            let body = self.build_body(request);

            debug!(model = %request.model, messages = request.messages.len(), "Sending OpenAI chat request");

            let response = self.build_request(&url).json(&body).send().await?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                let err = Self::parse_error(status.as_u16(), &error_text);
                error!(error = %err, status = status.as_u16(), "OpenAI API error");
                tracing::Span::current().record("error", tracing::field::display(&err));
                return Err(err.into());
            }

            let response_text = response.text().await?;
            let parsed: OpenAIChatResponse = serde_json::from_str(&response_text).map_err(|e| {
                let err = LlmError::response_format(
                    "valid OpenAI response",
                    format!("parse error: {e}, response: {response_text}"),
                );
                error!(error = %err, "OpenAI response parse error");
                tracing::Span::current().record("error", tracing::field::display(&err));
                err
            })?;

            let result = Self::parse_response(parsed)?;

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
                "OpenAI chat completed",
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
            gen_ai.system = "openai",
            model = %request.model,
            messages = request.messages.len(),
            "Starting OpenAI chat stream",
        );

        let url = self.chat_url();
        let mut body = self.build_body(request);
        body.stream = true;
        body.stream_options = Some(StreamOptions {
            include_usage: true,
        });

        let response = self.build_request(&url).json(&body).send().await?;

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
                    parse_sse_events(&text)
                }
                Err(e) => vec![Err(LlmError::stream(e.to_string()).into())],
            };
            futures::stream::iter(chunks)
        });

        Ok(Box::pin(parsed_stream))
    }

    fn provider_name(&self) -> &'static str {
        "openai"
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
    use crate::llms::openai::OpenAIConfig;

    mod openai_chat_response {
        use super::*;

        #[test]
        fn deserializes_minimal_response() {
            let json = r#"{
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop"
                }]
            }"#;

            let response: OpenAIChatResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.id, "chatcmpl-123");
            assert_eq!(response.model, "gpt-4o");
            assert_eq!(response.choices.len(), 1);
        }

        #[test]
        fn deserializes_with_usage() {
            let json = r#"{
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{
                    "message": {"content": "Hi"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }"#;

            let response: OpenAIChatResponse = serde_json::from_str(json).unwrap();

            let usage = response.usage.unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
            assert_eq!(usage.total_tokens, 15);
        }

        #[test]
        fn deserializes_with_service_tier() {
            let json = r#"{
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{
                    "message": {"content": "Hi"},
                    "finish_reason": "stop"
                }],
                "service_tier": "scale"
            }"#;

            let response: OpenAIChatResponse = serde_json::from_str(json).unwrap();

            assert_eq!(response.service_tier, Some("scale".to_owned()));
        }

        #[test]
        fn handles_missing_optional_fields() {
            let json = r#"{
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [{
                    "message": {"content": "Hi"},
                    "finish_reason": "stop"
                }]
            }"#;

            let response: OpenAIChatResponse = serde_json::from_str(json).unwrap();

            assert!(response.usage.is_none());
            assert!(response.service_tier.is_none());
        }
    }

    mod openai_choice {
        use super::*;

        #[test]
        fn deserializes_with_content() {
            let json = r#"{
                "message": {"content": "Hello world"},
                "finish_reason": "stop"
            }"#;

            let choice: OpenAIChoice = serde_json::from_str(json).unwrap();

            assert_eq!(choice.message.content, Some("Hello world".to_owned()));
            assert_eq!(choice.finish_reason, Some("stop".to_owned()));
        }

        #[test]
        fn deserializes_with_tool_calls() {
            let json = r#"{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }"#;

            let choice: OpenAIChoice = serde_json::from_str(json).unwrap();

            assert!(choice.message.content.is_none());
            let tool_calls = choice.message.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].id, "call_123");
        }

        #[test]
        fn handles_null_finish_reason() {
            let json = r#"{
                "message": {"content": "Partial"},
                "finish_reason": null
            }"#;

            let choice: OpenAIChoice = serde_json::from_str(json).unwrap();

            assert!(choice.finish_reason.is_none());
        }
    }

    mod openai_response_message {
        use super::*;

        #[test]
        fn deserializes_text_content() {
            let json = r#"{"content": "Hello!"}"#;
            let msg: OpenAIResponseMessage = serde_json::from_str(json).unwrap();

            assert_eq!(msg.content, Some("Hello!".to_owned()));
        }

        #[test]
        fn deserializes_refusal() {
            let json = r#"{"content": null, "refusal": "I cannot help with that"}"#;
            let msg: OpenAIResponseMessage = serde_json::from_str(json).unwrap();

            assert!(msg.content.is_none());
            assert_eq!(msg.refusal, Some("I cannot help with that".to_owned()));
        }

        #[test]
        fn deserializes_tool_calls() {
            let json = r#"{
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"}
                }]
            }"#;

            let msg: OpenAIResponseMessage = serde_json::from_str(json).unwrap();

            let tool_calls = msg.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].function.name, "search");
        }

        #[test]
        fn handles_empty_message() {
            let json = r"{}";
            let msg: OpenAIResponseMessage = serde_json::from_str(json).unwrap();

            assert!(msg.content.is_none());
            assert!(msg.refusal.is_none());
            assert!(msg.tool_calls.is_none());
        }
    }

    mod parse_response {
        use super::*;

        fn make_response(content: &str, finish_reason: &str) -> OpenAIChatResponse {
            OpenAIChatResponse {
                id: "test-id".to_owned(),
                model: "gpt-4o".to_owned(),
                choices: vec![OpenAIChoice {
                    message: OpenAIResponseMessage {
                        content: Some(content.to_owned()),
                        refusal: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(finish_reason.to_owned()),
                }],
                usage: None,
                service_tier: None,
            }
        }

        #[test]
        fn parses_text_response() {
            let response = make_response("Hello!", "stop");
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(
                result.message.content,
                Some(Content::Text("Hello!".to_owned()))
            );
            assert_eq!(result.stop_reason, StopReason::Stop);
        }

        #[test]
        fn parses_stop_finish_reason() {
            let response = make_response("Done", "stop");
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.stop_reason, StopReason::Stop);
        }

        #[test]
        fn parses_length_finish_reason() {
            let response = make_response("Truncated...", "length");
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.stop_reason, StopReason::Length);
        }

        #[test]
        fn parses_tool_calls_finish_reason() {
            let response = OpenAIChatResponse {
                id: "test".to_owned(),
                model: "gpt-4o".to_owned(),
                choices: vec![OpenAIChoice {
                    message: OpenAIResponseMessage {
                        content: None,
                        refusal: None,
                        tool_calls: Some(vec![OpenAIToolCall {
                            id: "call_123".to_owned(),
                            call_type: "function".to_owned(),
                            function: crate::llms::openai::client::OpenAIFunctionCall {
                                name: "test".to_owned(),
                                arguments: "{}".to_owned(),
                            },
                        }]),
                    },
                    finish_reason: Some("tool_calls".to_owned()),
                }],
                usage: None,
                service_tier: None,
            };

            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.stop_reason, StopReason::ToolCalls);
            assert!(result.message.tool_calls.is_some());
        }

        #[test]
        fn parses_content_filter_finish_reason() {
            let response = make_response("", "content_filter");
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.stop_reason, StopReason::ContentFilter);
        }

        #[test]
        fn defaults_unknown_finish_reason_to_stop() {
            let response = make_response("Hi", "unknown_reason");
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.stop_reason, StopReason::Stop);
        }

        #[test]
        fn defaults_none_finish_reason_to_stop() {
            let response = OpenAIChatResponse {
                id: "test".to_owned(),
                model: "gpt-4o".to_owned(),
                choices: vec![OpenAIChoice {
                    message: OpenAIResponseMessage {
                        content: Some("Hi".to_owned()),
                        refusal: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
                service_tier: None,
            };

            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.stop_reason, StopReason::Stop);
        }

        #[test]
        fn includes_model_in_response() {
            let response = make_response("Hi", "stop");
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.model, Some("gpt-4o".to_owned()));
        }

        #[test]
        fn includes_id_in_response() {
            let response = make_response("Hi", "stop");
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.id, Some("test-id".to_owned()));
        }

        #[test]
        fn includes_usage_when_present() {
            let mut response = make_response("Hi", "stop");
            response.usage = Some(Usage::new(10, 5));

            let result = OpenAI::parse_response(response).unwrap();

            let usage = result.usage.unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
        }

        #[test]
        fn includes_service_tier_when_present() {
            let mut response = make_response("Hi", "stop");
            response.service_tier = Some("default".to_owned());

            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(result.service_tier, Some("default".to_owned()));
        }

        #[test]
        fn includes_refusal_when_present() {
            let response = OpenAIChatResponse {
                id: "test".to_owned(),
                model: "gpt-4o".to_owned(),
                choices: vec![OpenAIChoice {
                    message: OpenAIResponseMessage {
                        content: None,
                        refusal: Some("I cannot assist with that request.".to_owned()),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_owned()),
                }],
                usage: None,
                service_tier: None,
            };

            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(
                result.message.refusal,
                Some("I cannot assist with that request.".to_owned())
            );
        }

        #[test]
        fn errors_on_empty_choices() {
            let response = OpenAIChatResponse {
                id: "test".to_owned(),
                model: "gpt-4o".to_owned(),
                choices: vec![],
                usage: None,
                service_tier: None,
            };

            let result = OpenAI::parse_response(response);

            assert!(result.is_err());
        }

        #[test]
        fn converts_tool_calls_correctly() {
            let response = OpenAIChatResponse {
                id: "test".to_owned(),
                model: "gpt-4o".to_owned(),
                choices: vec![OpenAIChoice {
                    message: OpenAIResponseMessage {
                        content: None,
                        refusal: None,
                        tool_calls: Some(vec![
                            OpenAIToolCall {
                                id: "call_1".to_owned(),
                                call_type: "function".to_owned(),
                                function: crate::llms::openai::client::OpenAIFunctionCall {
                                    name: "get_weather".to_owned(),
                                    arguments: r#"{"city":"Tokyo"}"#.to_owned(),
                                },
                            },
                            OpenAIToolCall {
                                id: "call_2".to_owned(),
                                call_type: "function".to_owned(),
                                function: crate::llms::openai::client::OpenAIFunctionCall {
                                    name: "get_time".to_owned(),
                                    arguments: r#"{"timezone":"JST"}"#.to_owned(),
                                },
                            },
                        ]),
                    },
                    finish_reason: Some("tool_calls".to_owned()),
                }],
                usage: None,
                service_tier: None,
            };

            let result = OpenAI::parse_response(response).unwrap();

            let tool_calls = result.message.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 2);
            assert_eq!(tool_calls[0].id, "call_1");
            assert_eq!(tool_calls[0].function.name, "get_weather");
            assert_eq!(tool_calls[1].id, "call_2");
            assert_eq!(tool_calls[1].function.name, "get_time");
        }
    }

    mod chat_provider_impl {
        use super::*;

        fn test_client() -> OpenAI {
            OpenAI::new(OpenAIConfig::new("test-key")).unwrap()
        }

        #[test]
        fn provider_name_is_openai() {
            let client = test_client();
            assert_eq!(client.provider_name(), "openai");
        }

        #[test]
        fn default_model_returns_config_model() {
            let config = OpenAIConfig::new("key").model("gpt-4-turbo");
            let client = OpenAI::new(config).unwrap();

            assert_eq!(client.default_model(), "gpt-4-turbo");
        }

        #[test]
        fn supports_streaming() {
            let client = test_client();
            assert!(client.supports_streaming());
        }

        #[test]
        fn supports_tools() {
            let client = test_client();
            assert!(client.supports_tools());
        }

        #[test]
        fn supports_vision() {
            let client = test_client();
            assert!(client.supports_vision());
        }

        #[test]
        fn supports_json_mode() {
            let client = test_client();
            assert!(client.supports_json_mode());
        }
    }

    mod realistic_responses {
        use super::*;

        #[test]
        fn parses_gpt4o_response() {
            let json = r#"{
                "id": "chatcmpl-AYqxL4Xqo9erJtSNrAhdDzKP6Weex",
                "object": "chat.completion",
                "created": 1732918041,
                "model": "gpt-4o-2024-08-06",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?",
                        "refusal": null
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 19,
                    "completion_tokens": 9,
                    "total_tokens": 28,
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "audio_tokens": 0
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                        "accepted_prediction_tokens": 0,
                        "rejected_prediction_tokens": 0
                    }
                },
                "service_tier": "default",
                "system_fingerprint": "fp_831e067d82"
            }"#;

            let response: OpenAIChatResponse = serde_json::from_str(json).unwrap();
            let result = OpenAI::parse_response(response).unwrap();

            assert_eq!(
                result.message.content,
                Some(Content::Text(
                    "Hello! How can I assist you today?".to_owned()
                ))
            );
            assert_eq!(result.stop_reason, StopReason::Stop);
            assert_eq!(result.model, Some("gpt-4o-2024-08-06".to_owned()));
            assert_eq!(result.service_tier, Some("default".to_owned()));

            let usage = result.usage.unwrap();
            assert_eq!(usage.input_tokens, 19);
            assert_eq!(usage.output_tokens, 9);
        }

        #[test]
        fn parses_tool_call_response() {
            let json = r#"{
                "id": "chatcmpl-xyz",
                "object": "chat.completion",
                "created": 1732918041,
                "model": "gpt-4o-2024-08-06",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_vKI6XCWp8QSTSsLAY0J0vUcH",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\":\"Tokyo\",\"unit\":\"celsius\"}"
                            }
                        }],
                        "refusal": null
                    },
                    "logprobs": null,
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125
                }
            }"#;

            let response: OpenAIChatResponse = serde_json::from_str(json).unwrap();
            let result = OpenAI::parse_response(response).unwrap();

            assert!(result.message.content.is_none());
            assert_eq!(result.stop_reason, StopReason::ToolCalls);

            let tool_calls = result.message.tool_calls.unwrap();
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].id, "call_vKI6XCWp8QSTSsLAY0J0vUcH");
            assert_eq!(tool_calls[0].function.name, "get_weather");
        }
    }
}
