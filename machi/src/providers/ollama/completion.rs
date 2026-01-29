//! Ollama completion model implementation

use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage, Usage};
use crate::http_client::{self, HttpClientExt};
use crate::streaming::{self, RawStreamingChoice};
use crate::{json_utils, message, OneOrMany};
use async_stream::try_stream;
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info_span;
use tracing_futures::Instrument;

use super::client::Client;
use super::message::Message;

// ---------- Completion Constants ----------

pub const LLAMA3_2: &str = "llama3.2";
pub const LLAVA: &str = "llava";
pub const MISTRAL: &str = "mistral";

// ---------- Completion Response ----------

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;
    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        match resp.message {
            Message::Assistant {
                content,
                thinking,
                tool_calls,
                ..
            } => {
                let mut assistant_contents = Vec::new();
                if !content.is_empty() {
                    assistant_contents.push(completion::AssistantContent::text(&content));
                }
                for tc in tool_calls.iter() {
                    assistant_contents.push(completion::AssistantContent::tool_call(
                        tc.function.name.clone(),
                        tc.function.name.clone(),
                        tc.function.arguments.clone(),
                    ));
                }
                let choice = OneOrMany::many(assistant_contents).map_err(|_| {
                    CompletionError::ResponseError("No content provided".to_owned())
                })?;
                let prompt_tokens = resp.prompt_eval_count.unwrap_or(0);
                let completion_tokens = resp.eval_count.unwrap_or(0);

                let raw_response = CompletionResponse {
                    model: resp.model,
                    created_at: resp.created_at,
                    done: resp.done,
                    done_reason: resp.done_reason,
                    total_duration: resp.total_duration,
                    load_duration: resp.load_duration,
                    prompt_eval_count: resp.prompt_eval_count,
                    prompt_eval_duration: resp.prompt_eval_duration,
                    eval_count: resp.eval_count,
                    eval_duration: resp.eval_duration,
                    message: Message::Assistant {
                        content,
                        thinking,
                        images: None,
                        name: None,
                        tool_calls,
                    },
                };

                Ok(completion::CompletionResponse {
                    choice,
                    usage: Usage {
                        input_tokens: prompt_tokens,
                        output_tokens: completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                    raw_response,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Chat response does not include an assistant message".into(),
            )),
        }
    }
}

// ---------- Completion Request ----------

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaCompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    pub stream: bool,
    think: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    options: serde_json::Value,
}

impl TryFrom<(&str, CompletionRequest)> for OllamaCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        if req.tool_choice.is_some() {
            tracing::warn!("WARNING: `tool_choice` not supported for Ollama");
        }
        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(req.chat_history);

        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let mut think = false;

        let options = if let Some(mut extra) = req.additional_params {
            if extra.get("think").is_some() {
                think = extra["think"].take().as_bool().ok_or_else(|| {
                    CompletionError::RequestError("`think` must be a bool".into())
                })?;
            }
            json_utils::merge(json!({ "temperature": req.temperature }), extra)
        } else {
            json!({ "temperature": req.temperature })
        };

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            stream: false,
            think,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            options,
        })
    }
}

// ---------- Completion Model ----------

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_owned(),
        }
    }
}

// ---------- Streaming Response ----------

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        let input_tokens = self.prompt_eval_count.unwrap_or_default();
        let output_tokens = self.eval_count.unwrap_or_default();
        usage.input_tokens = input_tokens;
        usage.output_tokens = output_tokens;
        usage.total_tokens = input_tokens + output_tokens;

        Some(usage)
    }
}

// ---------- CompletionModel Implementation ----------

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into().as_str())
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "crate::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "ollama",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);
        let request = OllamaCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "crate::completions",
                "Ollama completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("api/chat")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let async_block = async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ));
            }

            let response: CompletionResponse = serde_json::from_slice(&response_body)?;
            let span = tracing::Span::current();
            span.record("gen_ai.response.model_name", &response.model);
            span.record(
                "gen_ai.usage.input_tokens",
                response.prompt_eval_count.unwrap_or_default(),
            );
            span.record(
                "gen_ai.usage.output_tokens",
                response.eval_count.unwrap_or_default(),
            );

            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(target: "crate::completions",
                    "Ollama completion response: {}",
                    serde_json::to_string_pretty(&response)?
                );
            }

            let response: completion::CompletionResponse<CompletionResponse> =
                response.try_into()?;

            Ok(response)
        };

        tracing::Instrument::instrument(async_block, span).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "crate::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "ollama",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &request.preamble);

        let mut request = OllamaCompletionRequest::try_from((self.model.as_ref(), request))?;
        request.stream = true;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "crate::completions",
                "Ollama streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("api/chat")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self.client.send_streaming(req).await?;
        let status = response.status();
        let mut byte_stream = response.into_body();

        if !status.is_success() {
            return Err(CompletionError::ProviderError(format!(
                "Got error status code trying to send a request to Ollama: {status}"
            )));
        }

        let stream = try_stream! {
            let span = tracing::Span::current();
            let mut tool_calls_final = Vec::new();
            let mut text_response = String::new();
            let mut thinking_response = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| http_client::Error::Instance(e.into()))?;

                for line in bytes.split(|&b| b == b'\n') {
                    if line.is_empty() {
                        continue;
                    }

                    tracing::debug!(target: "machi", "Received NDJSON line from Ollama: {}", String::from_utf8_lossy(line));

                    let response: CompletionResponse = serde_json::from_slice(line)?;

                    if let Message::Assistant { content, thinking, tool_calls, .. } = response.message {
                        if let Some(thinking_content) = thinking && !thinking_content.is_empty() {
                            thinking_response += &thinking_content;
                            yield RawStreamingChoice::ReasoningDelta {
                                id: None,
                                reasoning: thinking_content,
                            };
                        }

                        if !content.is_empty() {
                            text_response += &content;
                            yield RawStreamingChoice::Message(content);
                        }

                        for tool_call in tool_calls {
                            tool_calls_final.push(tool_call.clone());
                            yield RawStreamingChoice::ToolCall(
                                crate::streaming::RawStreamingToolCall::new(String::new(), tool_call.function.name, tool_call.function.arguments)
                            );
                        }
                    }

                    if response.done {
                        span.record("gen_ai.usage.input_tokens", response.prompt_eval_count);
                        span.record("gen_ai.usage.output_tokens", response.eval_count);
                        let message = Message::Assistant {
                            content: text_response.clone(),
                            thinking: if thinking_response.is_empty() { None } else { Some(thinking_response.clone()) },
                            images: None,
                            name: None,
                            tool_calls: tool_calls_final.clone()
                        };
                        span.record("gen_ai.output.messages", serde_json::to_string(&vec![message]).unwrap());
                        yield RawStreamingChoice::FinalResponse(
                            StreamingCompletionResponse {
                                total_duration: response.total_duration,
                                load_duration: response.load_duration,
                                prompt_eval_count: response.prompt_eval_count,
                                prompt_eval_duration: response.prompt_eval_duration,
                                eval_count: response.eval_count,
                                eval_duration: response.eval_duration,
                                done_reason: response.done_reason,
                            }
                        );
                        break;
                    }
                }
            }
        }.instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}

// ---------- Tool Definition ----------

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: completion::ToolDefinition,
}

impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        ToolDefinition {
            type_field: "function".to_owned(),
            function: completion::ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_chat_completion() {
        let sample_chat_response = json!({
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "The sky is blue because of Rayleigh scattering.",
                "images": null,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {
                                "location": "San Francisco, CA",
                                "format": "celsius"
                            }
                        }
                    }
                ]
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 61u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 468u64,
            "eval_duration": 7700000000u64
        });
        let sample_text = sample_chat_response.to_string();

        let chat_resp: CompletionResponse =
            serde_json::from_str(&sample_text).expect("Invalid JSON structure");
        let conv: completion::CompletionResponse<CompletionResponse> =
            chat_resp.try_into().unwrap();
        assert!(
            !conv.choice.is_empty(),
            "Expected non-empty choice in chat response"
        );
    }

    #[test]
    fn test_tool_definition_conversion() {
        let internal_tool = crate::completion::ToolDefinition {
            name: "get_current_weather".to_owned(),
            description: "Get the current weather for a location".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for"
                    },
                    "format": {
                        "type": "string",
                        "description": "The format to return the weather in",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "format"]
            }),
        };
        let ollama_tool: ToolDefinition = internal_tool.into();
        assert_eq!(ollama_tool.type_field, "function");
        assert_eq!(ollama_tool.function.name, "get_current_weather");
    }

    #[tokio::test]
    async fn test_chat_completion_with_thinking() {
        let sample_response = json!({
            "model": "qwen-thinking",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "thinking": "Let me think about this carefully.",
                "images": null,
                "tool_calls": []
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 61u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 468u64,
            "eval_duration": 7700000000u64
        });

        let chat_resp: CompletionResponse =
            serde_json::from_value(sample_response).expect("Failed to deserialize");

        if let Message::Assistant {
            thinking, content, ..
        } = &chat_resp.message
        {
            assert_eq!(
                thinking.as_ref().unwrap(),
                "Let me think about this carefully."
            );
            assert_eq!(content, "The answer is 42.");
        } else {
            panic!("Expected Assistant message");
        }
    }
}
