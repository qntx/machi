//! OpenAI-compatible Chat Completions wire format (build + parse).
//!
//! Pure helpers are always available so unit tests need no HTTP client.
//! The live [`OpenAiCompatSampler`] is behind the `openai` feature.

use machi_tools::ToolDefinition;
use machi_types::{ErrorCode, MachiError, Message, Role, ToolCall, ToolCallId, Usage};
use serde_json::{Value, json};

use crate::sample::{SampleRequest, SampleResponse, ToolChoice};

/// Configuration for an OpenAI-compatible endpoint.
#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    /// Base URL without trailing slash (e.g. `https://api.openai.com` or local proxy).
    pub base_url: String,
    /// API key sent as `Authorization: Bearer …` when non-empty.
    pub api_key: String,
    /// Path under base URL (default `/v1/chat/completions`).
    pub chat_path: String,
}

impl OpenAiCompatConfig {
    /// Chat Completions at `{base_url}/v1/chat/completions`.
    #[must_use]
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: trim_slash(base_url.into()),
            api_key: api_key.into(),
            chat_path: "/v1/chat/completions".into(),
        }
    }

    /// Full request URL.
    #[must_use]
    pub fn chat_url(&self) -> String {
        format!("{}{}", self.base_url, self.chat_path)
    }
}

fn trim_slash(mut s: String) -> String {
    while s.ends_with('/') {
        s.pop();
    }
    s
}

/// Build the JSON body for Chat Completions from a kernel [`SampleRequest`].
#[must_use]
pub fn build_chat_completions_body(req: &SampleRequest) -> Value {
    let messages: Vec<Value> = req.messages.iter().map(message_to_openai).collect();
    let mut map = serde_json::Map::new();
    map.insert("model".into(), json!(req.model));
    map.insert("messages".into(), Value::Array(messages));
    if !req.tools.is_empty() {
        map.insert(
            "tools".into(),
            Value::Array(req.tools.iter().map(tool_to_openai).collect()),
        );
        map.insert(
            "tool_choice".into(),
            tool_choice_to_openai(&req.tool_choice),
        );
    }
    if let Some(max) = req.max_output_tokens {
        map.insert("max_tokens".into(), json!(max));
    }
    if let Some(temp) = req.temperature {
        map.insert("temperature".into(), json!(temp));
    }
    if let Some(fmt) = &req.response_format {
        // Accept either a raw JSON Schema object or an already-wrapped OpenAI envelope.
        let wire = if fmt.get("type").and_then(Value::as_str) == Some("json_schema")
            || fmt.get("type").and_then(Value::as_str) == Some("json_object")
        {
            fmt.clone()
        } else {
            openai_json_schema_format(fmt)
        };
        map.insert("response_format".into(), wire);
    }
    Value::Object(map)
}

/// Parse `Retry-After` as seconds (integer). HTTP-date forms are not supported (returns `None`).
#[must_use]
pub fn parse_retry_after_header(raw: &str) -> Option<std::time::Duration> {
    let s = raw.trim();
    let secs: u64 = s.parse().ok()?;
    Some(std::time::Duration::from_secs(secs.min(7 * 24 * 3600)))
}

/// Wrap a JSON Schema object as an OpenAI-compatible `response_format` envelope.
#[must_use]
pub fn openai_json_schema_format(schema: &Value) -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "machi_output",
            "strict": true,
            "schema": schema,
        }
    })
}

fn message_to_openai(msg: &Message) -> Value {
    match msg.role {
        Role::Tool => {
            let mut map = serde_json::Map::new();
            map.insert("role".into(), json!("tool"));
            map.insert(
                "content".into(),
                json!(msg.content.clone().unwrap_or_default()),
            );
            if let Some(id) = &msg.tool_call_id {
                map.insert("tool_call_id".into(), json!(id.as_str()));
            }
            if let Some(name) = &msg.name {
                map.insert("name".into(), json!(name));
            }
            Value::Object(map)
        }
        Role::Assistant if !msg.tool_calls.is_empty() => {
            let tool_calls: Vec<Value> = msg
                .tool_calls
                .iter()
                .map(|tc| {
                    let args = match &tc.arguments {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    json!({
                        "id": tc.id.as_str(),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": args,
                        }
                    })
                })
                .collect();
            let mut map = serde_json::Map::new();
            map.insert("role".into(), json!("assistant"));
            map.insert("tool_calls".into(), Value::Array(tool_calls));
            map.insert(
                "content".into(),
                msg.content.as_ref().map_or(Value::Null, |c| json!(c)),
            );
            Value::Object(map)
        }
        role => {
            json!({
                "role": role.as_str(),
                "content": msg.text(),
            })
        }
    }
}

fn tool_to_openai(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn tool_choice_to_openai(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Named(name) => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

/// Parse a Chat Completions JSON response into a kernel [`SampleResponse`].
///
/// # Errors
///
/// Returns [`MachiError`] when the payload is missing choices/message or ids.
pub fn parse_chat_completions_response(body: &Value) -> Result<SampleResponse, MachiError> {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .ok_or_else(|| {
            MachiError::new(
                ErrorCode::LlmInvalidResponse,
                "chat completions response missing choices[0]",
            )
        })?;
    let message_v = choice.get("message").ok_or_else(|| {
        MachiError::new(
            ErrorCode::LlmInvalidResponse,
            "chat completions choice missing message",
        )
    })?;
    let content = message_v.get("content").and_then(|c| {
        if c.is_null() {
            None
        } else {
            c.as_str().map(str::to_owned)
        }
    });
    let tool_calls = parse_tool_calls(message_v.get("tool_calls"))?;
    let message = if tool_calls.is_empty() {
        Message::assistant(content.unwrap_or_default())
    } else {
        let mut m = Message::assistant_tools(tool_calls);
        m.content = content;
        m
    };
    let usage = parse_usage(body.get("usage"));
    let stop_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Ok(SampleResponse {
        message,
        usage,
        stop_reason,
    })
}

fn parse_tool_calls(raw: Option<&Value>) -> Result<Vec<ToolCall>, MachiError> {
    let Some(arr) = raw.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let id = item.get("id").and_then(Value::as_str).ok_or_else(|| {
            MachiError::new(ErrorCode::LlmInvalidResponse, "tool_call missing id")
        })?;
        let name = item
            .pointer("/function/name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                MachiError::new(
                    ErrorCode::LlmInvalidResponse,
                    "tool_call missing function.name",
                )
            })?;
        let args_raw = item
            .pointer("/function/arguments")
            .cloned()
            .unwrap_or(json!("{}"));
        let arguments = match args_raw {
            Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
            other => other,
        };
        out.push(ToolCall {
            id: ToolCallId::new(id)?,
            name: name.to_owned(),
            arguments,
        });
    }
    Ok(out)
}

fn parse_usage(raw: Option<&Value>) -> Usage {
    let Some(u) = raw else {
        return Usage::zero();
    };
    let input = u
        .get("prompt_tokens")
        .or_else(|| u.get("input_tokens"))
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    let output = u
        .get("completion_tokens")
        .or_else(|| u.get("output_tokens"))
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    Usage::new(input, output)
}

/// Map HTTP status (+ optional headers) to a typed LLM error.
///
/// Single construction path for provider HTTP failures; sets transport metadata
/// so [`crate::retry::decide_retry`] can honor `Retry-After` / status policy.
#[must_use]
pub fn http_status_error(status: u16, body: &str) -> MachiError {
    http_status_error_with_meta(status, body, None, None)
}

/// Like [`http_status_error`] with explicit header semantics.
#[must_use]
pub fn http_status_error_with_meta(
    status: u16,
    body: &str,
    retry_after: Option<std::time::Duration>,
    x_should_retry: Option<bool>,
) -> MachiError {
    use machi_types::RetryClass;

    use crate::retry::{HttpRetryClass, classify_http_status, error_code_for_http};

    let snippet: String = body.chars().take(256).collect();
    let class = classify_http_status(status, x_should_retry);
    let code = error_code_for_http(status, class);
    // Auth is fatal unless a credential-refresh adapter explicitly sets AuthRefresh.
    let retry = match (code, class) {
        (ErrorCode::LlmAuth, _) => RetryClass::Never,
        (_, HttpRetryClass::RateLimited | HttpRetryClass::Retry) => RetryClass::Backoff,
        (_, HttpRetryClass::Fatal) => RetryClass::Never,
    };
    let mut err = MachiError::new(code, format!("openai-compatible HTTP {status}: {snippet}"))
        .with_retry(retry)
        .with_http_status(status);
    if let Some(after) = retry_after {
        err = err.with_retry_after(after);
    }
    if let Some(hint) = x_should_retry {
        err = err.with_x_should_retry(hint);
    }
    err
}

#[cfg(feature = "openai")]
mod client {
    use async_trait::async_trait;
    use machi_types::{ErrorCode, MachiError};
    use tracing::{Instrument, info_span};

    use super::{
        OpenAiCompatConfig, build_chat_completions_body, http_status_error_with_meta,
        parse_chat_completions_response, parse_retry_after_header,
    };
    use crate::sample::{SampleRequest, SampleResponse};
    use crate::sampler::LlmSampler;

    /// HTTP client for OpenAI-compatible Chat Completions.
    #[derive(Debug, Clone)]
    pub struct OpenAiCompatSampler {
        config: OpenAiCompatConfig,
        client: reqwest::Client,
    }

    impl OpenAiCompatSampler {
        /// Create a sampler with a default `reqwest` client.
        ///
        /// # Errors
        ///
        /// Returns an error when the HTTP client cannot be built.
        pub fn new(config: OpenAiCompatConfig) -> Result<Self, MachiError> {
            let client = reqwest::Client::builder().build().map_err(|e| {
                MachiError::new(ErrorCode::LlmProvider, format!("http client build: {e}"))
            })?;
            Ok(Self { config, client })
        }

        /// Create with an existing client (tests inject custom base URL easily).
        #[must_use]
        pub fn with_client(config: OpenAiCompatConfig, client: reqwest::Client) -> Self {
            Self { config, client }
        }
    }

    #[async_trait]
    impl LlmSampler for OpenAiCompatSampler {
        async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, MachiError> {
            if request.cancel.is_cancelled() {
                return Err(MachiError::new(ErrorCode::LlmCancelled, "sample cancelled"));
            }
            if request.deadline.is_some_and(|d| d.is_expired()) {
                return Err(MachiError::new(
                    ErrorCode::LlmCancelled,
                    "sample deadline expired",
                ));
            }

            let body = build_chat_completions_body(&request);
            let url = self.config.chat_url();
            let span = info_span!(
                "machi.sample.http",
                machi.model = %request.model,
                machi.provider = "openai_compat",
            );

            async move {
                let mut req = self.client.post(&url).json(&body);
                if !self.config.api_key.is_empty() {
                    req = req.bearer_auth(&self.config.api_key);
                }

                let response = tokio::select! {
                    biased;
                    () = request.cancel.cancelled() => {
                        return Err(MachiError::llm_cancelled("sample cancelled during http"));
                    }
                    res = req.send() => res.map_err(|e| {
                        MachiError::new(ErrorCode::LlmProvider, format!("http request failed: {e}"))
                            .with_retry(machi_types::RetryClass::Backoff)
                    })?,
                };

                let status = response.status().as_u16();
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(parse_retry_after_header);
                let x_should_retry = response
                    .headers()
                    .get("x-should-retry")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
                        "true" | "1" | "yes" => Some(true),
                        "false" | "0" | "no" => Some(false),
                        _ => None,
                    });
                let text = tokio::select! {
                    biased;
                    () = request.cancel.cancelled() => {
                        return Err(MachiError::llm_cancelled("sample cancelled during body read"));
                    }
                    text = response.text() => text.map_err(|e| {
                        MachiError::new(
                            ErrorCode::LlmProvider,
                            format!("http body read failed: {e}"),
                        )
                    })?,
                };
                if !(200..300).contains(&status) {
                    return Err(http_status_error_with_meta(
                        status,
                        &text,
                        retry_after,
                        x_should_retry,
                    ));
                }
                let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                    MachiError::new(
                        ErrorCode::LlmInvalidResponse,
                        format!("invalid JSON body: {e}"),
                    )
                })?;
                parse_chat_completions_response(&value)
            }
            .instrument(span)
            .await
        }
    }
}

#[cfg(feature = "openai")]
pub use client::OpenAiCompatSampler;

#[cfg(test)]
mod tests {
    use machi_types::{ErrorCode, Message, ToolCallId};
    use serde_json::json;

    use super::*;

    #[test]
    fn build_body_includes_tools() {
        let req = SampleRequest {
            model: "gpt-test".into(),
            messages: vec![Message::user("hi")],
            tools: vec![ToolDefinition {
                name: "calc".into(),
                description: "calc".into(),
                parameters: json!({"type":"object"}),
            }],
            tool_choice: ToolChoice::Auto,
            response_format: None,
            max_output_tokens: Some(64),
            temperature: Some(0.0),
            cancel: tokio_util::sync::CancellationToken::new(),
            deadline: None,
        };
        let body = build_chat_completions_body(&req);
        assert_eq!(body.get("model"), Some(&json!("gpt-test")));
        assert_eq!(body.get("max_tokens"), Some(&json!(64)));
        assert!(
            body.get("tools")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty())
        );
        assert_eq!(body.get("tool_choice"), Some(&json!("auto")));
    }

    #[test]
    fn parse_assistant_text() {
        let body = json!({
            "choices": [{
                "message": { "role": "assistant", "content": "hello world" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 2 }
        });
        let resp = parse_chat_completions_response(&body).expect("parse");
        assert_eq!(resp.message.text(), "hello world");
        assert_eq!(resp.usage.input_tokens, 3);
        assert_eq!(resp.usage.output_tokens, 2);
        assert_eq!(resp.stop_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn response_format_wraps_raw_schema() {
        let schema = json!({"type": "object", "properties": {"ok": {"type": "boolean"}}});
        let req = SampleRequest {
            model: "gpt-test".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            response_format: Some(schema.clone()),
            max_output_tokens: None,
            temperature: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            deadline: None,
        };
        let body = build_chat_completions_body(&req);
        let fmt = body.get("response_format").expect("response_format");
        assert_eq!(fmt.get("type"), Some(&json!("json_schema")));
        assert_eq!(fmt.pointer("/json_schema/schema"), Some(&schema));
    }

    #[test]
    fn parse_retry_after_seconds() {
        assert_eq!(
            parse_retry_after_header("12"),
            Some(std::time::Duration::from_secs(12))
        );
        assert_eq!(parse_retry_after_header("nope"), None);
    }

    #[test]
    fn http_meta_sets_transport_fields() {
        let err = http_status_error_with_meta(
            429,
            "slow down",
            Some(std::time::Duration::from_secs(3)),
            Some(true),
        );
        assert_eq!(err.code(), ErrorCode::LlmRateLimit);
        assert_eq!(err.http_status(), Some(429));
        assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(3)));
        assert_eq!(err.x_should_retry(), Some(true));
    }

    #[test]
    fn parse_tool_calls() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calc",
                            "arguments": "{\"expr\":\"1+2\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1 }
        });
        let resp = parse_chat_completions_response(&body).expect("parse");
        assert_eq!(resp.message.tool_calls.len(), 1);
        let tc = resp.message.tool_calls.first().expect("tc");
        assert_eq!(tc.id.as_str(), "call_1");
        assert_eq!(tc.name, "calc");
        assert_eq!(tc.arguments.get("expr"), Some(&json!("1+2")));
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_calls"));
        let _ = ToolCallId::new("call_1");
    }
}
