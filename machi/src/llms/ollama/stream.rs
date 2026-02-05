//! Ollama stream parsing.

use serde::Deserialize;

use crate::error::{LlmError, Result};
use crate::stream::{StopReason, StreamChunk};
use crate::usage::Usage;

use super::client::OllamaToolCall;

/// Ollama streaming response chunk.
#[derive(Debug, Clone, Deserialize)]
struct OllamaStreamChunk {
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
struct OllamaStreamMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(default)]
    pub thinking: Option<String>,
}

/// Parse a streaming response line from Ollama.
pub fn parse_stream_line(line: &str) -> Option<Result<StreamChunk>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    match serde_json::from_str::<OllamaStreamChunk>(line) {
        Ok(chunk) => Some(Ok(convert_chunk(&chunk))),
        Err(e) => {
            tracing::warn!("Failed to parse Ollama chunk: {e}, line: {line}");
            Some(Err(LlmError::stream(format!("Parse error: {e}")).into()))
        }
    }
}

/// Generate a unique tool call ID.
fn generate_tool_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4())
}

/// Convert an Ollama stream chunk to our format.
fn convert_chunk(chunk: &OllamaStreamChunk) -> StreamChunk {
    // Handle completion
    if chunk.done {
        let stop_reason = match chunk.done_reason.as_deref() {
            Some("length") => StopReason::Length,
            // "stop", None, and any other value defaults to Stop
            _ => StopReason::Stop,
        };

        // If there's usage info, return that first
        if let (Some(prompt_count), Some(eval_count)) = (chunk.prompt_eval_count, chunk.eval_count)
        {
            return StreamChunk::Usage(Usage::new(prompt_count, eval_count));
        }

        return StreamChunk::done(Some(stop_reason));
    }

    // Handle thinking/reasoning content from reasoning models
    if let Some(thinking) = &chunk.message.thinking
        && !thinking.is_empty()
    {
        return StreamChunk::reasoning(thinking);
    }

    // Handle tool calls
    if let Some(tool_calls) = &chunk.message.tool_calls
        && let Some((index, tc)) = tool_calls.iter().enumerate().next()
    {
        // For Ollama, tool calls come complete in one chunk
        return StreamChunk::tool_use_start(index, generate_tool_call_id(), &tc.function.name);
    }

    // Handle text content
    if !chunk.message.content.is_empty() {
        return StreamChunk::text(&chunk.message.content);
    }

    // Empty chunk
    StreamChunk::text("")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::match_same_arms)]
mod tests {
    use super::*;

    mod parse_stream_line {
        use super::*;

        #[test]
        fn empty_line_returns_none() {
            assert!(parse_stream_line("").is_none());
        }

        #[test]
        fn whitespace_only_returns_none() {
            assert!(parse_stream_line("   ").is_none());
            assert!(parse_stream_line("\t").is_none());
            assert!(parse_stream_line("\n").is_none());
        }

        #[test]
        fn valid_json_returns_some_ok() {
            let line = r#"{"message":{"content":"Hi"},"done":false}"#;
            let result = parse_stream_line(line);

            assert!(result.is_some());
            assert!(result.expect("should be Some").is_ok());
        }

        #[test]
        fn invalid_json_returns_some_err() {
            let line = "not valid json";
            let result = parse_stream_line(line);

            assert!(result.is_some());
            assert!(result.expect("should be Some").is_err());
        }

        #[test]
        fn malformed_json_returns_error() {
            let line = r#"{"message":{"content":"Hi"}"#; // Missing closing brace
            let result = parse_stream_line(line);

            assert!(result.is_some());
            assert!(result.expect("should be Some").is_err());
        }

        #[test]
        fn trims_leading_whitespace() {
            let line = r#"   {"message":{"content":"Hi"},"done":false}"#;
            let result = parse_stream_line(line);

            assert!(result.is_some());
            assert!(result.expect("should be Some").is_ok());
        }

        #[test]
        fn trims_trailing_whitespace() {
            let line = r#"{"message":{"content":"Hi"},"done":false}   "#;
            let result = parse_stream_line(line);

            assert!(result.is_some());
            assert!(result.expect("should be Some").is_ok());
        }
    }

    mod text_content {
        use super::*;

        #[test]
        fn parses_simple_text_content() {
            let line = r#"{"message":{"content":"Hello"},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "Hello"));
        }

        #[test]
        fn parses_text_with_special_characters() {
            let line = r#"{"message":{"content":"Hello\nWorld"},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "Hello\nWorld"));
        }

        #[test]
        fn parses_text_with_unicode() {
            let line = r#"{"message":{"content":"你好世界 🌍"},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "你好世界 🌍"));
        }

        #[test]
        fn parses_text_with_quotes() {
            let line = r#"{"message":{"content":"He said \"hello\""},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "He said \"hello\""));
        }

        #[test]
        fn empty_content_returns_empty_text() {
            let line = r#"{"message":{"content":""},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t.is_empty()));
        }

        #[test]
        fn parses_incremental_text_chunks() {
            let chunks = [
                r#"{"message":{"content":"The"},"done":false}"#,
                r#"{"message":{"content":" weather"},"done":false}"#,
                r#"{"message":{"content":" is"},"done":false}"#,
                r#"{"message":{"content":" nice"},"done":false}"#,
            ];

            let mut accumulated = String::new();
            for chunk in chunks {
                if let Some(Ok(StreamChunk::Text(text))) = parse_stream_line(chunk) {
                    accumulated.push_str(&text);
                }
            }

            assert_eq!(accumulated, "The weather is nice");
        }
    }

    mod done_state {
        use super::*;

        #[test]
        fn done_with_stop_reason() {
            let line = r#"{"message":{"content":""},"done":true,"done_reason":"stop"}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Done { stop_reason } = result {
                assert_eq!(stop_reason, Some(StopReason::Stop));
            } else {
                panic!("Expected Done chunk, got {result:?}");
            }
        }

        #[test]
        fn done_with_length_reason() {
            let line = r#"{"message":{"content":""},"done":true,"done_reason":"length"}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Done { stop_reason } = result {
                assert_eq!(stop_reason, Some(StopReason::Length));
            } else {
                panic!("Expected Done chunk, got {result:?}");
            }
        }

        #[test]
        fn done_without_reason_defaults_to_stop() {
            let line = r#"{"message":{"content":""},"done":true}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Done { stop_reason } = result {
                assert_eq!(stop_reason, Some(StopReason::Stop));
            } else {
                panic!("Expected Done chunk, got {result:?}");
            }
        }

        #[test]
        fn done_with_null_reason_defaults_to_stop() {
            let line = r#"{"message":{"content":""},"done":true,"done_reason":null}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Done { stop_reason } = result {
                assert_eq!(stop_reason, Some(StopReason::Stop));
            } else {
                panic!("Expected Done chunk, got {result:?}");
            }
        }

        #[test]
        fn done_with_unknown_reason_defaults_to_stop() {
            let line = r#"{"message":{"content":""},"done":true,"done_reason":"unknown"}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Done { stop_reason } = result {
                assert_eq!(stop_reason, Some(StopReason::Stop));
            } else {
                panic!("Expected Done chunk, got {result:?}");
            }
        }
    }

    mod usage_info {
        use super::*;

        #[test]
        fn parses_usage_on_done() {
            let line = r#"{"message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":100,"eval_count":50}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Usage(usage) = result {
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 50);
                assert_eq!(usage.total_tokens, 150);
            } else {
                panic!("Expected Usage chunk, got {result:?}");
            }
        }

        #[test]
        fn usage_takes_precedence_over_done() {
            // When both usage info and done are present, usage should be returned
            let line = r#"{"message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Usage(_)));
        }

        #[test]
        fn missing_prompt_count_skips_usage() {
            let line =
                r#"{"message":{"content":""},"done":true,"done_reason":"stop","eval_count":5}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            // Should return Done, not Usage
            assert!(matches!(result, StreamChunk::Done { .. }));
        }

        #[test]
        fn missing_eval_count_skips_usage() {
            let line = r#"{"message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":10}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            // Should return Done, not Usage
            assert!(matches!(result, StreamChunk::Done { .. }));
        }

        #[test]
        fn zero_token_counts_are_valid() {
            let line =
                r#"{"message":{"content":""},"done":true,"prompt_eval_count":0,"eval_count":0}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Usage(usage) = result {
                assert_eq!(usage.input_tokens, 0);
                assert_eq!(usage.output_tokens, 0);
            } else {
                panic!("Expected Usage chunk, got {result:?}");
            }
        }

        #[test]
        fn large_token_counts() {
            let line = r#"{"message":{"content":""},"done":true,"prompt_eval_count":1000000,"eval_count":500000}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Usage(usage) = result {
                assert_eq!(usage.input_tokens, 1_000_000);
                assert_eq!(usage.output_tokens, 500_000);
            } else {
                panic!("Expected Usage chunk, got {result:?}");
            }
        }
    }

    mod thinking_content {
        use super::*;

        #[test]
        fn parses_thinking_content() {
            let line =
                r#"{"message":{"content":"","thinking":"Let me analyze this..."},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(
                matches!(result, StreamChunk::ReasoningContent(ref t) if t == "Let me analyze this...")
            );
        }

        #[test]
        fn empty_thinking_returns_text() {
            let line = r#"{"message":{"content":"Hello","thinking":""},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            // Empty thinking should fall through to text content
            assert!(matches!(result, StreamChunk::Text(ref t) if t == "Hello"));
        }

        #[test]
        fn null_thinking_returns_text() {
            let line = r#"{"message":{"content":"Hello","thinking":null},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "Hello"));
        }

        #[test]
        fn thinking_takes_precedence_over_text() {
            let line =
                r#"{"message":{"content":"visible","thinking":"internal reasoning"},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            // Thinking should be returned before text
            assert!(
                matches!(result, StreamChunk::ReasoningContent(ref t) if t == "internal reasoning")
            );
        }

        #[test]
        fn thinking_with_unicode() {
            let line = r#"{"message":{"content":"","thinking":"让我思考一下..."},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(
                matches!(result, StreamChunk::ReasoningContent(ref t) if t == "让我思考一下...")
            );
        }
    }

    mod tool_calls {
        use super::*;

        #[test]
        fn parses_tool_call() {
            let line = r#"{"message":{"content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::ToolUseStart { index, name, .. } = result {
                assert_eq!(index, 0);
                assert_eq!(name, "get_weather");
            } else {
                panic!("Expected ToolUseStart chunk, got {result:?}");
            }
        }

        #[test]
        fn tool_call_generates_unique_id() {
            let line = r#"{"message":{"content":"","tool_calls":[{"function":{"name":"test","arguments":{}}}]},"done":false}"#;

            let result1 = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");
            let result2 = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            if let (
                StreamChunk::ToolUseStart { id: id1, .. },
                StreamChunk::ToolUseStart { id: id2, .. },
            ) = (result1, result2)
            {
                assert_ne!(id1, id2, "Tool call IDs should be unique");
                assert!(id1.starts_with("call_"));
                assert!(id2.starts_with("call_"));
            } else {
                panic!("Expected ToolUseStart chunks");
            }
        }

        #[test]
        fn empty_tool_calls_array_returns_text() {
            let line = r#"{"message":{"content":"Hello","tool_calls":[]},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "Hello"));
        }

        #[test]
        fn null_tool_calls_returns_text() {
            let line = r#"{"message":{"content":"Hello","tool_calls":null},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "Hello"));
        }
    }

    mod generate_tool_call_id {
        use super::*;

        #[test]
        fn generates_prefixed_id() {
            let id = generate_tool_call_id();
            assert!(id.starts_with("call_"));
        }

        #[test]
        fn generates_unique_ids() {
            let ids: Vec<String> = (0..100).map(|_| generate_tool_call_id()).collect();
            let unique: std::collections::HashSet<_> = ids.iter().collect();
            assert_eq!(ids.len(), unique.len(), "All IDs should be unique");
        }

        #[test]
        fn id_has_valid_uuid_suffix() {
            let id = generate_tool_call_id();
            let suffix = id.strip_prefix("call_").expect("should have prefix");
            assert!(uuid::Uuid::parse_str(suffix).is_ok());
        }
    }

    mod realistic_stream_sequences {
        use super::*;

        #[test]
        fn typical_chat_completion_stream() {
            let chunks = [
                r#"{"message":{"content":"The"},"done":false}"#,
                r#"{"message":{"content":" capital"},"done":false}"#,
                r#"{"message":{"content":" of"},"done":false}"#,
                r#"{"message":{"content":" France"},"done":false}"#,
                r#"{"message":{"content":" is"},"done":false}"#,
                r#"{"message":{"content":" Paris"},"done":false}"#,
                r#"{"message":{"content":"."},"done":false}"#,
                r#"{"message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":15,"eval_count":8}"#,
            ];

            let mut text = String::new();
            let mut usage = None;

            for line in chunks {
                match parse_stream_line(line) {
                    Some(Ok(StreamChunk::Text(t))) => text.push_str(&t),
                    Some(Ok(StreamChunk::Usage(u))) => usage = Some(u),
                    Some(Ok(StreamChunk::Done { .. })) => {}
                    _ => {}
                }
            }

            assert_eq!(text, "The capital of France is Paris.");
            assert!(usage.is_some());
            let u = usage.expect("should have usage");
            assert_eq!(u.input_tokens, 15);
            assert_eq!(u.output_tokens, 8);
        }

        #[test]
        fn reasoning_model_stream() {
            let chunks = [
                r#"{"message":{"content":"","thinking":"Let me think about this problem..."},"done":false}"#,
                r#"{"message":{"content":"","thinking":"The answer requires calculation..."},"done":false}"#,
                r#"{"message":{"content":"The answer is 42."},"done":false}"#,
                r#"{"message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":50,"eval_count":100}"#,
            ];

            let mut reasoning = String::new();
            let mut text = String::new();

            for line in chunks {
                match parse_stream_line(line) {
                    Some(Ok(StreamChunk::ReasoningContent(r))) => reasoning.push_str(&r),
                    Some(Ok(StreamChunk::Text(t))) => text.push_str(&t),
                    _ => {}
                }
            }

            assert!(reasoning.contains("think about this problem"));
            assert!(reasoning.contains("requires calculation"));
            assert_eq!(text, "The answer is 42.");
        }

        #[test]
        fn tool_use_stream() {
            let chunks = [
                r#"{"message":{"content":"I'll check the weather for you."},"done":false}"#,
                r#"{"message":{"content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},"done":false}"#,
                r#"{"message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":20,"eval_count":15}"#,
            ];

            let mut has_text = false;
            let mut has_tool_call = false;
            let mut tool_name = String::new();

            for line in chunks {
                match parse_stream_line(line) {
                    Some(Ok(StreamChunk::Text(t))) if !t.is_empty() => has_text = true,
                    Some(Ok(StreamChunk::ToolUseStart { name, .. })) => {
                        has_tool_call = true;
                        tool_name = name;
                    }
                    _ => {}
                }
            }

            assert!(has_text);
            assert!(has_tool_call);
            assert_eq!(tool_name, "get_weather");
        }

        #[test]
        fn stream_truncated_by_length() {
            let chunks = [
                r#"{"message":{"content":"This is a very long response that"},"done":false}"#,
                r#"{"message":{"content":""},"done":true,"done_reason":"length","prompt_eval_count":10,"eval_count":4096}"#,
            ];

            let mut chunk_count = 0;
            let mut has_usage = false;

            for line in chunks {
                if let Some(Ok(chunk)) = parse_stream_line(line) {
                    chunk_count += 1;
                    if matches!(chunk, StreamChunk::Usage(_)) {
                        has_usage = true;
                    }
                }
            }

            assert_eq!(chunk_count, 2);
            // Usage takes precedence over Done when both token counts are present
            assert!(has_usage);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn handles_missing_message_fields_with_defaults() {
            // Minimal valid chunk with defaults
            let line = r#"{"message":{},"done":false}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            // Should return empty text due to defaults
            assert!(matches!(result, StreamChunk::Text(ref t) if t.is_empty()));
        }

        #[test]
        fn handles_extra_fields_gracefully() {
            let line = r#"{"message":{"content":"Hi","role":"assistant","extra_field":"ignored"},"done":false,"model":"llama3"}"#;
            let result = parse_stream_line(line)
                .expect("should be Some")
                .expect("should be Ok");

            assert!(matches!(result, StreamChunk::Text(ref t) if t == "Hi"));
        }

        #[test]
        fn handles_very_long_content() {
            let long_content = "a".repeat(10000);
            let line = format!(r#"{{"message":{{"content":"{long_content}"}},"done":false}}"#);
            let result = parse_stream_line(&line)
                .expect("should be Some")
                .expect("should be Ok");

            if let StreamChunk::Text(text) = result {
                assert_eq!(text.len(), 10000);
            } else {
                panic!("Expected Text chunk");
            }
        }
    }
}
