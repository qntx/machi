//! OpenAI SSE stream parsing.

use serde::Deserialize;

use crate::error::Result;
use crate::stream::{StopReason, StreamChunk};
use crate::usage::Usage;

/// OpenAI streaming chunk.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamChunk {
    pub choices: Vec<OpenAIStreamChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// OpenAI stream choice.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamChoice {
    pub delta: OpenAIStreamDelta,
    pub finish_reason: Option<String>,
}

/// OpenAI stream delta.
#[derive(Debug, Clone, Default, Deserialize)]
struct OpenAIStreamDelta {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

/// OpenAI stream tool call delta.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    pub function: Option<OpenAIStreamFunctionCall>,
}

/// OpenAI stream function call delta.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIStreamFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Parse SSE events from a text buffer.
pub fn parse_sse_events(text: &str) -> Vec<Result<StreamChunk>> {
    let mut results = Vec::new();

    for line in text.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        // Parse data lines
        if let Some(data) = line.strip_prefix("data: ") {
            let data = data.trim();

            // Handle stream end
            if data == "[DONE]" {
                results.push(Ok(StreamChunk::done(Some(StopReason::Stop))));
                continue;
            }

            // Parse JSON chunk
            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                Ok(chunk) => {
                    results.extend(convert_chunk(&chunk));
                }
                Err(e) => {
                    tracing::warn!("Failed to parse SSE chunk: {e}, data: {data}");
                }
            }
        }
    }

    results
}

/// Convert an OpenAI stream chunk to our format.
fn convert_chunk(chunk: &OpenAIStreamChunk) -> Vec<Result<StreamChunk>> {
    let mut results = Vec::new();

    for choice in &chunk.choices {
        // Handle text content
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            results.push(Ok(StreamChunk::text(content)));
        }

        // Handle tool calls
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                // Tool call start (has id and name)
                if let (Some(id), Some(func)) = (&tc.id, &tc.function)
                    && let Some(name) = &func.name
                {
                    results.push(Ok(StreamChunk::tool_use_start(tc.index, id, name)));
                }

                // Tool call delta (arguments)
                if let Some(func) = &tc.function
                    && let Some(args) = &func.arguments
                    && !args.is_empty()
                {
                    results.push(Ok(StreamChunk::tool_use_delta(tc.index, args)));
                }
            }
        }

        // Handle finish reason
        if let Some(reason) = &choice.finish_reason {
            let stop_reason = StopReason::parse(reason);
            if matches!(stop_reason, StopReason::ToolCalls) {
                // Mark tool calls as complete
                if let Some(tool_calls) = &choice.delta.tool_calls {
                    for tc in tool_calls {
                        results.push(Ok(StreamChunk::ToolUseComplete { index: tc.index }));
                    }
                }
            }
        }
    }

    // Handle usage
    if let Some(usage) = chunk.usage {
        results.push(Ok(StreamChunk::Usage(usage)));
    }

    results
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    mod parse_sse_events {
        use super::*;

        #[test]
        fn parses_text_content_chunk() {
            let data = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);

            let chunk = results[0].as_ref().unwrap();
            assert!(matches!(chunk, StreamChunk::Text(t) if t == "Hello"));
        }

        #[test]
        fn parses_done_signal() {
            let data = "data: [DONE]";
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
            assert!(matches!(results[0], Ok(StreamChunk::Done { .. })));
        }

        #[test]
        fn parses_multiple_chunks() {
            let data = r#"data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: [DONE]"#;

            let results = parse_sse_events(data);
            assert_eq!(results.len(), 3);
        }

        #[test]
        fn skips_empty_lines() {
            let data = "\n\n\ndata: [DONE]\n\n";
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
        }

        #[test]
        fn skips_sse_comments() {
            // SSE comments start with ':'
            let data = ": this is a comment\n: another comment\ndata: [DONE]";
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
            assert!(matches!(results[0], Ok(StreamChunk::Done { .. })));
        }

        #[test]
        fn handles_whitespace_in_data_line() {
            let data = "data:   [DONE]  ";
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
            assert!(matches!(results[0], Ok(StreamChunk::Done { .. })));
        }

        #[test]
        fn ignores_invalid_json_with_warning() {
            let data = "data: {invalid json}";
            let results = parse_sse_events(data);
            // Invalid JSON should be silently ignored (with tracing warning)
            assert!(results.is_empty());
        }

        #[test]
        fn ignores_non_data_lines() {
            let data = "event: message\nid: 123\ndata: [DONE]";
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
        }

        #[test]
        fn parses_empty_content() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#;
            let results = parse_sse_events(data);
            // Empty content should not produce a Text chunk
            assert!(results.is_empty());
        }

        #[test]
        fn parses_unicode_content() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":"你好世界 🌍"},"finish_reason":null}]}"#;
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
            if let Ok(StreamChunk::Text(text)) = &results[0] {
                assert_eq!(text, "你好世界 🌍");
            } else {
                panic!("Expected Text chunk");
            }
        }
    }

    mod tool_call_streaming {
        use super::*;

        #[test]
        fn parses_tool_call_start() {
            // First chunk with tool_call id and function name
            let data = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#;

            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);

            if let Ok(StreamChunk::ToolUseStart { index, id, name }) = &results[0] {
                assert_eq!(*index, 0);
                assert_eq!(id, "call_abc123");
                assert_eq!(name, "get_weather");
            } else {
                panic!("Expected ToolUseStart chunk, got {:?}", results[0]);
            }
        }

        #[test]
        fn parses_tool_call_arguments_delta() {
            // Subsequent chunk with partial arguments
            let data = r#"data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}"#;

            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);

            if let Ok(StreamChunk::ToolUseDelta {
                index,
                partial_json,
            }) = &results[0]
            {
                assert_eq!(*index, 0);
                assert_eq!(partial_json, r#"{"city":"#);
            } else {
                panic!("Expected ToolUseDelta chunk, got {:?}", results[0]);
            }
        }

        #[test]
        fn parses_complete_tool_call_sequence() {
            // Simulate complete tool call streaming sequence
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}

data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}

data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"test\"}"}}]},"finish_reason":null}]}

data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]"#;

            let results = parse_sse_events(data);

            // Should have: ToolUseStart, 2x ToolUseDelta, Done
            assert!(results.len() >= 3);

            // First: ToolUseStart
            assert!(matches!(&results[0], Ok(StreamChunk::ToolUseStart { .. })));

            // Middle: ToolUseDelta chunks
            let delta_count = results
                .iter()
                .filter(|r| matches!(r, Ok(StreamChunk::ToolUseDelta { .. })))
                .count();
            assert_eq!(delta_count, 2);

            // Last: Done
            assert!(matches!(results.last(), Some(Ok(StreamChunk::Done { .. }))));
        }

        #[test]
        fn parses_parallel_tool_calls() {
            // Multiple tool calls in same response
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"tool_a","arguments":""}},{"index":1,"id":"call_2","function":{"name":"tool_b","arguments":""}}]},"finish_reason":null}]}"#;

            let results = parse_sse_events(data);

            // Should have 2 ToolUseStart chunks
            let start_count = results
                .iter()
                .filter(|r| matches!(r, Ok(StreamChunk::ToolUseStart { .. })))
                .count();
            assert_eq!(start_count, 2);

            // Verify indices
            if let Ok(StreamChunk::ToolUseStart { index, name, .. }) = &results[0] {
                assert_eq!(*index, 0);
                assert_eq!(name, "tool_a");
            }
            if let Ok(StreamChunk::ToolUseStart { index, name, .. }) = &results[1] {
                assert_eq!(*index, 1);
                assert_eq!(name, "tool_b");
            }
        }

        #[test]
        fn handles_tool_call_without_arguments() {
            // Tool call start without arguments field
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","function":{"name":"no_args_tool"}}]},"finish_reason":null}]}"#;

            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
            assert!(matches!(&results[0], Ok(StreamChunk::ToolUseStart { .. })));
        }

        #[test]
        fn skips_empty_arguments_delta() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]},"finish_reason":null}]}"#;

            let results = parse_sse_events(data);
            // Empty arguments should not produce delta
            assert!(results.is_empty());
        }
    }

    mod finish_reasons {
        use super::*;

        #[test]
        fn parses_stop_finish_reason() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]"#;

            let results = parse_sse_events(data);
            assert!(matches!(
                results.last(),
                Some(Ok(StreamChunk::Done {
                    stop_reason: Some(StopReason::Stop)
                }))
            ));
        }

        #[test]
        fn parses_length_finish_reason() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":"length"}]}

data: [DONE]"#;

            let results = parse_sse_events(data);
            // [DONE] should have Stop reason by default
            let done = results.last().unwrap().as_ref().unwrap();
            assert!(matches!(done, StreamChunk::Done { .. }));
        }

        #[test]
        fn parses_tool_calls_finish_reason() {
            // With tool_calls present in delta
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0}]},"finish_reason":"tool_calls"}]}"#;

            let results = parse_sse_events(data);
            // Should produce ToolUseComplete
            let complete_count = results
                .iter()
                .filter(|r| matches!(r, Ok(StreamChunk::ToolUseComplete { .. })))
                .count();
            assert_eq!(complete_count, 1);
        }

        #[test]
        fn parses_content_filter_finish_reason() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}

data: [DONE]"#;

            let results = parse_sse_events(data);
            // Content filter should still reach Done
            assert!(
                results
                    .iter()
                    .any(|r| matches!(r, Ok(StreamChunk::Done { .. })))
            );
        }
    }

    mod usage_parsing {
        use super::*;

        #[test]
        fn parses_usage_in_final_chunk() {
            // OpenAI returns usage in final chunk when stream_options.include_usage is true
            let data = r#"data: {"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;

            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);

            if let Ok(StreamChunk::Usage(usage)) = &results[0] {
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 20);
                assert_eq!(usage.total_tokens, 30);
            } else {
                panic!("Expected Usage chunk, got {:?}", results[0]);
            }
        }

        #[test]
        fn parses_usage_with_details() {
            let data = r#"data: {"id":"1","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":{"reasoning_tokens":10}}}"#;

            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);

            if let Ok(StreamChunk::Usage(usage)) = &results[0] {
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 50);
                assert_eq!(usage.cached_tokens(), 20);
                assert_eq!(usage.reasoning_tokens(), 10);
            } else {
                panic!("Expected Usage chunk with details");
            }
        }
    }

    mod realistic_scenarios {
        use super::*;

        #[test]
        fn full_text_response_stream() {
            // Simulate realistic GPT-4o text response
            let data = r#": OPENAI-BETA: realtime=v1

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}

data: [DONE]"#;

            let results = parse_sse_events(data);

            // Should have: Text("Hello"), Text("!"), Usage, Done
            let text_chunks: Vec<_> = results
                .iter()
                .filter_map(|r| {
                    if let Ok(StreamChunk::Text(t)) = r {
                        Some(t.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(text_chunks, vec!["Hello", "!"]);

            // Check usage present
            assert!(
                results
                    .iter()
                    .any(|r| matches!(r, Ok(StreamChunk::Usage(_))))
            );

            // Check done present
            assert!(
                results
                    .iter()
                    .any(|r| matches!(r, Ok(StreamChunk::Done { .. })))
            );
        }

        #[test]
        fn tool_call_response_stream() {
            // Simulate tool call response from GPT-4o
            let data = r#"data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"city"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Tokyo"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]"#;

            let results = parse_sse_events(data);

            // Verify ToolUseStart
            let start = results
                .iter()
                .find(|r| matches!(r, Ok(StreamChunk::ToolUseStart { .. })))
                .expect("Should have ToolUseStart");
            if let Ok(StreamChunk::ToolUseStart { id, name, .. }) = start {
                assert_eq!(id, "call_weather");
                assert_eq!(name, "get_weather");
            }

            // Verify deltas
            let deltas: Vec<_> = results
                .iter()
                .filter_map(|r| {
                    if let Ok(StreamChunk::ToolUseDelta { partial_json, .. }) = r {
                        Some(partial_json.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            assert!(!deltas.is_empty());

            // Concatenated should form valid JSON
            let full_args: String = deltas.concat();
            assert_eq!(full_args, r#"{"city":"Tokyo"}"#);
        }

        #[test]
        fn mixed_text_and_tool_response() {
            // Some models return text before tool calls
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":"I'll check the weather for you."},"finish_reason":null}]}

data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{}"}}]},"finish_reason":null}]}

data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]"#;

            let results = parse_sse_events(data);

            // Should have both text and tool call
            assert!(
                results
                    .iter()
                    .any(|r| matches!(r, Ok(StreamChunk::Text(_))))
            );
            assert!(
                results
                    .iter()
                    .any(|r| matches!(r, Ok(StreamChunk::ToolUseStart { .. })))
            );
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn handles_null_content_field() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":null},"finish_reason":null}]}"#;
            let results = parse_sse_events(data);
            // null content should not produce chunk
            assert!(results.is_empty());
        }

        #[test]
        fn handles_missing_delta_fields() {
            let data =
                r#"data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":null}]}"#;
            let results = parse_sse_events(data);
            assert!(results.is_empty());
        }

        #[test]
        fn handles_empty_choices_array() {
            let data = r#"data: {"id":"1","choices":[]}"#;
            let results = parse_sse_events(data);
            assert!(results.is_empty());
        }

        #[test]
        fn handles_multiple_choices() {
            // n > 1 produces multiple choices
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null},{"index":1,"delta":{"content":"B"},"finish_reason":null}]}"#;
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 2);
        }

        #[test]
        fn handles_special_characters_in_content() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":"Line1\nLine2\tTabbed\"Quoted\""},"finish_reason":null}]}"#;
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
            if let Ok(StreamChunk::Text(text)) = &results[0] {
                assert!(text.contains('\n'));
                assert!(text.contains('\t'));
                assert!(text.contains('"'));
            }
        }

        #[test]
        fn preserves_whitespace_content() {
            let data = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":"   "},"finish_reason":null}]}"#;
            let results = parse_sse_events(data);
            assert_eq!(results.len(), 1);
            if let Ok(StreamChunk::Text(text)) = &results[0] {
                assert_eq!(text, "   ");
            }
        }
    }
}
