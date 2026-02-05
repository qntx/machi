//! OpenAI SSE stream parsing.

use crate::error::Result;
use crate::stream::{StopReason, StreamChunk};

use super::types::OpenAIStreamChunk;

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
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_chunk() {
        let data = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let results = parse_sse_events(data);
        assert_eq!(results.len(), 1);

        let chunk = results[0].as_ref().expect("should not be error");
        assert!(matches!(chunk, StreamChunk::Text(t) if t == "Hello"));
    }

    #[test]
    fn test_parse_done() {
        let data = "data: [DONE]";
        let results = parse_sse_events(data);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], Ok(StreamChunk::Done { .. })));
    }

    #[test]
    fn test_parse_multiple_lines() {
        let data = r#"data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: [DONE]"#;

        let results = parse_sse_events(data);
        assert_eq!(results.len(), 3);
    }
}
