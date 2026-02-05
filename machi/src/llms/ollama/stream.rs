//! Ollama stream parsing.

use crate::error::{LlmError, Result};
use crate::stream::{StopReason, StreamChunk};
use crate::usage::Usage;

use super::types::OllamaStreamChunk;

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

    // Handle tool calls
    if let Some(tool_calls) = &chunk.message.tool_calls
        && let Some((index, tc)) = tool_calls.iter().enumerate().next()
    {
        let _args = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
        // For Ollama, tool calls come complete in one chunk
        return StreamChunk::tool_use_start(index, format!("call_{index}"), &tc.function.name);
    }

    // Handle text content
    if !chunk.message.content.is_empty() {
        return StreamChunk::text(&chunk.message.content);
    }

    // Empty chunk
    StreamChunk::text("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_chunk() {
        let line =
            r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hello"},"done":false}"#;

        let result = parse_stream_line(line);
        assert!(result.is_some());

        let chunk = result
            .expect("should parse successfully")
            .expect("should not be error");
        assert!(matches!(chunk, StreamChunk::Text(ref t) if t == "Hello"));
    }

    #[test]
    fn test_parse_done_chunk() {
        let line = r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}"#;

        let result = parse_stream_line(line);
        assert!(result.is_some());

        let chunk = result
            .expect("should parse successfully")
            .expect("should not be error");
        if let StreamChunk::Usage(usage) = chunk {
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
        } else {
            unreachable!("Expected usage chunk, got {chunk:?}");
        }
    }

    #[test]
    fn test_parse_empty_line() {
        let result = parse_stream_line("");
        assert!(result.is_none());
    }
}
