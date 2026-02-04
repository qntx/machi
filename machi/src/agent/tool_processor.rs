//! Unified tool call processing logic.
//!
//! This module provides a centralized processor for handling tool calls,
//! eliminating code duplication between sync and streaming execution paths.

use serde_json::Value;
use tracing::debug;

use crate::{
    error::Result,
    memory::{ActionStep, ToolCall},
    message::{ChatMessage, ChatMessageToolCall},
    tool::ToolBox,
    tools::FinalAnswerArgs,
};

use super::events::{StepResult, StreamEvent, StreamItem};

/// Result of processing tool calls.
pub struct ToolProcessResult {
    /// The step outcome (Continue or FinalAnswer).
    pub outcome: StepResult,
    /// Streaming events generated during processing (for streaming mode).
    pub events: Vec<StreamItem>,
}

/// Unified tool call processor for both sync and streaming execution.
pub struct ToolProcessor<'a> {
    tools: &'a ToolBox,
    streaming: bool,
}

impl<'a> ToolProcessor<'a> {
    /// Create a new tool processor.
    pub fn new(tools: &'a ToolBox) -> Self {
        Self {
            tools,
            streaming: false,
        }
    }

    /// Enable streaming mode to collect events.
    #[must_use]
    pub const fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    /// Process tool calls from a model response.
    ///
    /// This handles:
    /// - Extracting tool calls from native format or parsing from text
    /// - Recording tool calls in the action step
    /// - Handling `final_answer` tool specially
    /// - Executing other tools and collecting observations
    /// - Generating streaming events (if streaming mode enabled)
    pub async fn process(
        &self,
        step: &mut ActionStep,
        message: &ChatMessage,
    ) -> Result<ToolProcessResult> {
        let mut events = Vec::new();

        let Some(tool_calls) = Self::extract_tool_calls(step, message) else {
            return Ok(ToolProcessResult {
                outcome: StepResult::Continue,
                events,
            });
        };

        let mut observations = Vec::with_capacity(tool_calls.len());
        let mut final_answer = None;

        for tc in &tool_calls {
            let tool_name = tc.name();
            let tool_id = tc.id.clone();

            // Record tool call in step
            step.tool_calls
                .get_or_insert_with(Vec::new)
                .push(ToolCall::new(&tool_id, tool_name, tc.arguments().clone()));

            // Emit tool call start event (streaming mode)
            if self.streaming {
                events.push(Ok(StreamEvent::ToolCallStart {
                    id: tool_id.clone(),
                    name: tool_name.to_string(),
                }));
            }

            // Handle final_answer specially
            if tool_name == "final_answer" {
                let answer = Self::extract_final_answer(tc);
                final_answer = Some(answer);
                step.is_final_answer = true;

                if self.streaming {
                    events.push(Ok(StreamEvent::ToolCallComplete {
                        id: tool_id,
                        name: tool_name.to_string(),
                        result: Ok("Final answer recorded".to_string()),
                    }));
                }
                continue;
            }

            // Execute regular tool
            let (result_str, observation) =
                self.execute_tool(tool_name, tc.arguments().clone()).await;

            if result_str.is_err() {
                step.error = Some(observation.clone());
            }

            if self.streaming {
                events.push(Ok(StreamEvent::ToolCallComplete {
                    id: tool_id,
                    name: tool_name.to_string(),
                    result: result_str,
                }));
            }

            observations.push(observation);
        }

        // Store observations
        if !observations.is_empty() {
            step.observations = Some(observations.join("\n"));
        }

        // Determine outcome
        let outcome = match final_answer {
            Some(answer) => {
                step.action_output = Some(answer.clone());
                StepResult::FinalAnswer(answer)
            }
            None => StepResult::Continue,
        };

        Ok(ToolProcessResult { outcome, events })
    }

    /// Extract tool calls from model response.
    ///
    /// First checks for native tool calls, then tries to parse from text.
    pub fn extract_tool_calls(
        step: &ActionStep,
        message: &ChatMessage,
    ) -> Option<Vec<ChatMessageToolCall>> {
        // Check for native tool calls first
        if let Some(tc) = &message.tool_calls {
            return Some(tc.clone());
        }

        // Try to parse from text output
        if let Some(text) = &step.model_output {
            if let Some(parsed) = Self::parse_text_tool_call(text) {
                debug!(step = step.step_number, tool = %parsed.name(), "Parsed tool call from text");
                return Some(vec![parsed]);
            }
            debug!(step = step.step_number, output = %text, "Model returned text without tool call");
        } else {
            debug!(step = step.step_number, "Model returned empty response");
        }

        None
    }

    /// Parse a tool call from text output.
    ///
    /// For models that don't support native function calling, this extracts
    /// JSON tool call format from the text.
    pub fn parse_text_tool_call(text: &str) -> Option<ChatMessageToolCall> {
        // Find the first JSON object in the text
        let json_str = text.find('{').map(|start| {
            let mut depth = 0;
            let mut end = start;
            for (i, c) in text[start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            &text[start..end]
        })?;

        let json: Value = serde_json::from_str(json_str).ok()?;
        let name = json.get("name")?.as_str()?;
        let arguments = json
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(serde_json::Map::default()));

        Some(ChatMessageToolCall::new(
            format!("text_parsed_{}", uuid::Uuid::new_v4().simple()),
            name.to_string(),
            arguments,
        ))
    }

    /// Extract final answer value from tool call arguments.
    fn extract_final_answer(tc: &ChatMessageToolCall) -> Value {
        tc.parse_arguments::<FinalAnswerArgs>()
            .map_or_else(|_| tc.arguments().clone(), |args| args.answer)
    }

    /// Execute a tool and return result string and observation.
    async fn execute_tool(
        &self,
        name: &str,
        args: Value,
    ) -> (std::result::Result<String, String>, String) {
        match self.tools.call(name, args).await {
            Ok(result) => {
                let s = format!("Tool '{name}' returned: {result}");
                (Ok(s.clone()), s)
            }
            Err(e) => {
                let s = format!("Tool '{name}' failed: {e}");
                (Err(s.clone()), s)
            }
        }
    }
}
