//! Unified tool call processing logic.
//!
//! This module provides a centralized processor for handling tool calls,
//! eliminating code duplication between sync and streaming execution paths.
//!
//! # Parallel Execution
//!
//! Tool calls can be executed in parallel when multiple tools are invoked
//! in a single step. Use `process_parallel` for concurrent execution with
//! optional concurrency limits.

use serde_json::Value;
use tracing::debug;

use crate::{
    error::Result,
    memory::{ActionStep, ToolCall},
    message::{ChatMessage, ChatMessageToolCall},
    tool::ToolBox,
    tools::FinalAnswerArgs,
};

use super::events::StepResult;

/// Result of processing tool calls.
pub struct ToolProcessResult {
    /// The step outcome (Continue or FinalAnswer).
    pub outcome: StepResult,
}

/// Unified tool call processor for both sync and streaming execution.
pub struct ToolProcessor<'a> {
    tools: &'a ToolBox,
    /// Maximum concurrent tool calls (None = unlimited).
    max_concurrent: Option<usize>,
}

impl<'a> ToolProcessor<'a> {
    /// Create a new tool processor with concurrency limit.
    ///
    /// # Arguments
    ///
    /// * `tools` - The toolbox containing available tools
    /// * `max_concurrent` - Maximum number of concurrent tool executions.
    ///   - `None` = unlimited parallelism (default)
    ///   - `Some(1)` = sequential execution
    ///   - `Some(n)` = up to n concurrent executions
    pub const fn with_concurrency(tools: &'a ToolBox, max_concurrent: Option<usize>) -> Self {
        Self {
            tools,
            max_concurrent,
        }
    }

    /// Process tool calls from a model response with parallel execution.
    ///
    /// This method executes multiple tool calls concurrently, respecting
    /// the configured concurrency limit. Tool calls are processed as follows:
    ///
    /// 1. Extract tool calls from native format or parse from text
    /// 2. Record all tool calls in the action step
    /// 3. Handle `final_answer` specially (not executed, just recorded)
    /// 4. Execute remaining tools in parallel
    /// 5. Collect observations and determine outcome
    ///
    /// # Arguments
    ///
    /// * `step` - The action step to record tool calls and observations
    /// * `message` - The model response message containing tool calls
    pub async fn process_parallel(
        &self,
        step: &mut ActionStep,
        message: &ChatMessage,
    ) -> Result<ToolProcessResult> {
        let Some(tool_calls) = Self::extract_tool_calls(step, message) else {
            return Ok(ToolProcessResult {
                outcome: StepResult::Continue,
            });
        };

        // Separate final_answer from regular tool calls
        let mut final_answer = None;
        let mut regular_calls: Vec<(&str, String, Value)> = Vec::new();

        for tc in &tool_calls {
            let tool_name = tc.name();
            let tool_id = tc.id.clone();

            // Record tool call in step
            step.tool_calls
                .get_or_insert_with(Vec::new)
                .push(ToolCall::new(&tool_id, tool_name, tc.arguments().clone()));

            // Handle final_answer specially - don't execute, just record
            if tool_name == "final_answer" {
                let answer = Self::extract_final_answer(tc);
                final_answer = Some(answer);
                step.is_final_answer = true;
            } else {
                // Queue for parallel execution
                regular_calls.push((tool_name, tool_id, tc.arguments().clone()));
            }
        }

        // Execute regular tools in parallel
        let observations = if regular_calls.is_empty() {
            Vec::new()
        } else {
            debug!(
                count = regular_calls.len(),
                max_concurrent = ?self.max_concurrent,
                "Executing tool calls in parallel"
            );

            let results = self
                .tools
                .call_parallel(regular_calls, self.max_concurrent)
                .await;

            let mut obs = Vec::with_capacity(results.len());
            for result in results {
                let observation = result.to_observation();

                if result.is_err() {
                    step.error = Some(observation.clone());
                }

                obs.push(observation);
            }
            obs
        };

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

        Ok(ToolProcessResult { outcome })
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
}
