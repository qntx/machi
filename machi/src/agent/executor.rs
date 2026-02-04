//! Agent execution logic for the ReAct loop.
//!
//! This module contains the core execution methods for running agent steps,
//! processing model responses, and handling tool calls.

use std::sync::atomic::Ordering;

use futures::StreamExt;
use serde_json::Value;
use tracing::{debug, warn};

use crate::{
    error::{AgentError, Result},
    memory::{ActionStep, FinalAnswerStep, Timing, ToolCall},
    message::{ChatMessage, ChatMessageToolCall},
    providers::common::GenerateOptions,
};

use super::{Agent, events::StepResult};

// ============================================================================
// Execution Loop
// ============================================================================

impl Agent {
    /// Execute the main ReAct loop until completion or max steps.
    pub(crate) async fn execute_loop(&mut self) -> Result<Value> {
        while self.step_number < self.config.max_steps {
            if self.interrupt_flag.load(Ordering::SeqCst) {
                return Err(AgentError::Interrupted);
            }

            self.step_number += 1;
            let mut step = ActionStep {
                step_number: self.step_number,
                timing: Timing::start_now(),
                ..Default::default()
            };

            let result = self.execute_step(&mut step).await;
            step.timing.complete();
            self.record_telemetry(&step);

            // Invoke callbacks
            let ctx = self.create_callback_context();
            self.callbacks.callback(&step, &ctx);

            match result {
                Ok(StepResult::FinalAnswer(answer)) => {
                    if let Err(e) = self.validate_answer(&answer) {
                        warn!(error = %e, "Final answer check failed");
                        step.error = Some(format!("Final answer check failed: {e}"));
                        self.telemetry.record_error(&e.to_string());
                        self.memory.add_step(step);
                        continue;
                    }
                    self.memory.add_step(step);

                    // Callback for final answer
                    let final_step = FinalAnswerStep {
                        output: answer.clone(),
                    };
                    self.callbacks.callback(&final_step, &ctx);

                    return Ok(answer);
                }
                Ok(StepResult::Continue) => {
                    self.memory.add_step(step);
                }
                Err(e) => {
                    let err_msg = e.to_string();
                    step.error = Some(err_msg.clone());
                    self.telemetry.record_error(&err_msg);
                    self.memory.add_step(step);
                    warn!(step = self.step_number, error = %e, "Step failed");
                }
            }
        }

        Err(AgentError::max_steps(
            self.step_number,
            self.config.max_steps,
        ))
    }

    /// Execute a single step: generate response and process tool calls.
    async fn execute_step(&self, step: &mut ActionStep) -> Result<StepResult> {
        let messages = self.memory.to_messages(false);
        step.model_input_messages = Some(messages.clone());

        let options = GenerateOptions::new().with_tools(self.tools.definitions());
        debug!(step = step.step_number, "Generating model response");

        let message = self.generate_response(messages, options, step).await?;
        self.process_response(step, &message).await
    }

    /// Validate the final answer against configured checks.
    pub(crate) fn validate_answer(&self, answer: &Value) -> Result<()> {
        if self.final_answer_checks.is_empty() {
            return Ok(());
        }
        self.final_answer_checks.validate(answer, &self.memory)
    }

    /// Record telemetry data for a step.
    pub(crate) fn record_telemetry(&mut self, step: &ActionStep) {
        if let Some(ref tool_calls) = step.tool_calls {
            for tc in tool_calls {
                self.telemetry.record_tool_call(&tc.name);
            }
        }
        self.telemetry
            .record_step(self.step_number, step.token_usage.as_ref());
    }
}

// ============================================================================
// Model Response Generation
// ============================================================================

impl Agent {
    /// Generate a response from the model, handling both streaming and non-streaming.
    pub(crate) async fn generate_response(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
        step: &mut ActionStep,
    ) -> Result<ChatMessage> {
        if self.model.supports_streaming() {
            self.generate_response_streaming(messages, options, step)
                .await
        } else {
            self.generate_response_sync(messages, options, step).await
        }
    }

    /// Generate response using streaming API.
    async fn generate_response_streaming(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
        step: &mut ActionStep,
    ) -> Result<ChatMessage> {
        let mut stream = self.model.generate_stream(messages, options).await?;
        let mut deltas = Vec::new();

        while let Some(result) = stream.next().await {
            match result {
                Ok(delta) => {
                    if let Some(usage) = &delta.token_usage {
                        step.token_usage = Some(*usage);
                    }
                    deltas.push(delta);
                }
                Err(e) => return Err(e),
            }
        }

        let message = crate::message::aggregate_stream_deltas(&deltas);
        step.model_output_message = Some(message.clone());
        step.model_output = message.text_content();
        Ok(message)
    }

    /// Generate response using synchronous API.
    async fn generate_response_sync(
        &self,
        messages: Vec<ChatMessage>,
        options: GenerateOptions,
        step: &mut ActionStep,
    ) -> Result<ChatMessage> {
        let response = self.model.generate(messages, options).await?;
        step.model_output_message = Some(response.message.clone());
        step.token_usage = response.token_usage;
        step.model_output = response.message.text_content();
        Ok(response.message)
    }
}

// ============================================================================
// Tool Call Processing
// ============================================================================

impl Agent {
    /// Process tool calls from the model response.
    pub(crate) async fn process_response(
        &self,
        step: &mut ActionStep,
        message: &ChatMessage,
    ) -> Result<StepResult> {
        let Some(tool_calls) = Self::extract_tool_calls(step, message) else {
            return Ok(StepResult::Continue);
        };

        let mut observations = Vec::with_capacity(tool_calls.len());
        let mut final_answer = None;

        for tc in &tool_calls {
            let tool_name = tc.name();
            step.tool_calls
                .get_or_insert_with(Vec::new)
                .push(ToolCall::new(&tc.id, tool_name, tc.arguments().clone()));

            if tool_name == "final_answer" {
                // Try parsing as FinalAnswerArgs, fallback to raw arguments
                let answer = tc
                    .parse_arguments::<crate::tools::FinalAnswerArgs>()
                    .map_or_else(|_| tc.arguments().clone(), |args| args.answer);
                final_answer = Some(answer);
                step.is_final_answer = true;
                continue;
            }

            let observation = self.execute_tool(tool_name, tc.arguments().clone()).await;
            if let Err(ref e) = observation {
                step.error = Some(e.clone());
            }
            observations.push(observation.unwrap_or_else(|e| e));
        }

        if !observations.is_empty() {
            step.observations = Some(observations.join("\n"));
        }

        match final_answer {
            Some(answer) => {
                step.action_output = Some(answer.clone());
                Ok(StepResult::FinalAnswer(answer))
            }
            None => Ok(StepResult::Continue),
        }
    }

    /// Extract tool calls from model response or parse from text.
    pub(crate) fn extract_tool_calls(
        step: &ActionStep,
        message: &ChatMessage,
    ) -> Option<Vec<ChatMessageToolCall>> {
        // First check for native tool calls
        if let Some(tc) = &message.tool_calls {
            return Some(tc.clone());
        }

        // Try to parse tool call from text output
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

    /// Execute a single tool and return the result as a formatted string.
    pub(crate) async fn execute_tool(
        &self,
        name: &str,
        args: Value,
    ) -> std::result::Result<String, String> {
        match self.tools.call(name, args).await {
            Ok(result) => Ok(format!("Tool '{name}' returned: {result}")),
            Err(e) => Err(format!("Tool '{name}' failed: {e}")),
        }
    }

    /// Parse a tool call from text output (for models that don't support native tool calling).
    pub(crate) fn parse_text_tool_call(text: &str) -> Option<ChatMessageToolCall> {
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
}
