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
    memory::{ActionStep, FinalAnswerStep, Timing},
    message::ChatMessage,
    providers::common::GenerateOptions,
};

use super::{Agent, events::StepResult, tool_processor::ToolProcessor};

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

            // Invoke callbacks - metrics are collected via callback handlers
            let ctx = self.create_callback_context();
            self.callbacks.callback(&step, &ctx);

            match result {
                Ok(StepResult::FinalAnswer(answer)) => {
                    if let Err(e) = self.validate_answer(&answer) {
                        warn!(error = %e, "Final answer check failed");
                        step.error = Some(format!("Final answer check failed: {e}"));
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

        // Use unified tool processor with parallel execution support
        let processor =
            ToolProcessor::with_concurrency(&self.tools, self.config.max_parallel_tool_calls);
        let result = processor.process_parallel(step, &message).await?;
        Ok(result.outcome)
    }

    /// Validate the final answer against configured checks.
    pub(crate) fn validate_answer(&self, answer: &Value) -> Result<()> {
        if self.final_answer_checks.is_empty() {
            return Ok(());
        }
        self.final_answer_checks.validate(answer, &self.memory)
    }
}

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
