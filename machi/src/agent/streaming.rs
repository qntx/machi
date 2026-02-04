//! Streaming execution for the Agent.
//!
//! This module provides streaming variants of agent execution that yield
//! events as they occur, enabling real-time feedback during agent runs.

use std::sync::atomic::Ordering;

use async_stream::stream;
use futures::{Stream, StreamExt};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::{
    error::AgentError,
    memory::{ActionStep, FinalAnswerStep, Timing},
    providers::common::GenerateOptions,
};

use super::{
    Agent,
    events::{StepResult, StreamEvent, StreamItem},
    tool_processor::ToolProcessor,
};

impl Agent {
    /// Stream execution events with full options.
    ///
    /// This is the core streaming implementation that yields events at both
    /// step-level and token-level granularity.
    #[expect(
        tail_expr_drop_order,
        reason = "stream yields control flow intentionally"
    )]
    pub(crate) fn stream_execution(&mut self) -> impl Stream<Item = StreamItem> + '_ {
        info!("Starting streaming agent run");

        stream! {
            let mut final_answer: Option<Value> = None;

            while self.step_number < self.config.max_steps {
                // Check for interruption
                if self.interrupt_flag.load(Ordering::SeqCst) {
                    yield Err(AgentError::Interrupted);
                    break;
                }

                self.step_number += 1;
                let mut step = ActionStep {
                    step_number: self.step_number,
                    timing: Timing::start_now(),
                    ..Default::default()
                };

                // Execute step and yield events
                let step_result = self.execute_step_streaming(&mut step).await;

                // Yield all streaming events from the step
                for event in step_result.events {
                    yield event;
                }

                // Finalize step timing and telemetry
                step.timing.complete();
                self.record_telemetry(&step);

                // Invoke callbacks
                let ctx = self.create_callback_context();
                self.callbacks.callback(&step, &ctx);

                // Yield step complete event
                let step_clone = step.clone();
                self.memory.add_step(step);
                yield Ok(StreamEvent::StepComplete {
                    step: self.step_number,
                    action_step: Box::new(step_clone),
                });

                // Handle step result
                match step_result.outcome {
                    Ok(StepResult::FinalAnswer(answer)) => {
                        // Validate answer
                        if let Err(e) = self.validate_answer(&answer) {
                            warn!(error = %e, "Final answer check failed");
                            yield Ok(StreamEvent::Error(format!("Final answer check failed: {e}")));
                            continue;
                        }

                        // Callback for final answer
                        let final_step = FinalAnswerStep { output: answer.clone() };
                        self.callbacks.callback(&final_step, &ctx);
                        self.memory.add_step(FinalAnswerStep { output: answer.clone() });

                        final_answer = Some(answer.clone());
                        yield Ok(StreamEvent::FinalAnswer { answer });
                        break;
                    }
                    Ok(StepResult::Continue) => {
                        // Continue to next step
                    }
                    Err(e) => {
                        warn!(step = self.step_number, error = %e, "Step failed");
                        yield Ok(StreamEvent::Error(e.to_string()));
                    }
                }
            }

            // Handle max steps reached
            if final_answer.is_none() && self.step_number >= self.config.max_steps {
                let error_msg = format!("Maximum steps ({}) reached", self.config.max_steps);
                self.memory.add_step(FinalAnswerStep {
                    output: Value::String(error_msg.clone()),
                });
                yield Err(AgentError::max_steps(self.step_number, self.config.max_steps));
            }
        }
    }

    /// Execute a single step with streaming, returning events and outcome.
    async fn execute_step_streaming(&self, step: &mut ActionStep) -> StepStreamResult {
        let mut events = Vec::new();

        // Prepare messages and options
        let messages = self.memory.to_messages(false);
        step.model_input_messages = Some(messages.clone());
        let options = GenerateOptions::new().with_tools(self.tools.definitions());
        debug!(step = step.step_number, "Generating model response");

        // Generate response with streaming events
        let model_result = if self.model.supports_streaming() {
            self.stream_model_response(messages, options, step, &mut events)
                .await
        } else {
            self.sync_model_response(messages, options, step, &mut events)
                .await
        };

        // Process result using unified tool processor
        let outcome = match model_result {
            Ok(message) => {
                let processor = ToolProcessor::new(&self.tools).with_streaming();
                match processor.process(step, &message).await {
                    Ok(result) => {
                        events.extend(result.events);
                        Ok(result.outcome)
                    }
                    Err(e) => {
                        step.error = Some(e.to_string());
                        Err(e)
                    }
                }
            }
            Err(e) => {
                step.error = Some(e.to_string());
                Err(e)
            }
        };

        StepStreamResult { events, outcome }
    }

    /// Stream model response and collect deltas.
    async fn stream_model_response(
        &self,
        messages: Vec<crate::message::ChatMessage>,
        options: GenerateOptions,
        step: &mut ActionStep,
        events: &mut Vec<StreamItem>,
    ) -> crate::error::Result<crate::message::ChatMessage> {
        let stream_result = self.model.generate_stream(messages, options).await;

        match stream_result {
            Ok(mut model_stream) => {
                let mut deltas = Vec::new();
                while let Some(result) = model_stream.next().await {
                    match result {
                        Ok(delta) => {
                            // Yield text delta for each token
                            if let Some(content) = &delta.content
                                && !content.is_empty()
                            {
                                events.push(Ok(StreamEvent::TextDelta(content.clone())));
                            }
                            if let Some(usage) = &delta.token_usage {
                                step.token_usage = Some(*usage);
                                events.push(Ok(StreamEvent::TokenUsage(*usage)));
                            }
                            deltas.push(delta);
                        }
                        Err(e) => {
                            let err_str = e.to_string();
                            step.error = Some(err_str.clone());
                            events.push(Err(AgentError::internal(&err_str)));
                            return Err(e);
                        }
                    }
                }
                let message = crate::message::aggregate_stream_deltas(&deltas);
                step.model_output_message = Some(message.clone());
                step.model_output = message.text_content();
                Ok(message)
            }
            Err(e) => Err(e),
        }
    }

    /// Synchronous model response (for models without streaming).
    async fn sync_model_response(
        &self,
        messages: Vec<crate::message::ChatMessage>,
        options: GenerateOptions,
        step: &mut ActionStep,
        events: &mut Vec<StreamItem>,
    ) -> crate::error::Result<crate::message::ChatMessage> {
        match self.model.generate(messages, options).await {
            Ok(response) => {
                step.model_output_message = Some(response.message.clone());
                step.token_usage = response.token_usage;
                step.model_output = response.message.text_content();
                if let Some(text) = &step.model_output {
                    events.push(Ok(StreamEvent::TextDelta(text.clone())));
                }
                if let Some(usage) = response.token_usage {
                    events.push(Ok(StreamEvent::TokenUsage(usage)));
                }
                Ok(response.message)
            }
            Err(e) => Err(e),
        }
    }
}

/// Result of streaming a single step.
struct StepStreamResult {
    /// Events generated during the step.
    events: Vec<StreamItem>,
    /// The outcome of the step.
    outcome: crate::error::Result<StepResult>,
}
