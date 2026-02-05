//! Agent execution loop implementation.

use serde_json::Value;
use tracing::{debug, info, warn};

use crate::chat::ChatRequest;
use crate::error::{Error, Result};
use crate::message::Message;
use crate::stream::StopReason;
use crate::tool::{ToolCallResult, ToolConfirmationRequest, ToolConfirmationResponse};

use super::Agent;
use super::memory::{ActionStep, MemoryStep, ToolCallInfo};

/// Special tool name for final answers.
const FINAL_ANSWER_TOOL: &str = "final_answer";

impl Agent {
    /// Execute the main agent loop.
    pub(super) async fn execute_loop(&mut self) -> Result<Value> {
        while self.step_number < self.config.max_steps {
            // Check for interruption
            if self.is_interrupted() {
                return Err(Error::Interrupted);
            }

            self.step_number += 1;
            debug!(step = self.step_number, "Executing step");

            // Execute one step
            match self.execute_step().await? {
                StepResult::Continue => {}
                StepResult::FinalAnswer(value) => return Ok(value),
            }
        }

        Err(Error::max_steps(self.config.max_steps))
    }

    /// Execute a single step.
    async fn execute_step(&mut self) -> Result<StepResult> {
        // Build the request
        let request = self.build_request();

        // Call the LLM
        let response = self.provider.chat(&request).await?;

        // Record usage
        let usage = response.usage;

        // Check for tool calls
        if response.has_tool_calls() {
            let tool_calls = response
                .tool_calls()
                .expect("tool_calls should be present when has_tool_calls is true");

            // Check for final_answer
            for tc in tool_calls {
                if tc.name() == FINAL_ANSWER_TOOL {
                    let answer = tc.function.arguments_value();
                    // Extract the actual answer from the arguments
                    let final_value = answer.get("answer").cloned().unwrap_or(answer);
                    info!("Final answer received");
                    return Ok(StepResult::FinalAnswer(final_value));
                }
            }

            // Execute tool calls
            let results = self.execute_tool_calls(tool_calls).await;

            // Record the action step
            let mut action_step = ActionStep::new();
            if let Some(text) = response.text() {
                action_step = action_step.with_thought(text);
            }
            action_step.usage = usage;

            for result in &results {
                action_step.add_tool_call(ToolCallInfo {
                    id: result.id.clone(),
                    name: result.name.clone(),
                    arguments: Value::Null, // TODO: store actual args
                    output: result.result.as_ref().ok().cloned(),
                    error: result.result.as_ref().err().map(ToString::to_string),
                });
            }

            self.memory.add_step(MemoryStep::Action(action_step));

            // Add assistant message with tool calls
            self.memory.add_message(response.message.clone());

            // Add tool response messages
            for result in results {
                let content = result.to_string_for_llm();
                let tool_msg = Message::tool(&result.id, content);
                self.memory.add_message(tool_msg);
            }

            return Ok(StepResult::Continue);
        }

        // No tool calls - treat text response as implicit final answer
        if let Some(text) = response.text()
            && !text.is_empty()
            && response.stop_reason != StopReason::ToolCalls
        {
            info!("Implicit final answer from text response");
            return Ok(StepResult::FinalAnswer(Value::String(text)));
        }

        // Add the assistant message
        self.memory.add_message(response.message);

        Ok(StepResult::Continue)
    }

    /// Build a chat request from current state.
    fn build_request(&self) -> ChatRequest {
        let mut request = ChatRequest::new(self.provider.default_model())
            .messages(self.memory.messages().to_vec());

        // Add tools
        if !self.tools.is_empty() {
            let mut tools = self.tools.to_tools();

            // Add final_answer tool
            tools.push(crate::tool::ToolDefinition::new(
                FINAL_ANSWER_TOOL,
                "Provide the final answer to the user's task. Call this when you have completed the task.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "answer": {
                            "description": "The final answer or result"
                        }
                    },
                    "required": ["answer"]
                }),
            ));

            request = request.tools(tools);
        }

        // Apply config
        if let Some(temp) = self.config.temperature {
            request = request.temperature(temp);
        }
        if let Some(max_tokens) = self.config.max_tokens {
            request = request.max_tokens(max_tokens);
        }

        request
    }

    /// Execute tool calls and return results.
    async fn execute_tool_calls(
        &mut self,
        tool_calls: &[crate::message::ToolCall],
    ) -> Vec<ToolCallResult> {
        let mut results = Vec::with_capacity(tool_calls.len());

        for tc in tool_calls {
            // Skip final_answer as it's handled separately
            if tc.name() == FINAL_ANSWER_TOOL {
                continue;
            }

            let result = self.execute_single_tool_call(tc).await;
            results.push(result);
        }

        results
    }

    /// Execute a single tool call.
    async fn execute_single_tool_call(&mut self, tc: &crate::message::ToolCall) -> ToolCallResult {
        let name = tc.name();
        let id = tc.id.clone();

        // Check if tool exists
        if !self.tools.contains(name) {
            return ToolCallResult {
                id,
                name: name.to_owned(),
                result: Err(crate::error::ToolError::not_found(name)),
            };
        }

        // Check execution policy
        if self.tools.is_forbidden(name) {
            return ToolCallResult {
                id,
                name: name.to_owned(),
                result: Err(crate::error::ToolError::forbidden(name)),
            };
        }

        // Handle confirmation if required
        if self.tools.requires_confirmation(name)
            && let Some(handler) = &self.confirmation_handler
        {
            let args = tc.function.arguments_value();
            let request = ToolConfirmationRequest::new(&id, name, args);
            let response = handler.confirm(&request).await;

            match response {
                ToolConfirmationResponse::Denied => {
                    return ToolCallResult {
                        id,
                        name: name.to_owned(),
                        result: Err(crate::error::ToolError::confirmation_denied(name)),
                    };
                }
                ToolConfirmationResponse::ApproveAll => {
                    self.tools.mark_auto_approved(name);
                }
                ToolConfirmationResponse::Approved => {}
            }
        }

        // Execute the tool
        let args = tc.function.arguments_value();
        debug!(tool = name, "Executing tool");

        match self.tools.call(name, args).await {
            Ok(output) => {
                info!(tool = name, "Tool executed successfully");
                ToolCallResult {
                    id,
                    name: name.to_owned(),
                    result: Ok(output),
                }
            }
            Err(e) => {
                warn!(tool = name, error = %e, "Tool execution failed");
                ToolCallResult {
                    id,
                    name: name.to_owned(),
                    result: Err(e),
                }
            }
        }
    }
}

/// Result of executing a single step.
enum StepResult {
    /// Continue to next step.
    Continue,
    /// Final answer received.
    FinalAnswer(Value),
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_step_result() {
        // Basic test for step result enum
        use super::StepResult;

        let cont = StepResult::Continue;
        assert!(matches!(cont, StepResult::Continue));

        let final_ans = StepResult::FinalAnswer(serde_json::json!(42));
        assert!(matches!(final_ans, StepResult::FinalAnswer(_)));
    }
}
