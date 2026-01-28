//! Rig backend adapter.
//!
//! This module provides integration with the `rig` crate for LLM operations.
//!
//! # Architecture
//!
//! The rig crate has its own tool system. To integrate with machi's tool system,
//! we use a hybrid approach:
//!
//! 1. For simple prompts: Use rig's `Prompt` trait directly
//! 2. For tool-based interactions: Parse tool calls from LLM response
//!
//! Note: The rig agent should be constructed with tools configured via rig's API.
//! Machi's tools are executed separately after parsing tool calls from responses.

use std::sync::Mutex;

use rig::completion::Prompt;

use crate::backend::{Backend, BackendResponse, ToolCall, ToolResult as BackendToolResult};
use crate::error::{Error, Result};
use crate::tools::ToolDefinition;

/// Adapter for the rig agent framework.
///
/// Wraps a rig agent that implements the `Prompt` trait.
/// The agent should be configured with tools via rig's API when created.
///
/// # Example
///
/// ```ignore
/// use rig::providers::openai;
/// use rig::client::{CompletionClient, ProviderClient};
///
/// let client = openai::Client::from_env();
/// let agent = client.agent("gpt-4o")
///     .preamble("You are a helpful Web3 assistant with wallet capabilities.")
///     .tool(GetAddressTool)  // rig tools
///     .build();
///
/// let backend = RigBackend::new(agent);
/// ```
pub struct RigBackend<A> {
    agent: A,
    /// Conversation history for multi-turn tool calls.
    history: Mutex<Vec<ConversationMessage>>,
}

/// A message in the conversation history.
#[derive(Debug, Clone)]
enum ConversationMessage {
    User(String),
    Assistant(String),
    ToolResult { id: String, content: String },
}

impl<A> RigBackend<A> {
    /// Create a new rig backend with the given agent.
    pub fn new(agent: A) -> Self {
        Self {
            agent,
            history: Mutex::new(Vec::new()),
        }
    }

    /// Get a reference to the underlying agent.
    pub const fn agent(&self) -> &A {
        &self.agent
    }

    /// Clear conversation history.
    pub fn clear_history(&self) {
        if let Ok(mut history) = self.history.lock() {
            history.clear();
        }
    }

    /// Build a prompt with tool results.
    fn build_prompt_with_tool_results(&self, results: &[BackendToolResult]) -> String {
        let history = self.history.lock().unwrap();
        let mut prompt = String::new();

        for msg in history.iter() {
            match msg {
                ConversationMessage::User(s) => {
                    prompt.push_str(&format!("User: {s}\n"));
                }
                ConversationMessage::Assistant(s) => {
                    prompt.push_str(&format!("Assistant: {s}\n"));
                }
                ConversationMessage::ToolResult { id, content } => {
                    prompt.push_str(&format!("Tool Result [{id}]: {content}\n"));
                }
            }
        }

        prompt.push_str("Tool execution results:\n");
        for result in results {
            prompt.push_str(&format!("- {}: {}\n", result.tool_call_id, result.content));
        }
        prompt.push_str("\nPlease provide a response based on these tool results.");
        prompt
    }

    /// Parse tool calls from LLM response.
    /// Looks for JSON blocks with tool_call format.
    fn parse_tool_calls(response: &str) -> Option<Vec<ToolCall>> {
        // Try to find JSON tool call blocks in the response
        // Format: {"tool_call": {"name": "...", "arguments": {...}}}
        // Or: [{"tool_call": ...}, ...]

        // First, try to find a JSON array or object
        let trimmed = response.trim();

        // Look for tool call patterns
        if let Some(start) = trimmed.find("{\"tool_call\"") {
            if let Some(end) = trimmed[start..].find('}') {
                let json_str = &trimmed
                    [start..=start + end + trimmed[start + end + 1..].find('}').unwrap_or(0) + end];
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(tool_call) = parsed.get("tool_call") {
                        let name = tool_call.get("name")?.as_str()?.to_string();
                        let arguments = tool_call
                            .get("arguments")
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        let id = format!("call_{}", uuid_simple());
                        return Some(vec![ToolCall::new(id, name, arguments)]);
                    }
                }
            }
        }

        // Alternative: Look for function call syntax
        // <function_call>{"name": "...", "arguments": {...}}</function_call>
        if let Some(start) = trimmed.find("<function_call>") {
            if let Some(end) = trimmed.find("</function_call>") {
                let json_str = &trimmed[start + 15..end];
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                    let name = parsed.get("name")?.as_str()?.to_string();
                    let arguments = parsed
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::json!({}));
                    let id = format!("call_{}", uuid_simple());
                    return Some(vec![ToolCall::new(id, name, arguments)]);
                }
            }
        }

        None
    }
}

/// Generate a simple unique ID.
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{:x}{:x}", duration.as_secs(), duration.subsec_nanos())
}

impl<A> Backend for RigBackend<A>
where
    A: Prompt + Send + Sync,
{
    async fn complete(&self, prompt: &str) -> Result<String> {
        let response = self
            .agent
            .prompt(prompt)
            .await
            .map_err(|e| Error::Backend(format!("rig prompt failed: {e}")))?;
        Ok(response)
    }

    async fn complete_with_tools(
        &self,
        prompt: &str,
        tools: &[ToolDefinition],
    ) -> Result<BackendResponse> {
        // Build system message with tool descriptions
        let tools_desc = tools
            .iter()
            .map(|t| {
                let params: Vec<String> = t
                    .parameters
                    .iter()
                    .map(|p| format!("  - {}: {} ({})", p.name, p.r#type, p.description))
                    .collect();
                format!(
                    "- {}({}): {}\n  Parameters:\n{}",
                    t.name,
                    t.parameters
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                    t.description,
                    params.join("\n")
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let enhanced_prompt = format!(
            "You have access to the following tools:\n\n{}\n\n\
            To use a tool, respond with JSON in this format:\n\
            {{\"tool_call\": {{\"name\": \"tool_name\", \"arguments\": {{...}}}}}}\n\n\
            If you don't need to use a tool, just respond normally.\n\n\
            User request: {}",
            tools_desc, prompt
        );

        // Store in history
        {
            let mut history = self.history.lock().unwrap();
            history.push(ConversationMessage::User(prompt.to_string()));
        }

        let response = self
            .agent
            .prompt(&enhanced_prompt)
            .await
            .map_err(|e| Error::Backend(format!("rig prompt failed: {e}")))?;

        // Store assistant response
        {
            let mut history = self.history.lock().unwrap();
            history.push(ConversationMessage::Assistant(response.clone()));
        }

        // Try to parse tool calls from response
        if let Some(tool_calls) = Self::parse_tool_calls(&response) {
            Ok(BackendResponse::ToolCalls(tool_calls))
        } else {
            Ok(BackendResponse::Text(response))
        }
    }

    async fn continue_with_tool_results(
        &self,
        tool_results: &[BackendToolResult],
        tools: &[ToolDefinition],
    ) -> Result<BackendResponse> {
        // Store tool results in history
        {
            let mut history = self.history.lock().unwrap();
            for result in tool_results {
                history.push(ConversationMessage::ToolResult {
                    id: result.tool_call_id.clone(),
                    content: result.content.clone(),
                });
            }
        }

        let prompt = self.build_prompt_with_tool_results(tool_results);

        // Add tools description again
        let tools_desc = tools
            .iter()
            .map(|t| format!("- {}: {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n");

        let enhanced_prompt = format!(
            "Available tools:\n{}\n\n{}\n\n\
            Based on the tool results above, either:\n\
            1. Use another tool if needed: {{\"tool_call\": {{\"name\": \"...\", \"arguments\": {{...}}}}}}\n\
            2. Or provide a final response to the user.",
            tools_desc, prompt
        );

        let response = self
            .agent
            .prompt(&enhanced_prompt)
            .await
            .map_err(|e| Error::Backend(format!("rig prompt failed: {e}")))?;

        // Store assistant response
        {
            let mut history = self.history.lock().unwrap();
            history.push(ConversationMessage::Assistant(response.clone()));
        }

        // Try to parse tool calls from response
        if let Some(tool_calls) = Self::parse_tool_calls(&response) {
            Ok(BackendResponse::ToolCalls(tool_calls))
        } else {
            Ok(BackendResponse::Text(response))
        }
    }
}
