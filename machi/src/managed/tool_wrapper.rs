//! Tool wrapper for managed agents.
//!
//! This module provides the wrapper that exposes managed agents as tools,
//! enabling seamless integration with the agent's tool system.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::tool::{DynTool, ToolDefinition, ToolError};

use super::types::{BoxedManagedAgent, ManagedAgentArgs, ManagedAgentInfo};

/// A wrapper that exposes a managed agent as a tool.
///
/// This allows managed agents to be seamlessly integrated into an agent's
/// tool system, enabling the LLM to call them using the standard tool-calling
/// interface.
pub struct ManagedAgentTool {
    /// The wrapped managed agent.
    agent: Arc<Mutex<BoxedManagedAgent>>,
    /// Cached agent info.
    info: ManagedAgentInfo,
}

impl ManagedAgentTool {
    /// Create a new managed agent tool wrapper.
    #[must_use]
    pub fn new(agent: BoxedManagedAgent) -> Self {
        let info = agent.info();
        Self {
            agent: Arc::new(Mutex::new(agent)),
            info,
        }
    }

    /// Get the wrapped agent's name.
    #[must_use]
    pub fn agent_name(&self) -> &str {
        &self.info.name
    }

    /// Get the wrapped agent's description.
    #[must_use]
    pub fn agent_description(&self) -> &str {
        &self.info.description
    }

    /// Get the agent info for prompt generation.
    #[must_use]
    pub const fn agent_info(&self) -> &ManagedAgentInfo {
        &self.info
    }

    /// Get a clone of the internal Arc for registry use.
    pub(crate) fn clone_arc(&self) -> Arc<Mutex<BoxedManagedAgent>> {
        Arc::clone(&self.agent)
    }
}

impl std::fmt::Debug for ManagedAgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedAgentTool")
            .field("name", &self.info.name)
            .field("description", &self.info.description)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl DynTool for ManagedAgentTool {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn description(&self) -> String {
        self.info.description.clone()
    }

    fn definition(&self) -> ToolDefinition {
        make_tool_definition(&self.info)
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        call_managed_agent(&self.agent, args).await
    }
}

/// Internal cloneable tool wrapper for managed agents.
///
/// This is used by the registry to create multiple tool references
/// to the same underlying managed agent.
pub struct ManagedAgentToolClone {
    pub(crate) agent: Arc<Mutex<BoxedManagedAgent>>,
    pub(crate) info: ManagedAgentInfo,
}

impl std::fmt::Debug for ManagedAgentToolClone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedAgentToolClone")
            .field("name", &self.info.name)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl DynTool for ManagedAgentToolClone {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn description(&self) -> String {
        self.info.description.clone()
    }

    fn definition(&self) -> ToolDefinition {
        make_tool_definition(&self.info)
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        call_managed_agent(&self.agent, args).await
    }
}

/// Create a tool definition for a managed agent.
fn make_tool_definition(info: &ManagedAgentInfo) -> ToolDefinition {
    ToolDefinition {
        name: info.name.clone(),
        description: info.description.clone(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Long detailed description of the task."
                },
                "additional_args": {
                    "type": "object",
                    "description": "Dictionary of extra inputs to pass to the managed agent.",
                    "nullable": true
                }
            },
            "required": ["task"]
        }),
        output_type: Some("string".to_string()),
        output_schema: None,
    }
}

/// Call a managed agent with JSON arguments.
async fn call_managed_agent(
    agent: &Arc<Mutex<BoxedManagedAgent>>,
    args: Value,
) -> Result<Value, ToolError> {
    // Parse arguments
    let parsed: ManagedAgentArgs = match &args {
        Value::String(s) => {
            serde_json::from_str(s).map_err(|e| ToolError::InvalidArguments(e.to_string()))?
        }
        _ => {
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?
        }
    };

    // Call the managed agent
    let agent = agent.lock().await;
    let result = agent
        .call(&parsed.task, parsed.additional_args)
        .await
        .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

    Ok(Value::String(result))
}
