//! Managed agent system for multi-agent collaboration.
//!
//! This module provides the infrastructure for agents to manage and delegate
//! tasks to other agents, following the smolagents architecture pattern.
//!
//! # Architecture
//!
//! A managed agent is an agent that can be called by another (parent) agent
//! as if it were a tool. The parent agent delegates subtasks to managed agents,
//! which execute them independently and return results.
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! // Create a specialized research agent
//! let research_agent = Agent::builder()
//!     .model(model.clone())
//!     .name("researcher")
//!     .description("Expert at finding and summarizing information")
//!     .tool(Box::new(WebSearchTool::new()))
//!     .build();
//!
//! // Create a main agent that can delegate to the research agent
//! let mut main_agent = Agent::builder()
//!     .model(model)
//!     .managed_agent(research_agent)
//!     .build();
//!
//! let result = main_agent.run("Find recent news about Rust programming").await?;
//! ```

use std::{collections::HashMap, fmt::Write as _, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::error::{AgentError, Result};
use crate::tool::{BoxedTool, DynTool, ToolDefinition, ToolError};

/// Configuration for a managed agent's inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedAgentInputs {
    /// Task input specification.
    pub task: ManagedAgentInput,
    /// Additional arguments specification.
    pub additional_args: ManagedAgentInput,
}

impl Default for ManagedAgentInputs {
    fn default() -> Self {
        Self {
            task: ManagedAgentInput {
                input_type: "string".to_string(),
                description: "Long detailed description of the task.".to_string(),
                nullable: false,
            },
            additional_args: ManagedAgentInput {
                input_type: "object".to_string(),
                description: "Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.".to_string(),
                nullable: true,
            },
        }
    }
}

/// A single input specification for managed agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedAgentInput {
    /// The type of the input (e.g., "string", "object").
    #[serde(rename = "type")]
    pub input_type: String,
    /// Description of the input.
    pub description: String,
    /// Whether the input is nullable.
    #[serde(default)]
    pub nullable: bool,
}

/// Metadata describing a managed agent for prompt generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedAgentInfo {
    /// Unique name of the managed agent.
    pub name: String,
    /// Description of what the agent does.
    pub description: String,
    /// Input specifications.
    pub inputs: ManagedAgentInputs,
    /// Output type (always "string" for managed agents).
    pub output_type: String,
}

impl ManagedAgentInfo {
    /// Create new managed agent info.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            inputs: ManagedAgentInputs::default(),
            output_type: "string".to_string(),
        }
    }

    /// Generate a tool-calling prompt representation for this agent.
    #[must_use]
    pub fn to_tool_calling_prompt(&self) -> String {
        let inputs_json = serde_json::to_string(&self.inputs).unwrap_or_default();
        let mut result = String::with_capacity(
            self.name.len() + self.description.len() + inputs_json.len() + 64,
        );
        let _ = write!(
            result,
            "{}: {}\n    Takes inputs: {}\n    Returns an output of type: {}",
            self.name, self.description, inputs_json, self.output_type
        );
        result
    }
}

/// Arguments for calling a managed agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedAgentArgs {
    /// The task to delegate to the managed agent.
    pub task: String,
    /// Optional additional arguments/context.
    #[serde(default)]
    pub additional_args: Option<HashMap<String, Value>>,
}

/// Trait for types that can act as managed agents.
///
/// This trait allows any agent-like type to be used as a managed agent
/// within another agent's workflow.
#[async_trait]
pub trait ManagedAgent: Send + Sync {
    /// Get the unique name of this agent.
    fn name(&self) -> &str;

    /// Get a description of what this agent does.
    fn description(&self) -> &str;

    /// Execute a task and return the result.
    ///
    /// # Arguments
    ///
    /// * `task` - The task description to execute
    /// * `additional_args` - Optional additional context/arguments
    ///
    /// # Returns
    ///
    /// A string containing the agent's response/report.
    async fn call(
        &self,
        task: &str,
        additional_args: Option<HashMap<String, Value>>,
    ) -> Result<String>;

    /// Get metadata about this managed agent for prompt generation.
    fn info(&self) -> ManagedAgentInfo {
        ManagedAgentInfo::new(self.name(), self.description())
    }

    /// Whether to provide a run summary in the response.
    fn provide_run_summary(&self) -> bool {
        false
    }
}

/// A boxed dynamic managed agent.
pub type BoxedManagedAgent = Box<dyn ManagedAgent>;

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
        ToolDefinition {
            name: self.info.name.clone(),
            description: self.info.description.clone(),
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

    async fn call_json(&self, args: Value) -> std::result::Result<Value, ToolError> {
        // Parse arguments
        let parsed: ManagedAgentArgs = match &args {
            Value::String(s) => {
                serde_json::from_str(s).map_err(|e| ToolError::InvalidArguments(e.to_string()))?
            }
            _ => serde_json::from_value(args)
                .map_err(|e| ToolError::InvalidArguments(e.to_string()))?,
        };

        // Call the managed agent
        let agent = self.agent.lock().await;
        let result = agent
            .call(&parsed.task, parsed.additional_args)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        Ok(Value::String(result))
    }
}

/// A collection of managed agents.
#[derive(Default)]
pub struct ManagedAgentRegistry {
    /// Map of agent names to their tool wrappers.
    agents: HashMap<String, ManagedAgentTool>,
}

impl ManagedAgentRegistry {
    /// Create a new empty registry.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a managed agent to the registry.
    ///
    /// # Panics
    ///
    /// Panics if an agent with the same name already exists.
    #[track_caller]
    pub fn add(&mut self, agent: BoxedManagedAgent) {
        let name = agent.name().to_string();
        assert!(
            !self.agents.contains_key(&name),
            "Managed agent with name '{name}' already exists"
        );
        self.agents.insert(name, ManagedAgentTool::new(agent));
    }

    /// Try to add a managed agent, returning an error if the name is taken.
    pub fn try_add(&mut self, agent: BoxedManagedAgent) -> Result<()> {
        use std::collections::hash_map::Entry;

        let name = agent.name().to_string();
        match self.agents.entry(name) {
            Entry::Occupied(e) => Err(AgentError::configuration(format!(
                "Managed agent with name '{}' already exists",
                e.key()
            ))),
            Entry::Vacant(e) => {
                e.insert(ManagedAgentTool::new(agent));
                Ok(())
            }
        }
    }

    /// Get a managed agent by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ManagedAgentTool> {
        self.agents.get(name)
    }

    /// Get all managed agents as boxed tools.
    #[must_use]
    pub fn as_tools(&self) -> Vec<BoxedTool> {
        self.agents
            .values()
            .map(|agent| -> BoxedTool {
                Box::new(ManagedAgentToolClone {
                    agent: Arc::clone(&agent.agent),
                    info: agent.info.clone(),
                })
            })
            .collect()
    }

    /// Get info for all managed agents (for prompt generation).
    #[must_use]
    pub fn infos(&self) -> HashMap<String, ManagedAgentInfo> {
        self.agents
            .iter()
            .map(|(name, agent)| (name.clone(), agent.info.clone()))
            .collect()
    }

    /// Get the names of all managed agents.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.agents.keys().map(String::as_str).collect()
    }

    /// Check if a managed agent with the given name exists.
    #[inline]
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.agents.contains_key(name)
    }

    /// Get the number of managed agents.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if the registry is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Iterate over all managed agents.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ManagedAgentTool)> {
        self.agents.iter().map(|(k, v)| (k.as_str(), v))
    }
}

impl std::fmt::Debug for ManagedAgentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedAgentRegistry")
            .field("agents", &self.names())
            .finish()
    }
}

/// Internal cloneable tool wrapper for managed agents.
struct ManagedAgentToolClone {
    agent: Arc<Mutex<BoxedManagedAgent>>,
    info: ManagedAgentInfo,
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
        ToolDefinition {
            name: self.info.name.clone(),
            description: self.info.description.clone(),
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

    async fn call_json(&self, args: Value) -> std::result::Result<Value, ToolError> {
        let parsed: ManagedAgentArgs = match &args {
            Value::String(s) => {
                serde_json::from_str(s).map_err(|e| ToolError::InvalidArguments(e.to_string()))?
            }
            _ => serde_json::from_value(args)
                .map_err(|e| ToolError::InvalidArguments(e.to_string()))?,
        };

        let agent = self.agent.lock().await;
        let result = agent
            .call(&parsed.task, parsed.additional_args)
            .await
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;

        Ok(Value::String(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockManagedAgent {
        name: String,
        description: String,
    }

    #[async_trait]
    impl ManagedAgent for MockManagedAgent {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        async fn call(
            &self,
            task: &str,
            _additional_args: Option<HashMap<String, Value>>,
        ) -> Result<String> {
            Ok(format!("Mock agent '{}' processed: {}", self.name, task))
        }
    }

    #[test]
    fn test_managed_agent_info() {
        let info = ManagedAgentInfo::new("researcher", "Finds information");
        assert_eq!(info.name, "researcher");
        assert_eq!(info.output_type, "string");
    }

    #[test]
    fn test_managed_agent_registry() {
        let mut registry = ManagedAgentRegistry::new();

        let agent = MockManagedAgent {
            name: "test_agent".to_string(),
            description: "A test agent".to_string(),
        };

        registry.add(Box::new(agent));
        assert!(registry.contains("test_agent"));
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn test_managed_agent_tool() {
        let agent = MockManagedAgent {
            name: "helper".to_string(),
            description: "Helps with tasks".to_string(),
        };

        let tool = ManagedAgentTool::new(Box::new(agent));
        assert_eq!(tool.name(), "helper");

        let args = serde_json::json!({
            "task": "Do something"
        });

        let result = tool
            .call_json(args)
            .await
            .expect("tool call should succeed");
        assert!(
            result
                .as_str()
                .expect("result should be a string")
                .contains("Do something")
        );
    }
}
