//! Tool trait and utilities for defining agent tools.
//!
//! Tools are the primary way agents interact with the world. Each tool
//! represents a specific capability that an agent can invoke.
//!
//! # Result Type
//!
//! Tool functions should return `ToolResult<T>` for ergonomic error handling:
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! #[machi::tool]
//! async fn add(a: i64, b: i64) -> ToolResult<i64> {
//!     Ok(a + b)
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// A type alias for `Result<T, ToolError>`.
///
/// This provides ergonomic error handling for tool functions:
///
/// ```rust,ignore
/// #[machi::tool]
/// async fn my_tool(input: String) -> ToolResult<String> {
///     Ok(input.to_uppercase())
/// }
/// ```
pub type ToolResult<T> = Result<T, ToolError>;

/// Error type for tool execution failures.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ToolError {
    /// Error during tool execution.
    ExecutionError(String),
    /// Invalid arguments provided to the tool.
    InvalidArguments(String),
    /// Tool not found.
    NotFound(String),
    /// Tool is not initialized.
    NotInitialized,
    /// Generic error.
    Other(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExecutionError(msg) => write!(f, "Execution error: {msg}"),
            Self::InvalidArguments(msg) => write!(f, "Invalid arguments: {msg}"),
            Self::NotFound(name) => write!(f, "Tool not found: {name}"),
            Self::NotInitialized => write!(f, "Tool not initialized"),
            Self::Other(msg) => write!(f, "Tool error: {msg}"),
        }
    }
}

impl std::error::Error for ToolError {}

impl From<String> for ToolError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for ToolError {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(err: serde_json::Error) -> Self {
        Self::InvalidArguments(err.to_string())
    }
}

impl ToolError {
    /// Create an execution error.
    #[must_use]
    pub fn execution(msg: impl Into<String>) -> Self {
        Self::ExecutionError(msg.into())
    }

    /// Create an invalid arguments error.
    #[must_use]
    pub fn invalid_args(msg: impl Into<String>) -> Self {
        Self::InvalidArguments(msg.into())
    }
}

/// Definition of a tool for LLM function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolDefinition {
    /// Name of the tool.
    pub name: String,
    /// Description of what the tool does.
    pub description: String,
    /// JSON schema for the tool's parameters.
    pub parameters: Value,
    /// Output type string for LLM prompts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_type: Option<String>,
    /// JSON schema for structured output (optional).
    /// Used for `OpenAI`'s structured output / JSON mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

impl ToolDefinition {
    /// Create a new tool definition.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            output_type: None,
            output_schema: None,
        }
    }

    /// Set the output type.
    #[must_use]
    pub fn with_output_type(mut self, output_type: impl Into<String>) -> Self {
        self.output_type = Some(output_type.into());
        self
    }

    /// Set the output schema for structured output.
    #[must_use]
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Convert to `OpenAI` function calling format.
    #[must_use]
    pub fn to_openai_format(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        })
    }
}

/// The core trait for all tools that agents can use.
///
/// Tools encapsulate specific functionality that agents can invoke.
/// Each tool has a name, description, and can be called with typed arguments.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Static name of the tool.
    const NAME: &'static str;

    /// Arguments type for the tool.
    type Args: for<'de> Deserialize<'de> + Send;

    /// Output type of the tool.
    type Output: Serialize + Send;

    /// Error type for tool execution.
    type Error: Into<ToolError> + Send;

    /// Get the name of the tool.
    fn name(&self) -> &'static str;

    /// Get the description of the tool.
    fn description(&self) -> String;

    /// Get the JSON schema for the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Get the output type string for LLM prompts (e.g., "string", "integer", "object").
    ///
    /// This is similar to smolagents' `output_type` attribute, used for generating
    /// tool descriptions in prompts.
    fn output_type(&self) -> &'static str {
        "object"
    }

    /// Get the JSON schema for structured output (optional).
    ///
    /// Used for `OpenAI`'s structured output / JSON mode. Returns `None` by default.
    /// Override this to provide a schema that describes the structure of the output.
    fn output_schema(&self) -> Option<Value> {
        None
    }

    /// Execute the tool with the given arguments.
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;

    /// Get the tool definition for LLM function calling.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description(),
            parameters: self.parameters_schema(),
            output_type: Some(self.output_type().to_string()),
            output_schema: self.output_schema(),
        }
    }

    /// Call the tool with JSON arguments and return JSON output.
    ///
    /// Handles both JSON object and JSON string arguments (some LLMs return
    /// arguments as a JSON-encoded string rather than an object).
    async fn call_json(&self, args: Value) -> Result<Value, ToolError>
    where
        Self::Output: 'static,
    {
        // Handle both string and object arguments
        // Some LLMs return arguments as a JSON string instead of an object
        let typed_args: Self::Args = match &args {
            Value::String(s) => {
                serde_json::from_str(s).map_err(|e| ToolError::InvalidArguments(e.to_string()))?
            }
            _ => serde_json::from_value(args)
                .map_err(|e| ToolError::InvalidArguments(e.to_string()))?,
        };

        let result = self.call(typed_args).await.map_err(Into::into)?;

        serde_json::to_value(result).map_err(|e| ToolError::ExecutionError(e.to_string()))
    }
}

/// A boxed dynamic tool that can be used in collections.
pub type BoxedTool = Box<dyn DynTool>;

/// Object-safe version of the Tool trait for dynamic dispatch.
#[async_trait]
pub trait DynTool: Send + Sync {
    /// Get the name of the tool.
    fn name(&self) -> &str;

    /// Get the description of the tool.
    fn description(&self) -> String;

    /// Get the tool definition.
    fn definition(&self) -> ToolDefinition;

    /// Call the tool with JSON arguments.
    async fn call_json(&self, args: Value) -> Result<Value, ToolError>;
}

#[async_trait]
impl<T: Tool + 'static> DynTool for T
where
    T::Output: 'static,
{
    fn name(&self) -> &str {
        Tool::name(self)
    }

    fn description(&self) -> String {
        Tool::description(self)
    }

    fn definition(&self) -> ToolDefinition {
        Tool::definition(self)
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        Tool::call_json(self, args).await
    }
}

/// A collection of tools that can be used by an agent.
#[derive(Default)]
pub struct ToolBox {
    /// Map of tool names to tool instances.
    tools: std::collections::HashMap<String, BoxedTool>,
}

impl ToolBox {
    /// Create a new empty toolbox.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool to the toolbox.
    pub fn add<T: Tool + 'static>(&mut self, tool: T)
    where
        T::Output: 'static,
    {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }

    /// Add a boxed tool to the toolbox.
    pub fn add_boxed(&mut self, tool: BoxedTool) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&BoxedTool> {
        self.tools.get(name)
    }

    /// Get all tool definitions.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Get the names of all tools.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.tools.values().map(|t| t.name()).collect()
    }

    /// Check if the toolbox contains a tool with the given name.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of tools in the toolbox.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the toolbox is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Call a tool by name with JSON arguments.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::NotFound` if the tool doesn't exist, or propagates
    /// any error from the tool execution.
    pub async fn call(&self, name: &str, args: Value) -> Result<Value, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.call_json(args).await
    }

    /// Call multiple tools in parallel with optional concurrency limit.
    ///
    /// This method executes tool calls concurrently using tokio tasks,
    /// respecting the optional `max_concurrent` limit for resource control.
    ///
    /// # Arguments
    ///
    /// * `calls` - Iterator of (tool_name, tool_id, arguments) tuples
    /// * `max_concurrent` - Optional limit on concurrent executions
    ///
    /// # Returns
    ///
    /// A vector of `ToolCallResult` in the same order as input calls.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let calls = vec![
    ///     ("search", "call_1", json!({"query": "rust"})),
    ///     ("fetch", "call_2", json!({"url": "https://example.com"})),
    /// ];
    /// let results = toolbox.call_parallel(calls, Some(4)).await;
    /// ```
    pub async fn call_parallel(
        &self,
        calls: impl IntoIterator<Item = (&str, String, Value)>,
        max_concurrent: Option<usize>,
    ) -> Vec<ToolCallResult> {
        use futures::stream::{self, StreamExt};

        let calls: Vec<_> = calls.into_iter().collect();

        if calls.is_empty() {
            return Vec::new();
        }

        // For single call, execute directly without spawning
        if calls.len() == 1 {
            let (name, id, args) = calls
                .into_iter()
                .next()
                .expect("calls length already checked to be 1");
            let result = self.call(name, args).await;
            return vec![ToolCallResult {
                id,
                name: name.to_string(),
                result,
            }];
        }

        // Determine concurrency limit
        let concurrency = max_concurrent.unwrap_or(calls.len());

        // Create futures for each tool call
        let futures = calls.into_iter().map(|(name, id, args)| {
            let tool_name = name.to_string();
            let tool_id = id;
            let tool_args = args;

            async move {
                let result = if let Some(tool) = self.tools.get(&tool_name) {
                    tool.call_json(tool_args).await
                } else {
                    Err(ToolError::NotFound(tool_name.clone()))
                };

                ToolCallResult {
                    id: tool_id,
                    name: tool_name,
                    result,
                }
            }
        });

        // Execute with bounded concurrency using buffer_unordered
        stream::iter(futures)
            .buffer_unordered(concurrency)
            .collect()
            .await
    }
}

/// Result of a parallel tool call execution.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// The tool call ID.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The execution result.
    pub result: Result<Value, ToolError>,
}

impl ToolCallResult {
    /// Check if this tool call was a success.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        self.result.is_ok()
    }

    /// Check if this tool call failed.
    #[must_use]
    pub const fn is_err(&self) -> bool {
        self.result.is_err()
    }

    /// Convert to observation string for agent memory.
    #[must_use]
    pub fn to_observation(&self) -> String {
        match &self.result {
            Ok(value) => format!("Tool '{}' returned: {value}", self.name),
            Err(e) => format!("Tool '{}' failed: {e}", self.name),
        }
    }
}

impl fmt::Debug for ToolBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolBox")
            .field("tools", &self.names())
            .finish()
    }
}

/// Built-in tool for providing the final answer to a task.
///
/// This is a core tool that is always available and automatically added to agents.
/// It allows the agent to conclude a task by providing the final answer.
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalAnswerTool;

/// Arguments for the final answer tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FinalAnswerArgs {
    /// The final answer to the problem. Can be any JSON value.
    pub answer: Value,
}

#[async_trait]
impl Tool for FinalAnswerTool {
    const NAME: &'static str = "final_answer";
    type Args = FinalAnswerArgs;
    type Output = Value;
    type Error = ToolError;

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn description(&self) -> String {
        "Provides the final answer to the given problem.".to_string()
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the problem."
                }
            },
            "required": ["answer"]
        })
    }

    fn output_type(&self) -> &'static str {
        "any"
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.answer)
    }
}
