//! Tool trait and utilities for defining agent tools.
//!
//! Tools are the primary way agents interact with the world. Each tool
//! represents a specific capability that an agent can invoke.
//!
//! # `OpenAI` API Alignment
//!
//! This module aligns with `OpenAI`'s Function Calling API:
//! - `ToolDefinition` serializes to `{"type": "function", "function": {...}}` format
//! - Supports `strict` mode for Structured Outputs
//! - Compatible with both Chat Completions and Responses APIs

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Concurrency mode for a tool, describing how it interacts with parallel execution.
///
/// The [`Runner`](crate::agent::Runner) uses this to schedule tool calls intelligently:
/// - `ReadOnly` tools run concurrently with everything except `Exclusive` tools.
/// - `Safe` tools run concurrently with other `Safe` and `ReadOnly` tools.
/// - `Exclusive` tools run one at a time with no other tools executing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConcurrencyMode {
    /// Tool only reads state and can safely run alongside any non-exclusive tool.
    ReadOnly,
    /// Tool may mutate state but is safe to run concurrently with other `Safe`
    /// and `ReadOnly` tools (the default).
    #[default]
    Safe,
    /// Tool requires exclusive access — no other tools run while it executes.
    Exclusive,
}

/// How destructive a tool's effects are.
///
/// This metadata helps the [`Runner`](crate::agent::Runner) and middleware make
/// informed decisions about confirmation, logging, and error handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Destructiveness {
    /// Tool has no destructive side effects (the default).
    #[default]
    None,
    /// Tool's effects can be undone (e.g., file write with backup).
    Reversible,
    /// Tool's effects are permanent (e.g., delete, send email).
    Irreversible,
}

/// How a tool should behave when the agent run is interrupted or cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InterruptBehavior {
    /// Drop the tool call immediately without waiting (the default).
    #[default]
    Drop,
    /// Wait for the tool to finish before acknowledging the interruption.
    WaitComplete,
    /// Actively abort the tool's operation.
    Abort,
}

/// Behavioral metadata for a tool, informing the runner's scheduling and
/// safety decisions.
///
/// Tools provide metadata via [`DynTool::metadata`]. The runner uses this to:
/// - Schedule tool calls with appropriate concurrency
///   ([`ConcurrencyMode`]).
/// - Handle interruptions gracefully ([`InterruptBehavior`]).
/// - Enforce timeouts.
///
/// All fields default to the most permissive/safe values, so tools that don't
/// override [`DynTool::metadata`] behave exactly as before.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    /// How this tool interacts with concurrent execution.
    pub concurrency: ConcurrencyMode,
    /// How destructive the tool's effects are.
    pub destructiveness: Destructiveness,
    /// How the tool should respond to interruptions.
    pub interrupt_behavior: InterruptBehavior,
    /// Optional per-tool execution timeout.
    pub timeout: Option<Duration>,
}

impl Default for ToolMetadata {
    fn default() -> Self {
        Self {
            concurrency: ConcurrencyMode::Safe,
            destructiveness: Destructiveness::None,
            interrupt_behavior: InterruptBehavior::Drop,
            timeout: None,
        }
    }
}

impl ToolMetadata {
    /// Create metadata for a read-only tool.
    #[must_use]
    pub fn read_only() -> Self {
        Self {
            concurrency: ConcurrencyMode::ReadOnly,
            ..Self::default()
        }
    }

    /// Create metadata for an exclusive tool.
    #[must_use]
    pub fn exclusive() -> Self {
        Self {
            concurrency: ConcurrencyMode::Exclusive,
            ..Self::default()
        }
    }

    /// Set the concurrency mode.
    #[must_use]
    pub const fn with_concurrency(mut self, mode: ConcurrencyMode) -> Self {
        self.concurrency = mode;
        self
    }

    /// Set the destructiveness level.
    #[must_use]
    pub const fn with_destructiveness(mut self, level: Destructiveness) -> Self {
        self.destructiveness = level;
        self
    }

    /// Set the interrupt behavior.
    #[must_use]
    pub const fn with_interrupt_behavior(mut self, behavior: InterruptBehavior) -> Self {
        self.interrupt_behavior = behavior;
        self
    }

    /// Set an execution timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// Error type for tool execution failures.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
    /// Error during tool execution.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Invalid arguments provided to the tool.
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    /// Tool not found.
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Tool execution is forbidden by policy.
    #[error("Tool '{0}' is forbidden by policy")]
    Forbidden(String),

    /// Tool execution was denied by human confirmation.
    #[error("Tool '{0}' execution denied by confirmation")]
    ConfirmationDenied(String),

    /// Generic error.
    #[error("Tool error: {0}")]
    Other(String),
}

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

/// A type alias for `Result<T, ToolError>`.
pub type ToolResult<T> = Result<T, ToolError>;

/// Definition of a tool for LLM function calling.
///
/// # `OpenAI` API Alignment
///
/// This type serializes to `OpenAI`'s function calling format:
/// ```json
/// {
///     "type": "function",
///     "function": {
///         "name": "tool_name",
///         "description": "Tool description",
///         "parameters": { ... },
///         "strict": true
///     }
/// }
/// ```
///
/// When `strict` is enabled, the schema should include `"additionalProperties": false`
/// at each object level for proper Structured Outputs support.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct ToolDefinition {
    /// Name of the tool (e.g., "`get_weather`").
    /// Should be descriptive and use `snake_case`.
    pub name: String,

    /// Description of what the tool does.
    /// This helps the model decide when to use the tool.
    pub description: String,

    /// JSON schema for the tool's parameters.
    /// Should follow JSON Schema specification.
    pub parameters: Value,

    /// Whether to use strict schema validation (`OpenAI` Structured Outputs).
    /// When enabled, the model output will exactly match the schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ToolDefinition {
    /// Create a new tool definition.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: None,
        }
    }

    /// Create a tool definition with strict mode enabled.
    #[must_use]
    pub fn new_strict(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: Some(true),
        }
    }

    /// Enable strict schema validation (Structured Outputs).
    ///
    /// When strict mode is enabled:
    /// - The model output will exactly match the provided schema
    /// - All fields become required by default
    /// - `additionalProperties` is automatically set to `false`
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        if strict {
            // Ensure additionalProperties is false for strict mode
            if let Some(obj) = self.parameters.as_object_mut()
                && !obj.contains_key("additionalProperties")
            {
                obj.insert("additionalProperties".to_owned(), Value::Bool(false));
            }
        }
        self
    }

    /// Check if strict mode is enabled.
    #[must_use]
    pub const fn is_strict(&self) -> bool {
        matches!(self.strict, Some(true))
    }

    /// Returns the tool name.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tool description.
    #[inline]
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Custom serialization to `OpenAI` function calling format.
impl Serialize for ToolDefinition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        // Build function object
        let mut function = serde_json::Map::new();
        function.insert("name".to_owned(), Value::String(self.name.clone()));
        function.insert(
            "description".to_owned(),
            Value::String(self.description.clone()),
        );
        function.insert("parameters".to_owned(), self.parameters.clone());
        if let Some(strict) = self.strict {
            function.insert("strict".to_owned(), Value::Bool(strict));
        }

        // Build outer object: {"type": "function", "function": {...}}
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("type", "function")?;
        map.serialize_entry("function", &function)?;
        map.end()
    }
}

/// The core trait for all tools that agents can use.
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
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// Get the description of the tool.
    fn description(&self) -> String;

    /// Get the JSON schema for the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given arguments.
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;

    /// Get the tool definition for LLM function calling.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_owned(),
            description: self.description(),
            parameters: self.parameters_schema(),
            strict: None,
        }
    }

    /// Return behavioral metadata for this tool.
    ///
    /// The default returns [`ToolMetadata::default()`] (safe concurrency, non-destructive,
    /// drop on interrupt, no timeout). Override this to provide tool-specific metadata.
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::default()
    }

    /// Call the tool with JSON arguments and return JSON output.
    async fn call_json(&self, args: Value) -> Result<Value, ToolError>
    where
        Self::Output: 'static,
    {
        // Handle both string and object arguments
        let typed_args: Self::Args = match &args {
            Value::String(s) => {
                serde_json::from_str(s).map_err(|e| ToolError::InvalidArguments(e.to_string()))?
            }
            _ => serde_json::from_value(args)
                .map_err(|e| ToolError::InvalidArguments(e.to_string()))?,
        };

        let result = self.call(typed_args).await.map_err(Into::into)?;
        serde_json::to_value(result).map_err(|e| ToolError::Execution(e.to_string()))
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

    /// Return behavioral metadata for this tool.
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::default()
    }

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

    fn metadata(&self) -> ToolMetadata {
        Tool::metadata(self)
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        Tool::call_json(self, args).await
    }
}

/// Execution policy for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolExecutionPolicy {
    /// Agent can execute the tool autonomously without confirmation.
    #[default]
    Auto,
    /// Requires human confirmation before execution.
    RequireConfirmation,
    /// Tool execution is forbidden.
    Forbidden,
}

impl fmt::Display for ToolExecutionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::RequireConfirmation => write!(f, "require_confirmation"),
            Self::Forbidden => write!(f, "forbidden"),
        }
    }
}

/// Request for human confirmation before tool execution.
#[derive(Debug, Clone)]
pub struct ToolConfirmationRequest {
    /// The tool call ID.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The tool arguments as JSON.
    pub arguments: Value,
    /// Human-readable description of what the tool will do.
    pub description: String,
}

impl ToolConfirmationRequest {
    /// Create a new confirmation request.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        let name = name.into();
        let description = format!(
            "Tool '{}' wants to execute with arguments: {}",
            name,
            serde_json::to_string_pretty(&arguments).unwrap_or_else(|_| arguments.to_string())
        );
        Self {
            id: id.into(),
            name,
            arguments,
            description,
        }
    }
}

/// Response to a tool confirmation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolConfirmationResponse {
    /// User approved the tool execution.
    Approved,
    /// User denied the tool execution.
    Denied,
    /// User approved this and all future calls to this tool.
    ApproveAll,
}

impl ToolConfirmationResponse {
    /// Check if the response approves execution.
    #[must_use]
    pub const fn is_approved(&self) -> bool {
        matches!(self, Self::Approved | Self::ApproveAll)
    }
}

/// Handler for tool execution confirmation requests.
#[async_trait]
pub trait ConfirmationHandler: Send + Sync {
    /// Request confirmation for a tool execution.
    async fn confirm(&self, request: &ToolConfirmationRequest) -> ToolConfirmationResponse;
}

/// A shared confirmation handler for use across cloneable contexts.
pub type SharedConfirmationHandler = std::sync::Arc<dyn ConfirmationHandler>;

/// Default confirmation handler that auto-approves all requests.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoApproveHandler;

#[async_trait]
impl ConfirmationHandler for AutoApproveHandler {
    async fn confirm(&self, _request: &ToolConfirmationRequest) -> ToolConfirmationResponse {
        ToolConfirmationResponse::Approved
    }
}

/// Confirmation handler that always denies execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct AlwaysDenyHandler;

#[async_trait]
impl ConfirmationHandler for AlwaysDenyHandler {
    async fn confirm(&self, _request: &ToolConfirmationRequest) -> ToolConfirmationResponse {
        ToolConfirmationResponse::Denied
    }
}
