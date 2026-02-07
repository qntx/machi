//! Tool trait and utilities for defining agent tools.
//!
//! Tools are the primary way agents interact with the world. Each tool
//! represents a specific capability that an agent can invoke.
//!
//! # OpenAI API Alignment
//!
//! This module aligns with OpenAI's Function Calling API:
//! - `ToolDefinition` serializes to `{"type": "function", "function": {...}}` format
//! - Supports `strict` mode for Structured Outputs
//! - Compatible with both Chat Completions and Responses APIs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

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

impl ToolError {
    /// Create an execution error.
    #[must_use]
    pub fn execution(msg: impl Into<String>) -> Self {
        Self::Execution(msg.into())
    }

    /// Create an invalid arguments error.
    #[must_use]
    pub fn invalid_args(msg: impl Into<String>) -> Self {
        Self::InvalidArguments(msg.into())
    }

    /// Create a not found error.
    #[must_use]
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound(name.into())
    }

    /// Create a forbidden error.
    #[must_use]
    pub fn forbidden(tool_name: impl Into<String>) -> Self {
        Self::Forbidden(tool_name.into())
    }

    /// Create a confirmation denied error.
    #[must_use]
    pub fn confirmation_denied(tool_name: impl Into<String>) -> Self {
        Self::ConfirmationDenied(tool_name.into())
    }
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
/// # OpenAI API Alignment
///
/// This type serializes to OpenAI's function calling format:
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
    /// Name of the tool (e.g., "get_weather").
    /// Should be descriptive and use snake_case.
    pub name: String,

    /// Description of what the tool does.
    /// This helps the model decide when to use the tool.
    pub description: String,

    /// JSON schema for the tool's parameters.
    /// Should follow JSON Schema specification.
    pub parameters: Value,

    /// Whether to use strict schema validation (OpenAI Structured Outputs).
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

/// Custom serialization to OpenAI function calling format.
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

impl ToolExecutionPolicy {
    /// Check if the policy allows autonomous execution.
    #[must_use]
    pub const fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Check if the policy requires confirmation.
    #[must_use]
    pub const fn requires_confirmation(&self) -> bool {
        matches!(self, Self::RequireConfirmation)
    }

    /// Check if the policy forbids execution.
    #[must_use]
    pub const fn is_forbidden(&self) -> bool {
        matches!(self, Self::Forbidden)
    }
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

/// A boxed confirmation handler for dynamic dispatch.
pub type BoxedConfirmationHandler = Box<dyn ConfirmationHandler>;

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

/// Result of a tool call execution.
///
/// # OpenAI API Alignment
///
/// This maps to a tool message in the conversation:
/// ```json
/// {
///     "role": "tool",
///     "tool_call_id": "call_abc123",
///     "content": "{\"result\": ...}"
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// The tool call ID (maps to `tool_call_id` in API).
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The result of execution (success value or error).
    pub result: Result<Value, ToolError>,
}

impl ToolCallResult {
    /// Check if the call was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.result.is_ok()
    }

    /// Get the output value if successful.
    #[must_use]
    pub fn output(&self) -> Option<&Value> {
        self.result.as_ref().ok()
    }

    /// Get the error if failed.
    #[must_use]
    pub fn error(&self) -> Option<&ToolError> {
        self.result.as_ref().err()
    }

    /// Convert to a string representation for the LLM.
    #[must_use]
    pub fn to_string_for_llm(&self) -> String {
        match &self.result {
            Ok(value) => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
            Err(e) => format!("Error: {e}"),
        }
    }

    /// Create a successful result.
    #[must_use]
    pub fn success(id: impl Into<String>, name: impl Into<String>, value: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            result: Ok(value),
        }
    }

    /// Create a failed result.
    #[must_use]
    pub fn failure(id: impl Into<String>, name: impl Into<String>, error: ToolError) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            result: Err(error),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::no_effect_underscore_binding,
    clippy::unnecessary_wraps
)]
mod tests {
    use super::*;

    mod tool_definition {
        use super::*;

        fn sample_parameters() -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            })
        }

        #[test]
        fn new_creates_definition() {
            let def = ToolDefinition::new("get_weather", "Get weather info", sample_parameters());
            assert_eq!(def.name, "get_weather");
            assert_eq!(def.description, "Get weather info");
            assert!(def.strict.is_none());
        }

        #[test]
        fn new_strict_creates_strict_definition() {
            let def =
                ToolDefinition::new_strict("get_weather", "Get weather info", sample_parameters());
            assert_eq!(def.strict, Some(true));
            assert!(def.is_strict());
        }

        #[test]
        fn with_strict_enables_strict_mode() {
            let def =
                ToolDefinition::new("test", "Test tool", sample_parameters()).with_strict(true);
            assert!(def.is_strict());
            // Check additionalProperties was added
            assert_eq!(
                def.parameters.get("additionalProperties"),
                Some(&Value::Bool(false))
            );
        }

        #[test]
        fn with_strict_false_does_not_add_additional_properties() {
            let def =
                ToolDefinition::new("test", "Test tool", sample_parameters()).with_strict(false);
            assert!(!def.is_strict());
            assert!(def.parameters.get("additionalProperties").is_none());
        }

        #[test]
        fn with_strict_preserves_existing_additional_properties() {
            let params = serde_json::json!({
                "type": "object",
                "additionalProperties": true
            });
            let def = ToolDefinition::new("test", "Test", params).with_strict(true);
            // Should preserve existing value
            assert_eq!(
                def.parameters.get("additionalProperties"),
                Some(&Value::Bool(true))
            );
        }

        #[test]
        fn is_strict_returns_false_when_none() {
            let def = ToolDefinition::new("test", "Test", sample_parameters());
            assert!(!def.is_strict());
        }

        #[test]
        fn name_returns_name() {
            let def = ToolDefinition::new("my_tool", "Desc", sample_parameters());
            assert_eq!(def.name(), "my_tool");
        }

        #[test]
        fn description_returns_description() {
            let def = ToolDefinition::new("tool", "My description", sample_parameters());
            assert_eq!(def.description(), "My description");
        }

        #[test]
        fn serialize_to_openai_format() {
            let def = ToolDefinition::new("get_weather", "Get weather", sample_parameters());
            let json = serde_json::to_value(&def).unwrap();

            // Check outer structure
            assert_eq!(
                json.get("type"),
                Some(&Value::String("function".to_owned()))
            );
            assert!(json.get("function").is_some());

            // Check function object
            let function = json.get("function").unwrap();
            assert_eq!(
                function.get("name"),
                Some(&Value::String("get_weather".to_owned()))
            );
            assert_eq!(
                function.get("description"),
                Some(&Value::String("Get weather".to_owned()))
            );
            assert!(function.get("parameters").is_some());
        }

        #[test]
        fn serialize_with_strict() {
            let def = ToolDefinition::new("test", "Test", sample_parameters()).with_strict(true);
            let json = serde_json::to_value(&def).unwrap();
            let function = json.get("function").unwrap();
            assert_eq!(function.get("strict"), Some(&Value::Bool(true)));
        }

        #[test]
        fn serialize_without_strict_omits_field() {
            let def = ToolDefinition::new("test", "Test", sample_parameters());
            let json = serde_json::to_value(&def).unwrap();
            let function = json.get("function").unwrap();
            assert!(function.get("strict").is_none());
        }

        #[test]
        fn deserialize_from_simple_format() {
            let json = r#"{
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object"}
            }"#;
            let def: ToolDefinition = serde_json::from_str(json).unwrap();
            assert_eq!(def.name, "test_tool");
            assert_eq!(def.description, "A test tool");
        }

        #[test]
        fn clone_trait() {
            let def = ToolDefinition::new("test", "Test", sample_parameters());
            let cloned = def.clone();
            assert_eq!(cloned.name, def.name);
            assert_eq!(cloned.description, def.description);
        }
    }

    mod tool_execution_policy {
        use super::*;

        #[test]
        fn default_is_auto() {
            assert_eq!(ToolExecutionPolicy::default(), ToolExecutionPolicy::Auto);
        }

        #[test]
        fn is_auto_returns_true_for_auto() {
            assert!(ToolExecutionPolicy::Auto.is_auto());
            assert!(!ToolExecutionPolicy::RequireConfirmation.is_auto());
            assert!(!ToolExecutionPolicy::Forbidden.is_auto());
        }

        #[test]
        fn requires_confirmation_returns_true_for_require_confirmation() {
            assert!(ToolExecutionPolicy::RequireConfirmation.requires_confirmation());
            assert!(!ToolExecutionPolicy::Auto.requires_confirmation());
            assert!(!ToolExecutionPolicy::Forbidden.requires_confirmation());
        }

        #[test]
        fn is_forbidden_returns_true_for_forbidden() {
            assert!(ToolExecutionPolicy::Forbidden.is_forbidden());
            assert!(!ToolExecutionPolicy::Auto.is_forbidden());
            assert!(!ToolExecutionPolicy::RequireConfirmation.is_forbidden());
        }

        #[test]
        fn display_all_variants() {
            assert_eq!(ToolExecutionPolicy::Auto.to_string(), "auto");
            assert_eq!(
                ToolExecutionPolicy::RequireConfirmation.to_string(),
                "require_confirmation"
            );
            assert_eq!(ToolExecutionPolicy::Forbidden.to_string(), "forbidden");
        }

        #[test]
        fn serde_roundtrip() {
            for policy in [
                ToolExecutionPolicy::Auto,
                ToolExecutionPolicy::RequireConfirmation,
                ToolExecutionPolicy::Forbidden,
            ] {
                let json = serde_json::to_string(&policy).unwrap();
                let parsed: ToolExecutionPolicy = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed, policy);
            }
        }

        #[test]
        fn copy_trait() {
            let policy = ToolExecutionPolicy::RequireConfirmation;
            let copy = policy;
            assert_eq!(policy, copy);
        }
    }

    mod tool_confirmation_request {
        use super::*;

        #[test]
        fn new_creates_request() {
            let args = serde_json::json!({"city": "Tokyo"});
            let request = ToolConfirmationRequest::new("call_123", "get_weather", args.clone());

            assert_eq!(request.id, "call_123");
            assert_eq!(request.name, "get_weather");
            assert_eq!(request.arguments, args);
            assert!(request.description.contains("get_weather"));
            assert!(request.description.contains("Tokyo"));
        }

        #[test]
        fn new_handles_complex_arguments() {
            let args = serde_json::json!({
                "query": "test",
                "options": {"limit": 10}
            });
            let request = ToolConfirmationRequest::new("id", "search", args);
            assert!(request.description.contains("search"));
        }

        #[test]
        fn clone_trait() {
            let request = ToolConfirmationRequest::new("id", "tool", serde_json::json!({}));
            let cloned = request.clone();
            assert_eq!(cloned.id, request.id);
            assert_eq!(cloned.name, request.name);
        }
    }

    mod tool_confirmation_response {
        use super::*;

        #[test]
        fn is_approved_returns_true_for_approved() {
            assert!(ToolConfirmationResponse::Approved.is_approved());
        }

        #[test]
        fn is_approved_returns_true_for_approve_all() {
            assert!(ToolConfirmationResponse::ApproveAll.is_approved());
        }

        #[test]
        fn is_approved_returns_false_for_denied() {
            assert!(!ToolConfirmationResponse::Denied.is_approved());
        }

        #[test]
        fn copy_trait() {
            let response = ToolConfirmationResponse::Approved;
            let copy = response;
            assert_eq!(response, copy);
        }
    }

    mod confirmation_handlers {
        use super::*;

        #[tokio::test]
        async fn auto_approve_handler_approves() {
            let handler = AutoApproveHandler;
            let request = ToolConfirmationRequest::new("id", "tool", serde_json::json!({}));
            let response = handler.confirm(&request).await;
            assert_eq!(response, ToolConfirmationResponse::Approved);
        }

        #[tokio::test]
        async fn always_deny_handler_denies() {
            let handler = AlwaysDenyHandler;
            let request = ToolConfirmationRequest::new("id", "tool", serde_json::json!({}));
            let response = handler.confirm(&request).await;
            assert_eq!(response, ToolConfirmationResponse::Denied);
        }

        #[test]
        fn auto_approve_handler_is_default() {
            let _handler = AutoApproveHandler;
        }

        #[test]
        fn always_deny_handler_is_default() {
            let _handler = AlwaysDenyHandler;
        }

        #[test]
        fn handlers_are_copy() {
            let handler1 = AutoApproveHandler;
            let handler2 = handler1;
            let _ = handler1;
            let _ = handler2;
        }
    }

    mod tool_call_result {
        use super::*;

        #[test]
        fn success_creates_successful_result() {
            let result =
                ToolCallResult::success("call_1", "my_tool", serde_json::json!({"value": 42}));
            assert!(result.is_success());
            assert_eq!(result.id, "call_1");
            assert_eq!(result.name, "my_tool");
            assert!(result.output().is_some());
            assert!(result.error().is_none());
        }

        #[test]
        fn failure_creates_failed_result() {
            let result = ToolCallResult::failure(
                "call_2",
                "my_tool",
                ToolError::Execution("failed".to_owned()),
            );
            assert!(!result.is_success());
            assert!(result.output().is_none());
            assert!(result.error().is_some());
        }

        #[test]
        fn output_returns_value_on_success() {
            let value = serde_json::json!({"result": "ok"});
            let result = ToolCallResult::success("id", "tool", value.clone());
            assert_eq!(result.output(), Some(&value));
        }

        #[test]
        fn error_returns_error_on_failure() {
            let result =
                ToolCallResult::failure("id", "tool", ToolError::NotFound("missing".to_owned()));
            let err = result.error().unwrap();
            assert!(matches!(err, ToolError::NotFound(_)));
        }

        #[test]
        fn to_string_for_llm_success() {
            let result = ToolCallResult::success("id", "tool", serde_json::json!({"value": 42}));
            let llm_str = result.to_string_for_llm();
            assert!(llm_str.contains("42"));
        }

        #[test]
        fn to_string_for_llm_failure() {
            let result = ToolCallResult::failure(
                "id",
                "tool",
                ToolError::Execution("Something went wrong".to_owned()),
            );
            let llm_str = result.to_string_for_llm();
            assert!(llm_str.contains("Error"));
            assert!(llm_str.contains("Something went wrong"));
        }

        #[test]
        fn clone_trait() {
            let result = ToolCallResult::success("id", "tool", serde_json::json!({}));
            let cloned = result.clone();
            assert_eq!(cloned.id, result.id);
            assert_eq!(cloned.name, result.name);
        }
    }

    mod tool_result_type {
        use super::*;

        #[test]
        fn tool_result_is_result_type() {
            fn returns_tool_result() -> ToolResult<i32> {
                Ok(42)
            }
            assert_eq!(returns_tool_result().unwrap(), 42);
        }

        #[test]
        fn tool_result_can_hold_error() {
            fn returns_error() -> ToolResult<()> {
                Err(ToolError::Execution("failed".to_owned()))
            }
            assert!(returns_error().is_err());
        }
    }
}
