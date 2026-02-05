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
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::error::ToolError;

/// A type alias for `Result<T, ToolError>`.
pub type ToolResult<T> = Result<T, ToolError>;

/// Type of tool in the OpenAI API.
///
/// Currently only "function" is supported, but this enum allows
/// for future extensibility (e.g., "custom" tools with grammars).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ToolType {
    /// A function tool defined by JSON schema.
    #[default]
    Function,
}

impl ToolType {
    /// Returns the string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
        }
    }
}

impl fmt::Display for ToolType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

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

    /// Output type string for LLM prompts (internal use).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_type: Option<String>,
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
            output_type: None,
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
            output_type: None,
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

    /// Set the output type.
    #[must_use]
    pub fn with_output_type(mut self, output_type: impl Into<String>) -> Self {
        self.output_type = Some(output_type.into());
        self
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

    /// Get the output type string for LLM prompts.
    fn output_type(&self) -> &'static str {
        "object"
    }

    /// Get the JSON schema for the tool's output (optional).
    fn output_schema(&self) -> Option<Value> {
        None
    }

    /// Execute the tool with the given arguments.
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;

    /// Get the tool definition for LLM function calling.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_owned(),
            description: self.description(),
            parameters: self.parameters_schema(),
            strict: None,
            output_type: Some(self.output_type().to_owned()),
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

/// A collection of tools that can be used by an agent.
#[derive(Default)]
pub struct ToolBox {
    tools: HashMap<String, BoxedTool>,
    policies: HashMap<String, ToolExecutionPolicy>,
    auto_approved: HashSet<String>,
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
        self.tools.insert(tool.name().to_owned(), Box::new(tool));
    }

    /// Add a tool with an execution policy.
    pub fn add_with_policy<T: Tool + 'static>(&mut self, tool: T, policy: ToolExecutionPolicy)
    where
        T::Output: 'static,
    {
        let name = tool.name().to_owned();
        self.tools.insert(name.clone(), Box::new(tool));
        self.policies.insert(name, policy);
    }

    /// Add a boxed tool to the toolbox.
    pub fn add_boxed(&mut self, tool: BoxedTool) {
        self.tools.insert(tool.name().to_owned(), tool);
    }

    /// Add a boxed tool with an execution policy.
    pub fn add_boxed_with_policy(&mut self, tool: BoxedTool, policy: ToolExecutionPolicy) {
        let name = tool.name().to_owned();
        self.tools.insert(name.clone(), tool);
        self.policies.insert(name, policy);
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

    /// Set the execution policy for a tool.
    pub fn set_policy(&mut self, tool_name: impl Into<String>, policy: ToolExecutionPolicy) {
        self.policies.insert(tool_name.into(), policy);
    }

    /// Get the execution policy for a tool.
    #[must_use]
    pub fn get_policy(&self, tool_name: &str) -> ToolExecutionPolicy {
        if self.auto_approved.contains(tool_name) {
            return ToolExecutionPolicy::Auto;
        }
        self.policies
            .get(tool_name)
            .copied()
            .unwrap_or(ToolExecutionPolicy::Auto)
    }

    /// Mark a tool as auto-approved.
    pub fn mark_auto_approved(&mut self, tool_name: impl Into<String>) {
        self.auto_approved.insert(tool_name.into());
    }

    /// Check if a tool requires confirmation.
    #[must_use]
    pub fn requires_confirmation(&self, tool_name: &str) -> bool {
        !self.auto_approved.contains(tool_name)
            && self
                .policies
                .get(tool_name)
                .is_some_and(ToolExecutionPolicy::requires_confirmation)
    }

    /// Check if a tool is forbidden.
    #[must_use]
    pub fn is_forbidden(&self, tool_name: &str) -> bool {
        self.policies
            .get(tool_name)
            .is_some_and(ToolExecutionPolicy::is_forbidden)
    }

    /// Call a tool by name with JSON arguments.
    pub async fn call(&self, name: &str, args: Value) -> Result<Value, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_owned()))?;
        tool.call_json(args).await
    }

    /// Returns all tool definitions for use in chat requests.
    #[must_use]
    pub fn to_tools(&self) -> Vec<ToolDefinition> {
        self.definitions()
    }
}

impl fmt::Debug for ToolBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolBox")
            .field("tools", &self.names())
            .field("policies", &self.policies)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod tool_type {
        use super::*;

        #[test]
        fn as_str_returns_function() {
            assert_eq!(ToolType::Function.as_str(), "function");
        }

        #[test]
        fn display_matches_as_str() {
            assert_eq!(ToolType::Function.to_string(), "function");
        }

        #[test]
        fn default_is_function() {
            assert_eq!(ToolType::default(), ToolType::Function);
        }

        #[test]
        fn serde_roundtrip() {
            let tool_type = ToolType::Function;
            let json = serde_json::to_string(&tool_type).unwrap();
            assert_eq!(json, r#""function""#);
            let parsed: ToolType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, tool_type);
        }

        #[test]
        fn copy_trait() {
            let tool_type = ToolType::Function;
            let copy = tool_type;
            assert_eq!(tool_type, copy);
        }
    }

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
            assert!(def.output_type.is_none());
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
        fn with_output_type_sets_value() {
            let def = ToolDefinition::new("test", "Test", sample_parameters())
                .with_output_type("WeatherResult");
            assert_eq!(def.output_type, Some("WeatherResult".to_owned()));
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
            let _handler = AutoApproveHandler::default();
        }

        #[test]
        fn always_deny_handler_is_default() {
            let _handler = AlwaysDenyHandler::default();
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

    mod tool_box {
        use super::*;

        // Mock tool for testing
        struct MockTool {
            name: &'static str,
        }

        #[async_trait]
        impl Tool for MockTool {
            const NAME: &'static str = "mock_tool";
            type Args = Value;
            type Output = Value;
            type Error = ToolError;

            fn name(&self) -> &'static str {
                self.name
            }

            fn description(&self) -> String {
                format!("Mock tool: {}", self.name)
            }

            fn parameters_schema(&self) -> Value {
                serde_json::json!({"type": "object"})
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                Ok(serde_json::json!({"received": args}))
            }
        }

        #[test]
        fn new_creates_empty_toolbox() {
            let toolbox = ToolBox::new();
            assert!(toolbox.is_empty());
            assert_eq!(toolbox.len(), 0);
        }

        #[test]
        fn default_creates_empty_toolbox() {
            let toolbox = ToolBox::default();
            assert!(toolbox.is_empty());
        }

        #[test]
        fn add_inserts_tool() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "test_tool" });
            assert!(!toolbox.is_empty());
            assert_eq!(toolbox.len(), 1);
            assert!(toolbox.contains("test_tool"));
        }

        #[test]
        fn add_with_policy_sets_policy() {
            let mut toolbox = ToolBox::new();
            toolbox.add_with_policy(
                MockTool { name: "dangerous" },
                ToolExecutionPolicy::RequireConfirmation,
            );
            assert!(toolbox.requires_confirmation("dangerous"));
        }

        #[test]
        fn get_returns_tool() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "my_tool" });
            let tool = toolbox.get("my_tool");
            assert!(tool.is_some());
            assert_eq!(tool.unwrap().name(), "my_tool");
        }

        #[test]
        fn get_returns_none_for_missing() {
            let toolbox = ToolBox::new();
            assert!(toolbox.get("nonexistent").is_none());
        }

        #[test]
        fn definitions_returns_all_definitions() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "tool1" });
            toolbox.add(MockTool { name: "tool2" });
            let defs = toolbox.definitions();
            assert_eq!(defs.len(), 2);
        }

        #[test]
        fn names_returns_all_names() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "alpha" });
            toolbox.add(MockTool { name: "beta" });
            let names = toolbox.names();
            assert_eq!(names.len(), 2);
            assert!(names.contains(&"alpha"));
            assert!(names.contains(&"beta"));
        }

        #[test]
        fn contains_checks_existence() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "exists" });
            assert!(toolbox.contains("exists"));
            assert!(!toolbox.contains("missing"));
        }

        #[test]
        fn set_policy_updates_policy() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "tool" });
            toolbox.set_policy("tool", ToolExecutionPolicy::Forbidden);
            assert!(toolbox.is_forbidden("tool"));
        }

        #[test]
        fn get_policy_returns_auto_by_default() {
            let toolbox = ToolBox::new();
            assert_eq!(toolbox.get_policy("unknown"), ToolExecutionPolicy::Auto);
        }

        #[test]
        fn get_policy_returns_set_policy() {
            let mut toolbox = ToolBox::new();
            toolbox.set_policy("tool", ToolExecutionPolicy::RequireConfirmation);
            assert_eq!(
                toolbox.get_policy("tool"),
                ToolExecutionPolicy::RequireConfirmation
            );
        }

        #[test]
        fn mark_auto_approved_overrides_confirmation() {
            let mut toolbox = ToolBox::new();
            toolbox.set_policy("tool", ToolExecutionPolicy::RequireConfirmation);
            assert!(toolbox.requires_confirmation("tool"));

            toolbox.mark_auto_approved("tool");
            assert!(!toolbox.requires_confirmation("tool"));
            assert_eq!(toolbox.get_policy("tool"), ToolExecutionPolicy::Auto);
        }

        #[test]
        fn requires_confirmation_respects_policy() {
            let mut toolbox = ToolBox::new();
            toolbox.set_policy("confirm_tool", ToolExecutionPolicy::RequireConfirmation);
            toolbox.set_policy("auto_tool", ToolExecutionPolicy::Auto);

            assert!(toolbox.requires_confirmation("confirm_tool"));
            assert!(!toolbox.requires_confirmation("auto_tool"));
            assert!(!toolbox.requires_confirmation("unknown"));
        }

        #[test]
        fn is_forbidden_respects_policy() {
            let mut toolbox = ToolBox::new();
            toolbox.set_policy("forbidden_tool", ToolExecutionPolicy::Forbidden);
            toolbox.set_policy("allowed_tool", ToolExecutionPolicy::Auto);

            assert!(toolbox.is_forbidden("forbidden_tool"));
            assert!(!toolbox.is_forbidden("allowed_tool"));
            assert!(!toolbox.is_forbidden("unknown"));
        }

        #[test]
        fn to_tools_returns_definitions() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "tool1" });
            let tools = toolbox.to_tools();
            assert_eq!(tools.len(), 1);
        }

        #[tokio::test]
        async fn call_executes_tool() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "echo" });

            let result = toolbox
                .call("echo", serde_json::json!({"input": "hello"}))
                .await;
            assert!(result.is_ok());
            let value = result.unwrap();
            assert!(value.get("received").is_some());
        }

        #[tokio::test]
        async fn call_returns_error_for_missing_tool() {
            let toolbox = ToolBox::new();
            let result = toolbox.call("nonexistent", serde_json::json!({})).await;
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), ToolError::NotFound(_)));
        }

        #[test]
        fn debug_format() {
            let mut toolbox = ToolBox::new();
            toolbox.add(MockTool { name: "test" });
            let debug = format!("{:?}", toolbox);
            assert!(debug.contains("ToolBox"));
            assert!(debug.contains("test"));
        }

        #[test]
        fn add_boxed_inserts_tool() {
            let mut toolbox = ToolBox::new();
            let tool: BoxedTool = Box::new(MockTool { name: "boxed" });
            toolbox.add_boxed(tool);
            assert!(toolbox.contains("boxed"));
        }

        #[test]
        fn add_boxed_with_policy_sets_policy() {
            let mut toolbox = ToolBox::new();
            let tool: BoxedTool = Box::new(MockTool { name: "boxed" });
            toolbox.add_boxed_with_policy(tool, ToolExecutionPolicy::Forbidden);
            assert!(toolbox.is_forbidden("boxed"));
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

    mod integration {
        use super::*;

        struct CalculatorTool;

        #[derive(Deserialize)]
        struct CalcArgs {
            a: i64,
            b: i64,
            op: String,
        }

        #[derive(Serialize)]
        struct CalcResult {
            result: i64,
        }

        #[async_trait]
        impl Tool for CalculatorTool {
            const NAME: &'static str = "calculator";
            type Args = CalcArgs;
            type Output = CalcResult;
            type Error = ToolError;

            fn description(&self) -> String {
                "Perform basic arithmetic".to_owned()
            }

            fn parameters_schema(&self) -> Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                        "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]}
                    },
                    "required": ["a", "b", "op"]
                })
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                let result = match args.op.as_str() {
                    "add" => args.a + args.b,
                    "sub" => args.a - args.b,
                    "mul" => args.a * args.b,
                    "div" => {
                        if args.b == 0 {
                            return Err(ToolError::Execution("Division by zero".to_owned()));
                        }
                        args.a / args.b
                    }
                    _ => return Err(ToolError::InvalidArguments("Unknown operation".to_owned())),
                };
                Ok(CalcResult { result })
            }
        }

        #[test]
        fn tool_definition_generation() {
            let tool = CalculatorTool;
            let def = Tool::definition(&tool);
            assert_eq!(def.name, "calculator");
            assert_eq!(def.description, "Perform basic arithmetic");
            assert!(def.parameters.get("properties").is_some());
        }

        #[tokio::test]
        async fn tool_execution_success() {
            let tool = CalculatorTool;
            let args = serde_json::json!({"a": 10, "b": 5, "op": "add"});
            let result = Tool::call_json(&tool, args).await.unwrap();
            assert_eq!(result.get("result"), Some(&serde_json::json!(15)));
        }

        #[tokio::test]
        async fn tool_execution_error() {
            let tool = CalculatorTool;
            let args = serde_json::json!({"a": 10, "b": 0, "op": "div"});
            let result: Result<Value, ToolError> = Tool::call_json(&tool, args).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn tool_with_string_args() {
            let tool = CalculatorTool;
            let args = Value::String(r#"{"a": 3, "b": 4, "op": "mul"}"#.to_owned());
            let result = Tool::call_json(&tool, args).await.unwrap();
            assert_eq!(result.get("result"), Some(&serde_json::json!(12)));
        }

        #[tokio::test]
        async fn toolbox_workflow() {
            let mut toolbox = ToolBox::new();
            toolbox.add_with_policy(CalculatorTool, ToolExecutionPolicy::Auto);

            // Check tool exists
            assert!(toolbox.contains("calculator"));

            // Get definition for chat request
            let definitions = toolbox.to_tools();
            assert_eq!(definitions.len(), 1);

            // Execute tool
            let args = serde_json::json!({"a": 7, "b": 3, "op": "sub"});
            let result = toolbox.call("calculator", args).await.unwrap();
            assert_eq!(result.get("result"), Some(&serde_json::json!(4)));
        }

        #[test]
        fn openai_format_serialization() {
            let tool = CalculatorTool;
            let def = Tool::definition(&tool);
            let json = serde_json::to_value(&def).unwrap();

            // Verify OpenAI format
            assert_eq!(json["type"], "function");
            assert!(json["function"].is_object());
            assert_eq!(json["function"]["name"], "calculator");
            assert!(json["function"]["parameters"].is_object());
        }

        #[test]
        fn tool_call_result_workflow() {
            // Simulate successful tool execution
            let success_result = ToolCallResult::success(
                "call_abc123",
                "calculator",
                serde_json::json!({"result": 42}),
            );
            assert!(success_result.is_success());
            let llm_output = success_result.to_string_for_llm();
            assert!(llm_output.contains("42"));

            // Simulate failed tool execution
            let error_result = ToolCallResult::failure(
                "call_def456",
                "calculator",
                ToolError::Execution("Division by zero".to_owned()),
            );
            assert!(!error_result.is_success());
            let llm_error = error_result.to_string_for_llm();
            assert!(llm_error.contains("Error"));
        }
    }
}
