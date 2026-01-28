//! Tool definition types for LLM function calling.
//!
//! This module provides the core types for defining tools that can be
//! used by LLM backends. These definitions are backend-agnostic and
//! can be converted to rig, async-openai, or other formats.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool parameter definition for LLM function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    /// Parameter name.
    pub name: String,
    /// Parameter type (e.g., "string", "number", "boolean", "object").
    pub r#type: String,
    /// Parameter description.
    pub description: String,
    /// Whether this parameter is required.
    pub required: bool,
}

/// A tool definition that describes a tool's interface for LLMs.
///
/// This is backend-agnostic and can be converted to various formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (should be snake_case, e.g., "get_balance").
    pub name: String,
    /// Tool description for the LLM.
    pub description: String,
    /// Tool parameters.
    pub parameters: Vec<ToolParam>,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
        }
    }

    /// Add a required parameter.
    pub fn param(
        mut self,
        name: impl Into<String>,
        r#type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters.push(ToolParam {
            name: name.into(),
            r#type: r#type.into(),
            description: description.into(),
            required: true,
        });
        self
    }

    /// Add an optional parameter.
    pub fn optional_param(
        mut self,
        name: impl Into<String>,
        r#type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.parameters.push(ToolParam {
            name: name.into(),
            r#type: r#type.into(),
            description: description.into(),
            required: false,
        });
        self
    }

    /// Convert to JSON schema format (OpenAI compatible).
    pub fn to_json_schema(&self) -> Value {
        let properties: serde_json::Map<String, Value> = self
            .parameters
            .iter()
            .map(|p| {
                (
                    p.name.clone(),
                    serde_json::json!({
                        "type": p.r#type,
                        "description": p.description
                    }),
                )
            })
            .collect();

        let required: Vec<&str> = self
            .parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect();

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        })
    }
}

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Whether the execution was successful.
    pub success: bool,
    /// The result data or error message.
    pub data: Value,
}

impl ToolOutput {
    /// Create a successful result.
    pub fn ok(data: impl Serialize) -> Self {
        Self {
            success: true,
            data: serde_json::to_value(data).unwrap_or(Value::Null),
        }
    }

    /// Create a failed result.
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: Value::String(message.into()),
        }
    }

    /// Convert to string for LLM response.
    pub fn to_string(&self) -> String {
        if self.success {
            serde_json::to_string(&self.data).unwrap_or_default()
        } else {
            format!("Error: {}", self.data)
        }
    }
}
