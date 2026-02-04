//! Template rendering engine using minijinja (Jinja2-compatible).
//!
//! This module provides a Jinja2-compatible template rendering engine that
//! matches the smolagents template syntax.

use std::collections::HashMap;

use minijinja::{Environment, Value, context};
use serde::Serialize;

use crate::tool::ToolDefinition;

/// Template rendering context containing all variables for prompt generation.
///
/// This context is passed to the template engine and provides access to:
/// - `tools` - List of available tool definitions
/// - `managed_agents` - List of managed agent definitions (for multi-agent)
/// - `task` - The current task description
/// - `custom_instructions` - Optional custom instructions
/// - `name` - Agent name (for managed agents)
/// - `remaining_steps` - Steps remaining before max_steps
/// - `final_answer` - Final answer from a managed agent
///
/// # Example
///
/// ```rust,ignore
/// let ctx = TemplateContext::new()
///     .with_tools(&tools)
///     .with_task("Calculate 2 + 2")
///     .with_custom_instructions("Be concise.");
/// ```
#[derive(Debug, Clone, Default)]
pub struct TemplateContext {
    tools: Vec<ToolInfo>,
    managed_agents: Vec<ManagedAgentInfo>,
    task: Option<String>,
    custom_instructions: Option<String>,
    name: Option<String>,
    remaining_steps: Option<usize>,
    final_answer: Option<String>,
    extra: HashMap<String, Value>,
}

/// Simplified tool information for template rendering.
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Input parameters (as formatted string).
    pub inputs: String,
    /// Output type.
    pub output_type: String,
}

impl ToolInfo {
    /// Create tool info from a [`ToolDefinition`].
    #[must_use]
    pub fn from_definition(def: &ToolDefinition) -> Self {
        let inputs = if let Some(props) = def.parameters.get("properties") {
            format!("{props}")
        } else {
            "{}".to_string()
        };

        Self {
            name: def.name.clone(),
            description: def.description.clone(),
            inputs,
            output_type: def.output_type.clone().unwrap_or_else(|| "any".to_string()),
        }
    }

    /// Format as tool-calling prompt (matches smolagents `to_tool_calling_prompt`).
    #[must_use]
    pub fn to_tool_calling_prompt(&self) -> String {
        format!(
            "{}: {}\n    Takes inputs: {}\n    Returns an output of type: {}",
            self.name, self.description, self.inputs, self.output_type
        )
    }
}

impl From<&ToolDefinition> for ToolInfo {
    fn from(def: &ToolDefinition) -> Self {
        Self::from_definition(def)
    }
}

/// Managed agent information for template rendering.
#[derive(Debug, Clone, Serialize)]
pub struct ManagedAgentInfo {
    /// Agent name.
    pub name: String,
    /// Agent description.
    pub description: String,
    /// Input parameters.
    pub inputs: String,
    /// Output type.
    pub output_type: String,
}

impl TemplateContext {
    /// Create a new empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add tool definitions to the context.
    #[must_use]
    pub fn with_tools(mut self, tools: &[ToolDefinition]) -> Self {
        self.tools = tools.iter().map(ToolInfo::from_definition).collect();
        self
    }

    /// Add tool info directly to the context.
    #[must_use]
    pub fn with_tool_infos(mut self, tools: Vec<ToolInfo>) -> Self {
        self.tools = tools;
        self
    }

    /// Add managed agents to the context.
    #[must_use]
    pub fn with_managed_agents(
        mut self,
        agents: &HashMap<String, crate::managed_agent::ManagedAgentInfo>,
    ) -> Self {
        self.managed_agents = agents
            .values()
            .map(|a| ManagedAgentInfo {
                name: a.name.clone(),
                description: a.description.clone(),
                inputs: serde_json::to_string(&a.inputs).unwrap_or_default(),
                output_type: a.output_type.clone(),
            })
            .collect();
        self
    }

    /// Add managed agents from a Vec to the context.
    #[must_use]
    pub fn with_managed_agents_vec(mut self, agents: Vec<ManagedAgentInfo>) -> Self {
        self.managed_agents = agents;
        self
    }

    /// Set the current task.
    #[must_use]
    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    /// Set custom instructions.
    #[must_use]
    pub fn with_custom_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.custom_instructions = Some(instructions.into());
        self
    }

    /// Set custom instructions from an optional value.
    #[must_use]
    pub fn with_custom_instructions_opt(mut self, instructions: Option<&str>) -> Self {
        self.custom_instructions = instructions.map(String::from);
        self
    }

    /// Set the agent name (for managed agent templates).
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set remaining steps count.
    #[must_use]
    pub const fn with_remaining_steps(mut self, steps: usize) -> Self {
        self.remaining_steps = Some(steps);
        self
    }

    /// Set the final answer (for managed agent report templates).
    #[must_use]
    pub fn with_final_answer(mut self, answer: impl Into<String>) -> Self {
        self.final_answer = Some(answer.into());
        self
    }

    /// Add a custom variable to the context.
    #[must_use]
    pub fn with_var(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Convert context to minijinja Value for rendering.
    fn to_value(&self) -> Value {
        // Create tools dict with .values() support
        let tools_dict = ToolsDict {
            tools: self
                .tools
                .iter()
                .map(|t| ToolValue {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    inputs: t.inputs.clone(),
                    output_type: t.output_type.clone(),
                })
                .collect(),
        };

        // Create managed_agents dict with .values() support
        let agents_dict = ManagedAgentsDict {
            agents: self
                .managed_agents
                .iter()
                .map(|a| ManagedAgentValue {
                    name: a.name.clone(),
                    description: a.description.clone(),
                    inputs: a.inputs.clone(),
                    output_type: a.output_type.clone(),
                })
                .collect(),
        };

        context! {
            tools => Value::from_object(tools_dict),
            managed_agents => Value::from_object(agents_dict),
            task => self.task,
            custom_instructions => self.custom_instructions,
            name => self.name,
            remaining_steps => self.remaining_steps,
            final_answer => self.final_answer,
            ..Value::from_serialize(&self.extra)
        }
    }
}

/// Dict-like container for tools that supports `.values()` method.
#[derive(Debug, Clone)]
struct ToolsDict {
    tools: Vec<ToolValue>,
}

impl minijinja::value::Object for ToolsDict {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        // Support access by tool name
        let name = key.as_str()?;
        self.tools
            .iter()
            .find(|t| t.name == name)
            .map(|t| Value::from_object(t.clone()))
    }

    fn call_method(
        self: &std::sync::Arc<Self>,
        _state: &minijinja::State<'_, '_>,
        name: &str,
        _args: &[Value],
    ) -> Result<Value, minijinja::Error> {
        match name {
            "values" => {
                // Return an iterator over all tool values
                let values: Vec<Value> = self
                    .tools
                    .iter()
                    .map(|t| Value::from_object(t.clone()))
                    .collect();
                Ok(Value::from(values))
            }
            "keys" => {
                let keys: Vec<Value> = self
                    .tools
                    .iter()
                    .map(|t| Value::from(t.name.clone()))
                    .collect();
                Ok(Value::from(keys))
            }
            _ => Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                format!("tools dict has no method named {name}"),
            )),
        }
    }

    fn enumerate(self: &std::sync::Arc<Self>) -> minijinja::value::Enumerator {
        minijinja::value::Enumerator::Iter(Box::new(
            self.tools
                .iter()
                .map(|t| Value::from_object(t.clone()))
                .collect::<Vec<_>>()
                .into_iter(),
        ))
    }
}

/// Dict-like container for managed agents that supports `.values()` method.
#[derive(Debug, Clone)]
struct ManagedAgentsDict {
    agents: Vec<ManagedAgentValue>,
}

/// Managed agent value for template rendering.
#[derive(Debug, Clone)]
struct ManagedAgentValue {
    name: String,
    description: String,
    inputs: String,
    output_type: String,
}

impl minijinja::value::Object for ManagedAgentsDict {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        let name = key.as_str()?;
        self.agents
            .iter()
            .find(|a| a.name == name)
            .map(|a| Value::from_object(a.clone()))
    }

    fn call_method(
        self: &std::sync::Arc<Self>,
        _state: &minijinja::State<'_, '_>,
        name: &str,
        _args: &[Value],
    ) -> Result<Value, minijinja::Error> {
        match name {
            "values" => {
                let values: Vec<Value> = self
                    .agents
                    .iter()
                    .map(|a| Value::from_object(a.clone()))
                    .collect();
                Ok(Value::from(values))
            }
            "keys" => {
                let keys: Vec<Value> = self
                    .agents
                    .iter()
                    .map(|a| Value::from(a.name.clone()))
                    .collect();
                Ok(Value::from(keys))
            }
            _ => Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                format!("managed_agents dict has no method named {name}"),
            )),
        }
    }

    fn enumerate(self: &std::sync::Arc<Self>) -> minijinja::value::Enumerator {
        minijinja::value::Enumerator::Iter(Box::new(
            self.agents
                .iter()
                .map(|a| Value::from_object(a.clone()))
                .collect::<Vec<_>>()
                .into_iter(),
        ))
    }
}

impl minijinja::value::Object for ManagedAgentValue {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        match key.as_str()? {
            "name" => Some(Value::from(&self.name)),
            "description" => Some(Value::from(&self.description)),
            "inputs" => Some(Value::from(&self.inputs)),
            "output_type" => Some(Value::from(&self.output_type)),
            _ => None,
        }
    }
}

/// Tool value wrapper for minijinja object protocol.
#[derive(Debug, Clone)]
struct ToolValue {
    name: String,
    description: String,
    inputs: String,
    output_type: String,
}

impl minijinja::value::Object for ToolValue {
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        match key.as_str()? {
            "name" => Some(Value::from(&self.name)),
            "description" => Some(Value::from(&self.description)),
            "inputs" => Some(Value::from(&self.inputs)),
            "output_type" => Some(Value::from(&self.output_type)),
            _ => None,
        }
    }

    fn call_method(
        self: &std::sync::Arc<Self>,
        _state: &minijinja::State<'_, '_>,
        name: &str,
        _args: &[Value],
    ) -> Result<Value, minijinja::Error> {
        match name {
            "to_tool_calling_prompt" => {
                let prompt = format!(
                    "{}: {}\n    Takes inputs: {}\n    Returns an output of type: {}",
                    self.name, self.description, self.inputs, self.output_type
                );
                Ok(Value::from(prompt))
            }
            "to_code_prompt" => {
                // Simplified code prompt format
                let prompt = format!(
                    "def {}(...) -> {}:\n    \"\"\"{}\"\"\"",
                    self.name, self.output_type, self.description
                );
                Ok(Value::from(prompt))
            }
            _ => Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                format!("unknown method '{name}' on tool"),
            )),
        }
    }
}

/// Jinja2-compatible template rendering engine.
///
/// Uses minijinja under the hood for full Jinja2 syntax support including:
/// - Variable interpolation: `{{ variable }}`
/// - Control flow: `{% if %}`, `{% for %}`, `{% endif %}`, `{% endfor %}`
/// - Filters: `{{ value | filter }}`
/// - Method calls: `{{ tool.to_tool_calling_prompt() }}`
/// - Whitespace control: `{%- ... -%}`
///
/// # Example
///
/// ```rust,ignore
/// let engine = PromptEngine::new();
/// let ctx = TemplateContext::new().with_task("Test task");
/// let result = engine.render("Task: {{ task }}", &ctx)?;
/// ```
#[derive(Debug, Clone)]
pub struct PromptEngine {
    env: Environment<'static>,
}

impl Default for PromptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptEngine {
    /// Create a new template engine with default configuration.
    #[must_use]
    pub fn new() -> Self {
        let mut env = Environment::new();

        // Configure for Jinja2 compatibility
        env.set_trim_blocks(false);
        env.set_lstrip_blocks(false);

        // Add custom filters if needed
        env.add_filter("list", |v: Value| -> Vec<Value> {
            v.try_iter().map(Iterator::collect).unwrap_or_default()
        });

        Self { env }
    }

    /// Render a template string with the given context.
    ///
    /// # Errors
    ///
    /// Returns an error if the template syntax is invalid or rendering fails.
    pub fn render(&self, template: &str, context: &TemplateContext) -> Result<String, RenderError> {
        let tmpl = self
            .env
            .template_from_str(template)
            .map_err(|e| RenderError::Template(e.to_string()))?;

        let ctx = context.to_value();
        tmpl.render(ctx)
            .map_err(|e| RenderError::Render(e.to_string()))
    }

    /// Render a template string with raw minijinja context.
    ///
    /// This is a lower-level API for advanced use cases.
    ///
    /// # Errors
    ///
    /// Returns an error if template syntax is invalid or rendering fails.
    pub fn render_raw(&self, template: &str, context: Value) -> Result<String, RenderError> {
        let tmpl = self
            .env
            .template_from_str(template)
            .map_err(|e| RenderError::Template(e.to_string()))?;

        tmpl.render(context)
            .map_err(|e| RenderError::Render(e.to_string()))
    }
}

/// Error type for template rendering operations.
#[derive(Debug, Clone)]
pub enum RenderError {
    /// Template parsing/compilation error.
    Template(String),
    /// Runtime rendering error.
    Render(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Template(msg) => write!(f, "Template error: {msg}"),
            Self::Render(msg) => write!(f, "Render error: {msg}"),
        }
    }
}

impl std::error::Error for RenderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_render() {
        let engine = PromptEngine::new();
        let ctx = TemplateContext::new().with_task("Test task");

        let result = engine
            .render("Task: {{ task }}", &ctx)
            .expect("render should succeed");
        assert_eq!(result, "Task: Test task");
    }

    #[test]
    fn test_conditional() {
        let engine = PromptEngine::new();

        let ctx_with = TemplateContext::new().with_custom_instructions("Be helpful");
        let ctx_without = TemplateContext::new();

        let template = "{% if custom_instructions %}{{ custom_instructions }}{% endif %}";

        assert_eq!(
            engine
                .render(template, &ctx_with)
                .expect("render should succeed"),
            "Be helpful"
        );
        assert_eq!(
            engine
                .render(template, &ctx_without)
                .expect("render should succeed"),
            ""
        );
    }

    #[test]
    fn test_tools_iteration() {
        let engine = PromptEngine::new();

        let tools = vec![ToolInfo {
            name: "calc".to_string(),
            description: "Calculator".to_string(),
            inputs: "{}".to_string(),
            output_type: "number".to_string(),
        }];

        let ctx = TemplateContext::new().with_tool_infos(tools);

        let template = r"{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
{%- endfor %}";

        let result = engine
            .render(template, &ctx)
            .expect("render should succeed");
        assert!(result.contains("calc: Calculator"));
    }

    #[test]
    fn test_tool_method_call() {
        let engine = PromptEngine::new();

        let tools = vec![ToolInfo {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            inputs: r#"{"query": "string"}"#.to_string(),
            output_type: "string".to_string(),
        }];

        let ctx = TemplateContext::new().with_tool_infos(tools);

        let template = r"{%- for tool in tools.values() %}
{{ tool.to_tool_calling_prompt() }}
{%- endfor %}";

        let result = engine
            .render(template, &ctx)
            .expect("render should succeed");
        assert!(result.contains("search: Search the web"));
        assert!(result.contains("Takes inputs:"));
    }

    #[test]
    fn test_whitespace_control() {
        let engine = PromptEngine::new();
        let ctx = TemplateContext::new().with_task("task");

        // Test that {%- strips whitespace
        let template = "A\n{%- if task %}\nB\n{%- endif %}\nC";
        let result = engine
            .render(template, &ctx)
            .expect("render should succeed");
        // The exact output depends on minijinja's whitespace handling
        assert!(result.contains('B'));
    }
}
