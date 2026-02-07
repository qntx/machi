//! Agent configuration types.
//!
//! The [`Agent`] struct defines an agent's identity, behavior, capabilities,
//! and its own LLM provider. Each agent is self-contained and can be run
//! independently via [`Agent::run`], or orchestrated by a [`Runner`](super::Runner).
//!
//! # Design Philosophy
//!
//! Each agent owns its [`SharedChatProvider`], enabling heterogeneous multi-agent
//! systems where different agents use different LLMs (e.g., GPT-4o for reasoning,
//! Claude for writing, a local model for simple tasks).
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::agent::Agent;
//!
//! let agent = Agent::new("researcher")
//!     .instructions("You are a research assistant.")
//!     .model("gpt-4o")
//!     .provider(openai_provider.clone());
//!
//! let result = agent.run("Research Rust async patterns", Default::default()).await?;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::stream::Stream;

use serde_json::Value;

use crate::callback::SharedAgentHooks;
use crate::chat::{ResponseFormat, SharedChatProvider};
use crate::error::Result;
use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::tool::{BoxedTool, ToolDefinition, ToolExecutionPolicy};

use super::result::{RunConfig, RunEvent, RunResult, UserInput};

/// Schema specification for structured agent output.
///
/// When set on an [`Agent`], the [`Runner`](super::Runner) will:
///
/// 1. Set `response_format` to [`ResponseFormat::JsonSchema`](crate::chat::ResponseFormat::JsonSchema)
///    on every LLM request, constraining the model to produce valid JSON.
/// 2. Parse the LLM's text output as a JSON [`Value`](serde_json::Value) for the
///    final result in [`RunResult::output`](super::RunResult).
///
/// The caller can then deserialize the output into a concrete Rust type
/// using [`serde_json::from_value::<T>(result.output)`](serde_json::from_value).
///
/// # Examples
///
/// ```rust,ignore
/// use machi::prelude::*;
/// use serde::Deserialize;
/// use serde_json::json;
///
/// #[derive(Deserialize)]
/// struct Country {
///     name: String,
///     capital: String,
///     population: u64,
/// }
///
/// let schema = OutputSchema::new("country", json!({
///     "type": "object",
///     "properties": {
///         "name": { "type": "string" },
///         "capital": { "type": "string" },
///         "population": { "type": "integer" }
///     },
///     "required": ["name", "capital", "population"],
///     "additionalProperties": false
/// }));
///
/// let agent = Agent::new("geo")
///     .instructions("You provide country facts as structured JSON.")
///     .model("gpt-4o")
///     .provider(provider.clone())
///     .output_schema(schema);
///
/// let result = agent.run("Tell me about France", RunConfig::default()).await?;
/// let country: Country = serde_json::from_value(result.output)?;
/// ```
#[derive(Debug, Clone)]
pub struct OutputSchema {
    /// Schema name (used in the `response_format` API parameter).
    name: String,
    /// JSON Schema definition.
    schema: Value,
    /// Whether to enforce strict JSON schema validation (recommended).
    strict: bool,
}

impl OutputSchema {
    /// Creates a new output schema with strict mode enabled (recommended).
    ///
    /// Strict mode constrains the JSON schema features but guarantees that
    /// the model produces valid JSON conforming to the schema.
    #[must_use]
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            schema,
            strict: true,
        }
    }

    /// Creates a new output schema with strict mode explicitly set.
    #[must_use]
    pub fn with_strict(name: impl Into<String>, schema: Value, strict: bool) -> Self {
        Self {
            name: name.into(),
            schema,
            strict,
        }
    }

    /// Returns the schema name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the JSON Schema definition.
    #[must_use]
    pub const fn schema(&self) -> &Value {
        &self.schema
    }

    /// Returns whether strict mode is enabled.
    #[must_use]
    pub const fn is_strict(&self) -> bool {
        self.strict
    }

    /// Converts this into a [`ResponseFormat`] for use in [`ChatRequest`](crate::chat::ChatRequest).
    #[must_use]
    pub fn to_response_format(&self) -> ResponseFormat {
        if self.strict {
            ResponseFormat::json_schema(&self.name, self.schema.clone())
        } else {
            ResponseFormat::JsonSchema {
                json_schema: crate::chat::JsonSchemaSpec {
                    name: self.name.clone(),
                    schema: self.schema.clone(),
                    strict: Some(false),
                },
            }
        }
    }

    /// Creates an output schema by auto-generating JSON Schema from a Rust type.
    ///
    /// This is the most ergonomic way to create an [`OutputSchema`]. The type
    /// must derive [`schemars::JsonSchema`], which auto-generates the JSON
    /// Schema definition at compile time.
    ///
    /// The schema name is derived from the type name automatically.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use machi::prelude::*;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Country {
    ///     name: String,
    ///     capital: String,
    ///     population: u64,
    /// }
    ///
    /// let schema = OutputSchema::from_type::<Country>();
    /// ```
    #[cfg(feature = "schema")]
    #[must_use]
    pub fn from_type<T: schemars::JsonSchema>() -> Self {
        let root = schemars::schema_for!(T);
        let mut schema_value = serde_json::to_value(&root).unwrap_or_default();

        // Remove the $schema meta field — LLM APIs don't need it.
        if let Value::Object(ref mut map) = schema_value {
            map.remove("$schema");
        }

        let name = <T as schemars::JsonSchema>::schema_name();

        Self {
            name: name.into_owned(),
            schema: schema_value,
            strict: true,
        }
    }
}

/// Instructions that guide the agent's behavior.
///
/// Can be either a static string set at construction time, or a dynamic
/// closure that generates instructions based on runtime context.
#[derive(Clone)]
pub enum Instructions {
    /// Static instruction string.
    Static(String),
    /// Dynamic instruction generator.
    ///
    /// Receives the current agent name and returns instructions. Wrapped
    /// in `Arc` for cheap cloning and `Send + Sync` safety.
    Dynamic(Arc<dyn Fn(&str) -> String + Send + Sync>),
}

impl Instructions {
    /// Resolve the instructions to a string for the given agent name.
    #[must_use]
    pub fn resolve(&self, agent_name: &str) -> String {
        match self {
            Self::Static(s) => s.clone(),
            Self::Dynamic(f) => f(agent_name),
        }
    }
}

impl fmt::Debug for Instructions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(s) => f.debug_tuple("Static").field(s).finish(),
            Self::Dynamic(_) => f.debug_tuple("Dynamic").field(&"<closure>").finish(),
        }
    }
}

impl<S: Into<String>> From<S> for Instructions {
    fn from(s: S) -> Self {
        Self::Static(s.into())
    }
}

/// A pure configuration struct defining an AI agent.
///
/// `Agent` contains no execution logic. It describes *what* the agent is and
/// *what* it can do. The [`Runner`](super::Runner) handles *how* it runs.
///
/// # Fields
///
/// - **`name`** — unique identifier used in logging and multi-agent routing
/// - **`instructions`** — system prompt guiding the agent's behavior
/// - **`model`** — LLM model identifier (e.g., `"gpt-4o"`, `"llama3"`)
/// - **`tools`** — capabilities the agent can invoke via function calling
/// - **`managed_agents`** — sub-agents that get wrapped as tools for parallel dispatch
/// - **`hooks`** — optional per-agent lifecycle callbacks
/// - **`max_steps`** — safety limit on reasoning loop iterations
/// - **`provider`** — the LLM provider this agent uses for chat completions
/// - **`description`** — human-readable description (used when this agent is a managed agent)
pub struct Agent {
    /// Unique name identifying this agent.
    pub(crate) name: String,

    /// System-level instructions (prompt) for the agent.
    pub(crate) instructions: Instructions,

    /// LLM model identifier to use for this agent.
    pub(crate) model: String,

    /// The LLM provider this agent uses for chat completions.
    ///
    /// Each agent can have its own provider, enabling heterogeneous
    /// multi-agent systems with different LLMs.
    pub(crate) provider: Option<SharedChatProvider>,

    /// Tools available to this agent for function calling.
    pub(crate) tools: Vec<BoxedTool>,

    /// Sub-agents that can be dispatched as tools.
    ///
    /// Each managed agent is dispatched inline by the Runner,
    /// enabling parallel execution via `futures::future::join_all`.
    pub(crate) managed_agents: Vec<Self>,

    /// Optional per-agent lifecycle hooks.
    pub(crate) hooks: Option<SharedAgentHooks>,

    /// Maximum number of reasoning steps before the runner aborts.
    pub(crate) max_steps: usize,

    /// Human-readable description of what this agent does.
    ///
    /// When this agent is used as a managed agent, the description becomes the
    /// tool description visible to the parent agent's LLM.
    pub(crate) description: String,

    /// Per-tool execution policies (overrides the default `Auto` policy).
    ///
    /// Tools not listed here default to [`ToolExecutionPolicy::Auto`].
    pub(crate) tool_policies: HashMap<String, ToolExecutionPolicy>,

    /// Optional schema for structured JSON output.
    ///
    /// When set, the Runner constrains LLM responses to produce valid JSON
    /// conforming to this schema, and parses the text output as a JSON
    /// [`Value`](serde_json::Value) in [`RunResult::output`](super::RunResult).
    pub(crate) output_schema: Option<OutputSchema>,

    /// Input guardrails that validate user input before or alongside the LLM.
    ///
    /// These checks run during the first step of the agent run. Guardrails
    /// with `run_in_parallel: true` execute concurrently with the first LLM
    /// call; those with `run_in_parallel: false` execute sequentially before it.
    ///
    /// If any guardrail's tripwire is triggered, the run halts immediately
    /// with [`Error::InputGuardrailTriggered`](crate::Error::InputGuardrailTriggered).
    pub(crate) input_guardrails: Vec<InputGuardrail>,

    /// Output guardrails that validate the agent's final response.
    ///
    /// These checks run concurrently after the agent produces a final output.
    /// If any guardrail's tripwire is triggered, the output is discarded and
    /// [`Error::OutputGuardrailTriggered`](crate::Error::OutputGuardrailTriggered) is returned.
    pub(crate) output_guardrails: Vec<OutputGuardrail>,
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("model", &self.model)
            .field("provider", &self.provider.is_some())
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            )
            .field(
                "managed_agents",
                &self
                    .managed_agents
                    .iter()
                    .map(|a| &a.name)
                    .collect::<Vec<_>>(),
            )
            .field("hooks", &self.hooks.is_some())
            .field("max_steps", &self.max_steps)
            .field("description", &self.description)
            .field("tool_policies", &self.tool_policies)
            .field(
                "output_schema",
                &self.output_schema.as_ref().map(OutputSchema::name),
            )
            .field("input_guardrails", &self.input_guardrails)
            .field("output_guardrails", &self.output_guardrails)
            .finish()
    }
}

impl Agent {
    /// Default maximum number of reasoning steps.
    pub const DEFAULT_MAX_STEPS: usize = 10;

    /// Create a new agent with the given name and sensible defaults.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            description: format!("Agent: {name}"),
            name,
            instructions: Instructions::Static(String::new()),
            model: String::new(),
            provider: None,
            tools: Vec::new(),
            managed_agents: Vec::new(),
            hooks: None,
            max_steps: Self::DEFAULT_MAX_STEPS,
            tool_policies: HashMap::new(),
            output_schema: None,
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
        }
    }

    /// Set the system instructions (static string).
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Instructions::Static(instructions.into());
        self
    }

    /// Set dynamic instructions that are resolved at runtime.
    #[must_use]
    pub fn dynamic_instructions<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.instructions = Instructions::Dynamic(Arc::new(f));
        self
    }

    /// Set the LLM model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the LLM provider for this agent.
    #[must_use]
    pub fn provider(mut self, provider: SharedChatProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Add a tool to this agent.
    #[must_use]
    pub fn tool(mut self, tool: BoxedTool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set all tools for this agent.
    #[must_use]
    pub fn tools(mut self, tools: Vec<BoxedTool>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a managed (sub) agent.
    #[must_use]
    pub fn managed_agent(mut self, agent: Self) -> Self {
        self.managed_agents.push(agent);
        self
    }

    /// Set all managed agents.
    #[must_use]
    pub fn managed_agents(mut self, agents: Vec<Self>) -> Self {
        self.managed_agents = agents;
        self
    }

    /// Set per-agent lifecycle hooks.
    #[must_use]
    pub fn hooks(mut self, hooks: SharedAgentHooks) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set the maximum number of reasoning steps.
    #[must_use]
    pub const fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set the agent description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the execution policy for a specific tool.
    #[must_use]
    pub fn tool_policy(mut self, name: impl Into<String>, policy: ToolExecutionPolicy) -> Self {
        self.tool_policies.insert(name.into(), policy);
        self
    }

    /// Set the output schema for structured JSON output.
    ///
    /// When set, the LLM is constrained to produce JSON conforming to this
    /// schema, and the final output is parsed as a JSON [`Value`].
    #[must_use]
    pub fn output_schema(mut self, schema: OutputSchema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Add an input guardrail to this agent.
    ///
    /// Input guardrails validate user input before or alongside the first
    /// LLM call. See [`InputGuardrail`] for details on execution modes.
    #[must_use]
    pub fn input_guardrail(mut self, guardrail: InputGuardrail) -> Self {
        self.input_guardrails.push(guardrail);
        self
    }

    /// Add an output guardrail to this agent.
    ///
    /// Output guardrails validate the agent's final response after generation.
    /// See [`OutputGuardrail`] for details.
    #[must_use]
    pub fn output_guardrail(mut self, guardrail: OutputGuardrail) -> Self {
        self.output_guardrails.push(guardrail);
        self
    }

    /// Set structured output by inferring the JSON Schema from a Rust type.
    ///
    /// This is the most ergonomic way to enable structured output. The type
    /// must derive [`schemars::JsonSchema`] and [`serde::Deserialize`].
    ///
    /// The generated output can be deserialized with [`RunResult::parse`].
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use machi::prelude::*;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Country {
    ///     name: String,
    ///     capital: String,
    ///     population: u64,
    /// }
    ///
    /// let agent = Agent::new("geo")
    ///     .instructions("You provide country facts as structured JSON.")
    ///     .model("gpt-4o")
    ///     .provider(provider.clone())
    ///     .output_type::<Country>();
    ///
    /// let result = agent.run("Tell me about France", RunConfig::default()).await?;
    /// let country: Country = result.parse()?;
    /// ```
    #[cfg(feature = "schema")]
    #[must_use]
    pub fn output_type<T: schemars::JsonSchema>(self) -> Self {
        self.output_schema(OutputSchema::from_type::<T>())
    }

    /// Returns the agent's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the agent's model identifier.
    #[must_use]
    pub fn get_model(&self) -> &str {
        &self.model
    }

    /// Returns the agent's description.
    #[must_use]
    pub fn get_description(&self) -> &str {
        &self.description
    }

    /// Returns the maximum number of reasoning steps.
    #[must_use]
    pub const fn get_max_steps(&self) -> usize {
        self.max_steps
    }

    /// Returns `true` if a provider is configured.
    #[must_use]
    pub fn has_provider(&self) -> bool {
        self.provider.is_some()
    }

    /// Returns the number of tools registered on this agent.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Resolve the system instructions for this agent.
    #[must_use]
    pub fn resolve_instructions(&self) -> String {
        self.instructions.resolve(&self.name)
    }

    /// Returns `true` if this agent has any managed sub-agents.
    #[must_use]
    pub const fn has_managed_agents(&self) -> bool {
        !self.managed_agents.is_empty()
    }

    /// Returns the total number of tools including managed agent tools.
    #[must_use]
    pub fn total_tool_count(&self) -> usize {
        self.tools.len() + self.managed_agents.len()
    }

    /// Run this agent to completion with the given input.
    ///
    /// Accepts any type that implements `Into<UserInput>`, including:
    /// - `&str` or `String` for plain text
    /// - `Vec<ContentPart>` for multimodal content (text + images + audio)
    /// - [`UserInput`] for explicit construction
    ///
    /// This is a convenience wrapper around [`Runner::run`](super::Runner::run)
    /// that uses the agent's own provider.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Agent`] if no provider is configured, or propagates
    /// errors from the underlying [`Runner`](super::Runner) execution.
    pub fn run<'a>(
        &'a self,
        input: impl Into<UserInput>,
        config: RunConfig,
    ) -> Pin<Box<dyn Future<Output = Result<RunResult>> + Send + 'a>> {
        super::Runner::run(self, input, config)
    }

    /// Execute a streaming agent run, returning a stream of [`RunEvent`]s.
    ///
    /// This is a convenience wrapper around [`Runner::run_streamed`](super::Runner::run_streamed)
    /// that uses the agent's own provider.
    ///
    /// # Errors
    ///
    /// Errors are delivered as `Err(...)` items in the returned stream.
    pub fn run_streamed<'a>(
        &'a self,
        input: impl Into<UserInput>,
        config: RunConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<RunEvent>> + Send + 'a>> {
        super::Runner::run_streamed(self, input, config)
    }

    /// Build a [`ToolDefinition`] for this agent when used as a managed sub-agent.
    ///
    /// The definition exposes a single `task` string parameter, which the parent
    /// agent's LLM fills in to describe the work to delegate.
    #[must_use]
    pub fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            &self.name,
            &self.description,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to delegate to this agent."
                    }
                },
                "required": ["task"],
                "additionalProperties": false
            }),
        )
    }
}
