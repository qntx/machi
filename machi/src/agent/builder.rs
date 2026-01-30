//! Agent builder module providing a fluent API for constructing agents.
//!
//! This module uses the typestate pattern to provide compile-time guarantees
//! about builder configuration while maintaining a clean API.

use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use tokio::sync::RwLock;

use crate::{
    completion::{CompletionModel, Document, message::ToolChoice},
    store::VectorStoreIndexDyn,
    tool::{
        Tool, ToolDyn, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
};

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
use crate::tool::mcp::McpTool as RmcpTool;

use super::Agent;

/// Marker type indicating no tools have been configured.
pub struct NoTools;

/// Marker type indicating tools have been configured via ToolSet.
pub struct WithTools;

/// A fluent builder for creating [`Agent`] instances.
///
/// The builder uses typestate pattern via the `T` parameter to track
/// whether tools have been configured, enabling different methods
/// based on the current state.
///
/// # Example
/// ```rust,ignore
/// use machi::{providers::openai, agent::AgentBuilder};
///
/// let openai = openai::Client::from_env();
/// let model = openai.completion_model("gpt-4o");
///
/// let agent = AgentBuilder::new(model)
///     .preamble("You are a helpful assistant.")
///     .context("Important context document")
///     .tool(my_tool)
///     .temperature(0.8)
///     .build();
/// ```
pub struct AgentBuilder<M, T = NoTools>
where
    M: CompletionModel,
{
    /// Name of the agent for logging and debugging.
    name: Option<String>,
    /// Agent description for workflows and documentation.
    description: Option<String>,
    /// The completion model to use.
    model: M,
    /// System prompt.
    preamble: Option<String>,
    /// Static context documents.
    static_context: Vec<Document>,
    /// Additional model parameters.
    additional_params: Option<serde_json::Value>,
    /// Maximum tokens for completion.
    max_tokens: Option<u64>,
    /// Dynamic context stores with sample counts.
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
    /// Model temperature.
    temperature: Option<f64>,
    /// Tool server handle (for NoTools state).
    tool_server_handle: Option<ToolServerHandle>,
    /// Tool choice configuration.
    tool_choice: Option<ToolChoice>,
    /// Default max depth for multi-turn.
    default_max_depth: Option<usize>,
    /// Static tool names (for WithTools state).
    static_tools: Vec<String>,
    /// Dynamic tools stores (for WithTools state).
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
    /// Tool implementations (for WithTools state).
    tools: ToolSet,
    /// Marker for typestate.
    _marker: PhantomData<T>,
}

impl<M> AgentBuilder<M, NoTools>
where
    M: CompletionModel,
{
    /// Creates a new builder with the given completion model.
    pub fn new(model: M) -> Self {
        Self {
            name: None,
            description: None,
            model,
            preamble: None,
            static_context: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            dynamic_context: vec![],
            tool_server_handle: None,
            tool_choice: None,
            default_max_depth: None,
            static_tools: vec![],
            dynamic_tools: vec![],
            tools: ToolSet::default(),
            _marker: PhantomData,
        }
    }

    /// Sets an existing tool server handle.
    pub fn tool_server_handle(mut self, handle: ToolServerHandle) -> Self {
        self.tool_server_handle = Some(handle);
        self
    }

    /// Adds a tool to the agent, transitioning to WithTools state.
    pub fn tool(self, tool: impl Tool + 'static) -> AgentBuilder<M, WithTools> {
        let toolname = tool.name();
        let tools = ToolSet::from_tools(vec![tool]);

        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_server_handle: None,
            tool_choice: self.tool_choice,
            default_max_depth: self.default_max_depth,
            static_tools: vec![toolname],
            dynamic_tools: vec![],
            tools,
            _marker: PhantomData,
        }
    }

    /// Adds multiple boxed tools, transitioning to WithTools state.
    pub fn tools(self, tools: Vec<Box<dyn ToolDyn>>) -> AgentBuilder<M, WithTools> {
        let static_tools = tools.iter().map(|t| t.name()).collect();
        let tools = ToolSet::from_tools_boxed(tools);

        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_server_handle: None,
            tool_choice: self.tool_choice,
            default_max_depth: self.default_max_depth,
            static_tools,
            dynamic_tools: vec![],
            tools,
            _marker: PhantomData,
        }
    }

    /// Adds an MCP tool from rmcp, transitioning to WithTools state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tool(
        self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilder<M, WithTools> {
        let toolname = tool.name.clone().to_string();
        let tools = ToolSet::from_tools(vec![RmcpTool::from_mcp_server(tool, client)]);

        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_server_handle: None,
            tool_choice: self.tool_choice,
            default_max_depth: self.default_max_depth,
            static_tools: vec![toolname],
            dynamic_tools: vec![],
            tools,
            _marker: PhantomData,
        }
    }

    /// Adds multiple MCP tools from rmcp, transitioning to WithTools state.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools(
        self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
    ) -> AgentBuilder<M, WithTools> {
        let (static_tools, tool_vec) =
            tools
                .into_iter()
                .fold((Vec::new(), Vec::new()), |(mut names, mut set), tool| {
                    let name = tool.name.to_string();
                    let mcp_tool = RmcpTool::from_mcp_server(tool, client.clone());
                    names.push(name);
                    set.push(mcp_tool);
                    (names, set)
                });

        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_server_handle: None,
            tool_choice: self.tool_choice,
            default_max_depth: self.default_max_depth,
            static_tools,
            dynamic_tools: vec![],
            tools: ToolSet::from_tools(tool_vec),
            _marker: PhantomData,
        }
    }

    /// Adds dynamic tools, transitioning to WithTools state.
    pub fn dynamic_tools(
        self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet,
    ) -> AgentBuilder<M, WithTools> {
        AgentBuilder {
            name: self.name,
            description: self.description,
            model: self.model,
            preamble: self.preamble,
            static_context: self.static_context,
            additional_params: self.additional_params,
            max_tokens: self.max_tokens,
            dynamic_context: self.dynamic_context,
            temperature: self.temperature,
            tool_server_handle: None,
            tool_choice: self.tool_choice,
            default_max_depth: self.default_max_depth,
            static_tools: vec![],
            dynamic_tools: vec![(sample, Box::new(dynamic_tools))],
            tools: toolset,
            _marker: PhantomData,
        }
    }

    /// Builds the agent without tools.
    pub fn build(self) -> Agent<M> {
        let tool_server_handle = self
            .tool_server_handle
            .unwrap_or_else(|| ToolServer::new().run());

        Agent {
            name: self.name,
            description: self.description,
            model: Arc::new(self.model),
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_choice: self.tool_choice,
            dynamic_context: Arc::new(RwLock::new(self.dynamic_context)),
            tool_server_handle,
            default_max_depth: self.default_max_depth,
        }
    }
}

impl<M> AgentBuilder<M, WithTools>
where
    M: CompletionModel,
{
    /// Adds another tool to the agent.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.tools.add_tool(tool);
        self.static_tools.push(toolname);
        self
    }

    /// Adds multiple boxed tools.
    pub fn tools(mut self, tools: Vec<Box<dyn ToolDyn>>) -> Self {
        let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
        let toolset = ToolSet::from_tools_boxed(tools);
        self.tools.add_tools(toolset);
        self.static_tools.extend(names);
        self
    }

    /// Adds multiple MCP tools from rmcp.
    #[cfg(feature = "rmcp")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub fn rmcp_tools(
        mut self,
        tools: Vec<rmcp::model::Tool>,
        client: rmcp::service::ServerSink,
    ) -> Self {
        for tool in tools {
            let name = tool.name.to_string();
            let mcp_tool = RmcpTool::from_mcp_server(tool, client.clone());
            self.static_tools.push(name);
            self.tools.add_tool(mcp_tool);
        }
        self
    }

    /// Adds dynamic tools.
    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.dynamic_tools.push((sample, Box::new(dynamic_tools)));
        self.tools.add_tools(toolset);
        self
    }

    /// Builds the agent with configured tools.
    pub fn build(self) -> Agent<M> {
        let tool_server_handle = ToolServer::new()
            .static_tool_names(self.static_tools)
            .add_tools(self.tools)
            .add_dynamic_tools(self.dynamic_tools)
            .run();

        Agent {
            name: self.name,
            description: self.description,
            model: Arc::new(self.model),
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_choice: self.tool_choice,
            dynamic_context: Arc::new(RwLock::new(self.dynamic_context)),
            tool_server_handle,
            default_max_depth: self.default_max_depth,
        }
    }
}

/// Common methods available in all builder states.
impl<M, T> AgentBuilder<M, T>
where
    M: CompletionModel,
{
    /// Sets the agent name.
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the agent description.
    pub fn description(mut self, description: &str) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the system prompt.
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Removes the system prompt.
    pub fn without_preamble(mut self) -> Self {
        self.preamble = None;
        self
    }

    /// Appends to the existing preamble.
    pub fn append_preamble(mut self, doc: &str) -> Self {
        self.preamble = Some(format!("{}\n{}", self.preamble.unwrap_or_default(), doc));
        self
    }

    /// Adds a static context document.
    pub fn context(mut self, doc: &str) -> Self {
        self.static_context.push(Document {
            id: format!("static_doc_{}", self.static_context.len()),
            text: doc.into(),
            additional_props: HashMap::new(),
        });
        self
    }

    /// Adds dynamic context with a sample count.
    pub fn dynamic_context(
        mut self,
        sample: usize,
        dynamic_context: impl VectorStoreIndexDyn + Send + Sync + 'static,
    ) -> Self {
        self.dynamic_context
            .push((sample, Box::new(dynamic_context)));
        self
    }

    /// Sets the tool choice configuration.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets the default max depth for multi-turn conversations.
    pub fn default_max_depth(mut self, default_max_depth: usize) -> Self {
        self.default_max_depth = Some(default_max_depth);
        self
    }

    /// Sets the model temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum tokens for completion.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets additional model parameters.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

/// Type alias for backwards compatibility.
pub type AgentBuilderSimple<M> = AgentBuilder<M, WithTools>;
