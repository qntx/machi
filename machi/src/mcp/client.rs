use std::sync::Arc;

use rmcp::{
    ServiceExt,
    model::{ClientCapabilities, Implementation, InitializeRequestParams, Tool},
    service::ServerSink,
    transport::{StreamableHttpClientTransport, child_process::TokioChildProcess},
};
use tokio::task::JoinHandle;

use super::McpTool;
use super::error::McpError;
use super::transport::TransportConfig;

/// Trait for types that can provide MCP tools to an agent.
///
/// Implemented by both [`McpClient`] (single server) and [`MergedMcpClients`]
/// (multiple servers), enabling a unified API via [`AgentBuilder::mcp`].
///
/// [`AgentBuilder::mcp`]: crate::agent::AgentBuilder::mcp
pub trait IntoMcpTools {
    /// Converts into a vector of MCP tools for agent use.
    fn into_mcp_tools(self) -> Vec<McpTool>;
}

/// Configuration for MCP client identification.
#[derive(Debug, Clone)]
pub struct McpClientConfig {
    /// Client name sent to server during handshake.
    pub name: String,
    /// Client version sent to server during handshake.
    pub version: String,
}

impl Default for McpClientConfig {
    fn default() -> Self {
        Self {
            name: "machi".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// A high-level MCP client for connecting to a single server.
///
/// Supports both HTTP (remote) and stdio (local process) transports.
/// Use [`McpClientBuilder`] for connecting to multiple servers.
///
/// # Examples
///
/// ```rust,ignore
/// // HTTP server
/// let client = McpClient::http("http://localhost:8080").await?;
/// println!("Tools: {:?}", client.tool_names());
///
/// // Local process
/// let client = McpClient::stdio("python", &["server.py"]).await?;
/// ```
pub struct McpClient {
    sink: ServerSink,
    tools: Vec<Tool>,
    #[allow(dead_code)]
    _service_handle: Arc<JoinHandle<()>>,
}

impl McpClient {
    /// Connects to an HTTP MCP server.
    pub async fn http(url: impl Into<String>) -> Result<Self, McpError> {
        Self::connect(TransportConfig::http(url)).await
    }

    /// Spawns and connects to a local MCP server process.
    pub async fn stdio(command: impl Into<String>, args: &[&str]) -> Result<Self, McpError> {
        Self::connect(TransportConfig::stdio(command, args)).await
    }

    /// Connects using a transport configuration.
    pub async fn connect(config: TransportConfig) -> Result<Self, McpError> {
        Self::connect_with_config(config, McpClientConfig::default()).await
    }

    /// Connects with custom client configuration.
    pub async fn connect_with_config(
        transport: TransportConfig,
        config: McpClientConfig,
    ) -> Result<Self, McpError> {
        let init_params = InitializeRequestParams {
            meta: None,
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: config.name,
                version: config.version,
                ..Default::default()
            },
        };

        match transport {
            TransportConfig::Http { url } => Self::connect_http(&url, init_params).await,
            TransportConfig::Stdio {
                command,
                args,
                cwd,
                env,
            } => Self::connect_stdio(&command, &args, cwd, env, init_params).await,
        }
    }

    /// Returns the cached tools from the server.
    #[must_use]
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Returns the tool names.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_ref()).collect()
    }

    /// Returns the server sink for direct tool execution.
    #[must_use]
    pub fn sink(&self) -> &ServerSink {
        &self.sink
    }

    /// Consumes and returns the raw tools and sink.
    #[must_use]
    pub fn into_parts(self) -> (Vec<Tool>, ServerSink) {
        (self.tools, self.sink)
    }
}

// Private connection helpers
impl McpClient {
    async fn connect_http(url: &str, init: InitializeRequestParams) -> Result<Self, McpError> {
        let transport = StreamableHttpClientTransport::from_uri(url);

        let service = init
            .serve(transport)
            .await
            .map_err(|e| McpError::HttpConnectionFailed {
                url: url.to_string(),
                message: e.to_string(),
            })?;

        let sink = service.peer().clone();
        let tools = service
            .peer()
            .list_tools(Default::default())
            .await
            .map_err(|e| McpError::ListToolsFailed(e.to_string()))?
            .tools;

        let handle = tokio::spawn(async move {
            let _ = service.waiting().await;
        });

        Ok(Self {
            sink,
            tools,
            _service_handle: Arc::new(handle),
        })
    }

    async fn connect_stdio(
        command: &str,
        args: &[String],
        cwd: Option<String>,
        env: Option<Vec<(String, String)>>,
        init: InitializeRequestParams,
    ) -> Result<Self, McpError> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);

        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }

        if let Some(env_vars) = env {
            for (key, value) in env_vars {
                cmd.env(key, value);
            }
        }

        let transport = TokioChildProcess::new(cmd).map_err(|e| McpError::ProcessSpawnFailed {
            command: command.to_string(),
            message: e.to_string(),
        })?;

        let service = init
            .serve(transport)
            .await
            .map_err(|e| McpError::ProcessSpawnFailed {
                command: command.to_string(),
                message: e.to_string(),
            })?;

        let sink = service.peer().clone();
        let tools = service
            .peer()
            .list_tools(Default::default())
            .await
            .map_err(|e| McpError::ListToolsFailed(e.to_string()))?
            .tools;

        let handle = tokio::spawn(async move {
            let _ = service.waiting().await;
        });

        Ok(Self {
            sink,
            tools,
            _service_handle: Arc::new(handle),
        })
    }
}

impl IntoMcpTools for McpClient {
    fn into_mcp_tools(self) -> Vec<McpTool> {
        let (tools, sink) = self.into_parts();
        tools
            .into_iter()
            .map(|t| McpTool::new(t, sink.clone()))
            .collect()
    }
}

/// Merged MCP clients from multiple servers.
///
/// Created by [`McpClientBuilder::connect`]. Combines tools from all
/// connected servers while maintaining their individual connections.
pub struct MergedMcpClients {
    tools: Vec<McpTool>,
    #[allow(dead_code)]
    handles: Vec<Arc<JoinHandle<()>>>,
}

impl MergedMcpClients {
    fn from_clients(clients: Vec<(String, McpClient)>) -> Self {
        let mut tools = Vec::new();
        let mut handles = Vec::new();

        for (_, client) in clients {
            let handle = client._service_handle.clone();
            let (client_tools, sink) = client.into_parts();
            handles.push(handle);

            for tool in client_tools {
                tools.push(McpTool::new(tool, sink.clone()));
            }
        }

        Self { tools, handles }
    }

    /// Returns the tool names from all connected servers.
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        use crate::tool::ToolDyn;
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Consumes and returns all tools.
    #[must_use]
    pub fn into_tools(self) -> Vec<McpTool> {
        self.tools
    }
}

impl IntoMcpTools for MergedMcpClients {
    fn into_mcp_tools(self) -> Vec<McpTool> {
        self.into_tools()
    }
}

/// Builder for connecting to multiple MCP servers.
///
/// Provides a declarative API for configuring connections to multiple
/// servers, then merging them into a single toolset for agent use.
///
/// # Example
///
/// ```rust,ignore
/// use machi::mcp::McpClientBuilder;
///
/// let mcp = McpClientBuilder::new()
///     .http("math", "http://localhost:8080")
///     .stdio("local", "python", &["tools.py"])
///     .connect()
///     .await?;
///
/// let agent = client.agent(model).mcp(mcp).build();
/// ```
#[derive(Default)]
pub struct McpClientBuilder {
    configs: Vec<(String, TransportConfig)>,
    client_config: McpClientConfig,
}

impl McpClientBuilder {
    /// Creates a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an HTTP server connection.
    #[must_use]
    pub fn http(mut self, name: impl Into<String>, url: impl Into<String>) -> Self {
        self.configs.push((name.into(), TransportConfig::http(url)));
        self
    }

    /// Adds a stdio (local process) server connection.
    #[must_use]
    pub fn stdio(
        mut self,
        name: impl Into<String>,
        command: impl Into<String>,
        args: &[&str],
    ) -> Self {
        self.configs
            .push((name.into(), TransportConfig::stdio(command, args)));
        self
    }

    /// Sets custom client configuration for all connections.
    #[must_use]
    pub fn config(mut self, config: McpClientConfig) -> Self {
        self.client_config = config;
        self
    }

    /// Connects to all servers and returns individual clients.
    ///
    /// Use this when you need fine-grained control over each connection.
    pub async fn connect_all(self) -> Result<Vec<(String, McpClient)>, McpError> {
        let mut clients = Vec::with_capacity(self.configs.len());

        for (name, config) in self.configs {
            let client = McpClient::connect_with_config(config, self.client_config.clone()).await?;
            clients.push((name, client));
        }

        Ok(clients)
    }

    /// Connects to all servers and returns a merged client.
    ///
    /// This is the primary method for multi-server use with agents.
    /// Each tool retains its own server connection.
    pub async fn connect(self) -> Result<MergedMcpClients, McpError> {
        let clients = self.connect_all().await?;
        Ok(MergedMcpClients::from_clients(clients))
    }
}
