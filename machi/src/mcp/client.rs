//! MCP client for connecting to local and remote servers.

use std::sync::Arc;

use rmcp::{
    ServiceExt,
    model::{ClientCapabilities, Implementation, InitializeRequestParams, Tool},
    service::ServerSink,
    transport::{StreamableHttpClientTransport, child_process::TokioChildProcess},
};
use tokio::task::JoinHandle;

use super::error::McpError;
use super::transport::TransportConfig;

/// Configuration for MCP client.
#[derive(Debug, Clone)]
pub struct McpClientConfig {
    /// Client name for identification.
    pub name: String,
    /// Client version.
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

/// A high-level MCP client supporting both local and remote servers.
///
/// This provides a unified API for connecting to MCP servers regardless
/// of the underlying transport (HTTP or stdio).
///
/// # Examples
///
/// ## HTTP (Remote Server)
///
/// ```rust,ignore
/// let client = McpClient::http("http://localhost:8080").await?;
/// ```
///
/// ## Stdio (Local Process)
///
/// ```rust,ignore
/// let client = McpClient::stdio("python", &["server.py"]).await?;
/// ```
pub struct McpClient {
    sink: ServerSink,
    tools: Vec<Tool>,
    /// Background task keeping the service alive.
    /// Wrapped in Arc to allow Clone.
    #[allow(dead_code)]
    _service_handle: Arc<JoinHandle<()>>,
}

impl McpClient {
    /// Connects to an HTTP MCP server.
    ///
    /// # Arguments
    ///
    /// * `url` - The HTTP URL of the server (e.g., "http://localhost:8080")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = McpClient::http("http://localhost:8080").await?;
    /// println!("Tools: {:?}", client.tool_names());
    /// ```
    pub async fn http(url: impl Into<String>) -> Result<Self, McpError> {
        Self::connect(TransportConfig::http(url)).await
    }

    /// Spawns and connects to a local MCP server process.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute (e.g., "python", "node")
    /// * `args` - Command arguments (e.g., &["server.py"])
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = McpClient::stdio("python", &["my_mcp_server.py"]).await?;
    /// println!("Tools: {:?}", client.tool_names());
    /// ```
    pub async fn stdio(command: impl Into<String>, args: &[&str]) -> Result<Self, McpError> {
        Self::connect(TransportConfig::stdio(command, args)).await
    }

    /// Connects using a transport configuration.
    pub async fn connect(config: TransportConfig) -> Result<Self, McpError> {
        Self::connect_with_client_config(config, McpClientConfig::default()).await
    }

    /// Connects with custom client configuration.
    pub async fn connect_with_client_config(
        transport_config: TransportConfig,
        client_config: McpClientConfig,
    ) -> Result<Self, McpError> {
        let client_info = InitializeRequestParams {
            meta: None,
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: client_config.name,
                version: client_config.version,
                ..Default::default()
            },
        };

        match transport_config {
            TransportConfig::Http { url } => {
                let transport = StreamableHttpClientTransport::from_uri(url.as_str());

                let service = client_info.serve(transport).await.map_err(|e| {
                    McpError::HttpConnectionFailed {
                        url: url.clone(),
                        message: e.to_string(),
                    }
                })?;

                let sink = service.peer().clone();
                let tools = service
                    .peer()
                    .list_tools(Default::default())
                    .await
                    .map_err(|e| McpError::ListToolsFailed(e.to_string()))?
                    .tools;

                // Spawn the service to keep it running in the background
                let handle = tokio::spawn(async move {
                    let _ = service.waiting().await;
                });

                Ok(Self {
                    sink,
                    tools,
                    _service_handle: Arc::new(handle),
                })
            }

            TransportConfig::Stdio {
                command,
                args,
                cwd,
                env,
            } => {
                let mut cmd = tokio::process::Command::new(&command);
                cmd.args(&args);

                if let Some(dir) = cwd {
                    cmd.current_dir(dir);
                }

                if let Some(env_vars) = env {
                    for (key, value) in env_vars {
                        cmd.env(key, value);
                    }
                }

                let transport =
                    TokioChildProcess::new(cmd).map_err(|e| McpError::ProcessSpawnFailed {
                        command: command.clone(),
                        message: e.to_string(),
                    })?;

                let service = client_info.serve(transport).await.map_err(|e| {
                    McpError::ProcessSpawnFailed {
                        command: command.clone(),
                        message: e.to_string(),
                    }
                })?;

                let sink = service.peer().clone();
                let tools = service
                    .peer()
                    .list_tools(Default::default())
                    .await
                    .map_err(|e| McpError::ListToolsFailed(e.to_string()))?
                    .tools;

                // Spawn the service to keep it running in the background
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
    }

    /// Returns a reference to the cached tools.
    #[must_use]
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Returns the tool names.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_ref()).collect()
    }

    /// Returns the server sink for tool execution.
    #[must_use]
    pub fn sink(&self) -> &ServerSink {
        &self.sink
    }

    /// Consumes the client and returns the tools and sink.
    #[must_use]
    pub fn into_parts(self) -> (Vec<Tool>, ServerSink) {
        (self.tools, self.sink)
    }
}

/// Builder for connecting to multiple MCP servers.
///
/// Supports both HTTP and stdio transports with a declarative API
/// similar to Python agent frameworks.
///
/// # Example - Multiple Servers
///
/// ```rust,ignore
/// use machi::mcp::McpClientBuilder;
///
/// // Connect to multiple MCP servers and use with agent
/// let mcp = McpClientBuilder::new()
///     .http("calculator", "http://localhost:8080")
///     .stdio("local_tools", "python", &["tools.py"])
///     .connect()
///     .await?;
///
/// let agent = client
///     .agent(model)
///     .mcp(mcp)
///     .build();
/// ```
///
/// # Example - Get Individual Clients
///
/// ```rust,ignore
/// // Get named clients for more control
/// let clients = McpClientBuilder::new()
///     .http("math", "http://localhost:8080")
///     .http("weather", "http://localhost:8081")
///     .connect_all()
///     .await?;
///
/// for (name, client) in &clients {
///     println!("Server {}: {:?}", name, client.tool_names());
/// }
/// ```
#[derive(Default)]
pub struct McpClientBuilder {
    configs: Vec<(String, TransportConfig)>,
    client_config: McpClientConfig,
}

impl McpClientBuilder {
    /// Creates a new empty server collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an HTTP server.
    #[must_use]
    pub fn http(mut self, name: impl Into<String>, url: impl Into<String>) -> Self {
        self.configs.push((name.into(), TransportConfig::http(url)));
        self
    }

    /// Adds a stdio (local process) server.
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

    /// Sets custom client configuration.
    #[must_use]
    pub fn client_config(mut self, config: McpClientConfig) -> Self {
        self.client_config = config;
        self
    }

    /// Connects to all configured servers.
    pub async fn connect_all(self) -> Result<Vec<(String, McpClient)>, McpError> {
        let mut clients = Vec::with_capacity(self.configs.len());

        for (name, config) in self.configs {
            let client =
                McpClient::connect_with_client_config(config, self.client_config.clone()).await?;
            clients.push((name, client));
        }

        Ok(clients)
    }

    /// Connects to all servers and returns a merged client.
    ///
    /// This is the primary method for connecting to multiple MCP servers
    /// and using them with an agent. Each tool retains its own server connection.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mcp = McpClientBuilder::new()
    ///     .http("math", "http://localhost:8080")
    ///     .stdio("local", "python", &["server.py"])
    ///     .connect()
    ///     .await?;
    ///
    /// let agent = client.agent(model).mcp(mcp).build();
    /// ```
    pub async fn connect(self) -> Result<MergedMcpClients, McpError> {
        let clients = self.connect_all().await?;
        Ok(MergedMcpClients::from_clients(clients))
    }
}

/// Merged MCP clients from multiple servers.
///
/// Created by [`McpClientBuilder::connect`]. Can be used directly with
/// [`AgentBuilder::mcp`](crate::agent::AgentBuilder::mcp).
pub struct MergedMcpClients {
    tools: Vec<crate::mcp::McpTool>,
    /// Keep handles alive to maintain connections.
    #[allow(dead_code)]
    handles: Vec<Arc<JoinHandle<()>>>,
}

/// Trait for types that can be converted to MCP tools.
///
/// This allows both [`McpClient`] and [`MergedMcpClients`] to be used
/// with [`AgentBuilder::mcp`](crate::agent::AgentBuilder::mcp).
pub trait IntoMcpTools {
    /// Converts into a vector of MCP tools.
    fn into_mcp_tools(self) -> Vec<crate::mcp::McpTool>;
}

impl IntoMcpTools for McpClient {
    fn into_mcp_tools(self) -> Vec<crate::mcp::McpTool> {
        let (tools, sink) = self.into_parts();
        tools
            .into_iter()
            .map(|t| crate::mcp::McpTool::new(t, sink.clone()))
            .collect()
    }
}

impl IntoMcpTools for MergedMcpClients {
    fn into_mcp_tools(self) -> Vec<crate::mcp::McpTool> {
        self.into_tools()
    }
}

impl MergedMcpClients {
    /// Creates merged clients from a list of named clients.
    fn from_clients(clients: Vec<(String, McpClient)>) -> Self {
        let mut tools = Vec::new();
        let mut handles = Vec::new();

        for (_, client) in clients {
            // Extract handle before consuming client
            let handle = client._service_handle.clone();
            let (client_tools, sink) = client.into_parts();
            handles.push(handle);

            for tool in client_tools {
                tools.push(crate::mcp::McpTool::new(tool, sink.clone()));
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
    pub fn into_tools(self) -> Vec<crate::mcp::McpTool> {
        self.tools
    }
}
