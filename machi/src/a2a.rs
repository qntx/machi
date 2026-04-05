//! A2A (Agent-to-Agent) protocol integration for machi agents.
//!
//! This module bridges [A2A protocol](https://a2a-protocol.org/) remote agents
//! with the machi tool system, allowing machi agents to communicate with any
//! A2A-compatible remote agent as if it were a tool or sub-agent.
//!
//! # Architecture
//!
//! ```text
//! A2aAgent (wraps ra2a::A2AClient)
//!   ├── new(url)        → A2aAgentBuilder → connect()
//!   ├── agent_card()    → cached AgentCard metadata
//!   ├── send(text)      → invoke the remote agent
//!   └── into_tool()     → BoxedTool for Agent integration
//!
//! A2aTool (bridges A2aAgent → machi DynTool)
//! ```
//!
//! # Examples
//!
//! ```rust
//! use machi::a2a::A2aAgent;
//!
//! // Simple builder
//! let builder = A2aAgent::new("https://remote-agent.example.com");
//!
//! // With auth and custom name
//! let builder = A2aAgent::new("https://remote-agent.example.com")
//!     .bearer_auth("sk-xxx")
//!     .name("currency-agent");
//! ```

use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use ra2a::client::{Client, JsonRpcTransport, TransportConfig};
use ra2a::types::{AgentCard, Message as A2aMessage, SendMessageRequest, StreamResponse};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::error::ToolError;
use crate::tool::{BoxedTool, DynTool, ToolDefinition};

/// Builder for connecting to a remote A2A agent.
///
/// Created by [`A2aAgent::new`]. Use method chaining to configure
/// authentication, then call [`connect`](Self::connect).
///
/// # Examples
///
/// ```rust
/// use machi::a2a::A2aAgent;
///
/// let builder = A2aAgent::new("https://agent.example.com")
///     .bearer_auth("sk-xxx")
///     .header("X-Custom", "value")
///     .timeout(60)
///     .name("my-agent");
/// ```
#[derive(Debug)]
pub struct A2aAgentBuilder {
    config: TransportConfig,
    name: Option<String>,
}

impl A2aAgentBuilder {
    /// Set a Bearer authentication token.
    #[must_use]
    pub fn bearer_auth(mut self, token: impl Into<String>) -> Self {
        let value = format!("Bearer {}", token.into());
        if let Ok(v) = value.parse() {
            self.config
                .headers
                .insert(reqwest::header::AUTHORIZATION, v);
        }
        self
    }

    /// Add a custom HTTP header.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        let name_str = name.into();
        if let (Ok(k), Ok(v)) = (
            name_str.parse::<reqwest::header::HeaderName>(),
            value.into().parse::<reqwest::header::HeaderValue>(),
        ) {
            self.config.headers.insert(k, v);
        }
        self
    }

    /// Set an API key header.
    #[must_use]
    pub fn api_key(mut self, header_name: impl Into<String>, key: impl Into<String>) -> Self {
        let name_str = header_name.into();
        if let (Ok(k), Ok(v)) = (
            name_str.parse::<reqwest::header::HeaderName>(),
            key.into().parse::<reqwest::header::HeaderValue>(),
        ) {
            self.config.headers.insert(k, v);
        }
        self
    }

    /// Set the request timeout in seconds.
    #[must_use]
    pub const fn timeout(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self
    }

    /// Override the default connection name (defaults to the agent card name).
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Connect to the remote A2A agent and fetch its agent card.
    ///
    /// # Errors
    ///
    /// Returns an error if the A2A client fails to build or the agent card cannot be fetched.
    pub async fn connect(self) -> crate::Result<A2aAgent> {
        let url = self.config.base_url.clone();
        info!(url = %url, "Connecting to A2A agent");

        let transport = JsonRpcTransport::new(self.config).map_err(|e| {
            crate::error::AgentError::runtime(format!(
                "Failed to build A2A transport for '{url}': {e}",
            ))
        })?;

        let client = Client::new(Box::new(transport));

        // Fetch agent card to discover capabilities.
        let card = client.get_agent_card().await.map_err(|e| {
            crate::error::AgentError::runtime(format!(
                "Failed to fetch agent card from '{url}': {e}",
            ))
        })?;

        let name = self.name.unwrap_or_else(|| card.name.clone());
        info!(
            name = %name,
            skills = card.skills.len(),
            "A2A agent connected",
        );

        Ok(A2aAgent {
            client: Arc::new(client),
            card: Arc::new(RwLock::new(card)),
            name,
        })
    }
}

/// A connection to a remote A2A agent.
///
/// `A2aAgent` wraps an [`Client`](ra2a::client::Client) and provides
/// methods to send messages and convert the remote agent into a machi
/// [`BoxedTool`] for use with [`Agent`](crate::agent::Agent).
///
/// # Connection
///
/// Use [`A2aAgent::new`] to create a builder, configure it, then `.connect().await`.
///
/// # Examples
///
/// ```rust
/// use machi::a2a::A2aAgent;
///
/// let builder = A2aAgent::new("https://agent.example.com")
///     .bearer_auth("token")
///     .name("weather");
/// ```
pub struct A2aAgent {
    /// The underlying ra2a client.
    client: Arc<Client>,
    /// Cached agent card.
    card: Arc<RwLock<AgentCard>>,
    /// Human-readable name for this agent.
    name: String,
}

impl std::fmt::Debug for A2aAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2aAgent")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl A2aAgent {
    /// Create a builder for connecting to a remote A2A agent.
    ///
    /// # Arguments
    ///
    /// * `url` �?the base URL of the A2A agent
    ///
    /// # Examples
    ///
    /// ```rust
    /// use machi::a2a::A2aAgent;
    ///
    /// let builder = A2aAgent::new("https://agent.example.com")
    ///     .bearer_auth("sk-xxx");
    /// ```
    #[allow(clippy::new_ret_no_self)]
    pub fn new(url: impl Into<String>) -> A2aAgentBuilder {
        A2aAgentBuilder {
            config: TransportConfig::new(url),
            name: None,
        }
    }

    /// Get the name of this agent.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a clone of the cached agent card.
    pub async fn agent_card(&self) -> AgentCard {
        self.card.read().await.clone()
    }

    /// Refresh the agent card from the remote server.
    ///
    /// # Errors
    ///
    /// Returns an error if the remote server is unreachable or returns an invalid response.
    pub async fn refresh_card(&self) -> crate::Result<AgentCard> {
        let card = self.client.get_agent_card().await.map_err(|e| {
            crate::error::AgentError::runtime(format!(
                "Failed to refresh agent card for '{}': {e}",
                self.name,
            ))
        })?;
        let mut cached = self.card.write().await;
        *cached = card.clone();
        Ok(card)
    }

    /// Send a text message to the remote agent and collect the response.
    ///
    /// This consumes all streaming events and returns the final text output.
    ///
    /// # Errors
    ///
    /// Returns an error if the message fails to send or the response stream encounters an error.
    pub async fn send(&self, text: impl Into<String>) -> Result<String, ToolError> {
        let message = A2aMessage::user_text(text);
        self.send_message(message).await
    }

    /// Send a full [`Message`](ra2a::types::Message) and collect the response.
    ///
    /// # Errors
    ///
    /// Returns an error if the message fails to send or the response stream encounters an error.
    pub async fn send_message(&self, message: A2aMessage) -> Result<String, ToolError> {
        let request = SendMessageRequest::new(message);
        let mut stream = self
            .client
            .send_streaming_message(&request)
            .await
            .map_err(|e| {
                ToolError::Execution(format!("A2A agent '{}' send failed: {e}", self.name))
            })?;

        let mut output = String::new();
        while let Some(result) = stream.next().await {
            let event = result.map_err(|e| {
                ToolError::Execution(format!("A2A agent '{}' stream error: {e}", self.name))
            })?;
            Self::collect_event_text(&event, &mut output);
        }

        debug!(agent = %self.name, len = output.len(), "A2A response received");
        Ok(output)
    }

    /// Extract text content from a [`StreamResponse`] and append it to the output buffer.
    fn collect_event_text(event: &StreamResponse, output: &mut String) {
        match event {
            StreamResponse::Message(msg) => {
                if let Some(text) = msg.text_content() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&text);
                }
            }
            StreamResponse::StatusUpdate(update) => {
                // Extract text from the status message if available.
                if let Some(ref msg) = update.status.message
                    && let Some(text) = msg.text_content()
                {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&text);
                }
            }
            StreamResponse::ArtifactUpdate(update) => {
                // Extract text from artifact parts.
                for part in &update.artifact.parts {
                    if let Some(text) = part.as_text() {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push_str(text);
                    }
                }
            }
            StreamResponse::Task(task) => {
                // Extract text from the task status message.
                if let Some(ref msg) = task.status.message
                    && let Some(text) = msg.text_content()
                {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&text);
                }
                // Also extract text from task artifacts.
                Self::collect_artifact_text(&task.artifacts, output);
            }
        }
    }

    /// Extract text from a list of artifacts into the output buffer.
    fn collect_artifact_text(artifacts: &[ra2a::types::Artifact], output: &mut String) {
        for artifact in artifacts {
            for part in &artifact.parts {
                if let Some(text) = part.as_text() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(text);
                }
            }
        }
    }

    /// Convert this A2A agent into a [`BoxedTool`] for use with a machi agent.
    ///
    /// The tool name is derived from the agent name, and the description
    /// from the agent card. When called, the tool sends the input text to
    /// the remote agent and returns the response.
    #[must_use]
    pub fn into_tool(self) -> BoxedTool {
        Box::new(A2aTool {
            agent: Arc::new(self),
        })
    }

    /// Get a reference to the underlying [`Client`](ra2a::client::Client) for advanced usage.
    #[must_use]
    pub fn client(&self) -> &Client {
        &self.client
    }
}

/// Bridges a remote A2A agent to machi's [`DynTool`] interface.
///
/// Created by [`A2aAgent::into_tool`].
struct A2aTool {
    agent: Arc<A2aAgent>,
}

impl std::fmt::Debug for A2aTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2aTool")
            .field("name", &self.agent.name)
            .finish()
    }
}

#[async_trait]
impl DynTool for A2aTool {
    fn name(&self) -> &str {
        &self.agent.name
    }

    fn description(&self) -> String {
        // We cannot await here, so use try_read for best-effort description.
        self.agent.card.try_read().map_or_else(
            |_| "A2A remote agent".to_owned(),
            |card| {
                let mut desc = card.description.clone();
                if !card.skills.is_empty() {
                    desc.push_str("\n\nSkills:");
                    for skill in &card.skills {
                        let _ = write!(desc, "\n- {}: {}", skill.name, skill.description);
                    }
                }
                desc
            },
        )
    }

    fn definition(&self) -> ToolDefinition {
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the remote agent"
                }
            },
            "required": ["message"]
        });
        ToolDefinition::new(&self.agent.name, self.description(), params)
    }

    async fn call_json(&self, args: Value) -> Result<Value, ToolError> {
        let text = match args.get("message").and_then(|v| v.as_str()) {
            Some(t) => t.to_owned(),
            None => {
                return Err(ToolError::InvalidArguments(
                    "Missing required field 'message' (string)".into(),
                ));
            }
        };

        let response = self.agent.send(text).await?;
        Ok(Value::String(response))
    }
}
