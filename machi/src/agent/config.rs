//! Agent configuration types.

/// Configuration for an agent.
#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    /// Maximum number of steps (default: 20).
    #[doc(hidden)]
    pub max_steps: usize,
    /// Planning interval (run planning every N steps).
    pub planning_interval: Option<usize>,
    /// Agent name.
    pub name: Option<String>,
    /// Agent description.
    pub description: Option<String>,
    /// Whether to provide a run summary when acting as a managed agent.
    pub provide_run_summary: Option<bool>,
    /// Maximum number of concurrent tool calls (default: unlimited).
    ///
    /// When multiple tool calls are returned by the model, they can be executed
    /// in parallel up to this limit. Set to `Some(1)` to force sequential execution.
    /// Set to `None` for unlimited parallelism.
    pub max_parallel_tool_calls: Option<usize>,
}

impl AgentConfig {
    /// Default maximum number of steps for agent execution.
    pub const DEFAULT_MAX_STEPS: usize = 20;

    /// Create a new config with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_steps: Self::DEFAULT_MAX_STEPS,
            planning_interval: None,
            name: None,
            description: None,
            provide_run_summary: None,
            max_parallel_tool_calls: None,
        }
    }
}
