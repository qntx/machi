//! Agent module — core abstractions for building AI agents.
//!
//! This module implements a **Runner-driven, managed-agent** architecture that
//! combines the best ideas from `OpenAI` Agents SDK and `HuggingFace` smolagents:
//!
//! - **[`Agent`]** is a self-contained unit with its own LLM provider, enabling
//!   heterogeneous multi-agent systems where each agent uses a different model.
//! - **[`Runner`]** is a stateless execution engine that drives the agent through
//!   a ReAct-style reasoning loop (think → act → observe → repeat).
//! - **Managed agents** are sub-agents registered via [`Agent::managed_agent`],
//!   dispatched inline by the Runner as parallel tool calls — inspired by smolagents.
//!
//! # Quick Start
//!
//! ```rust
//! use machi::agent::{Agent, RunConfig};
//!
//! let agent = Agent::new("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .model("gpt-4o");
//!
//! assert_eq!(agent.name(), "assistant");
//! assert_eq!(agent.get_model(), "gpt-4o");
//! ```
//!
//! # Heterogeneous Multi-Agent
//!
//! ```rust
//! use machi::agent::Agent;
//!
//! let researcher = Agent::new("researcher")
//!     .instructions("You research topics thoroughly.")
//!     .model("gpt-4o");
//!
//! let writer = Agent::new("writer")
//!     .instructions("You write clear summaries.")
//!     .model("claude-sonnet");
//!
//! let orchestrator = Agent::new("orchestrator")
//!     .instructions("Delegate research and writing tasks to your team.")
//!     .model("gpt-4o")
//!     .managed_agent(researcher)
//!     .managed_agent(writer);
//!
//! assert_eq!(orchestrator.name(), "orchestrator");
//! ```

mod config;
pub mod error;
pub mod result;
mod runner;

pub use config::{Agent, Instructions, OutputSchema};
pub use error::AgentError;
pub use result::{
    NextStep, RunConfig, RunEvent, RunResult, StepInfo, ToolCallRecord, ToolCallRequest, UserInput,
};
pub use runner::Runner;
