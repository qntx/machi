//! Agent module — core abstractions for building AI agents.
//!
//! This module implements a **Runner-driven, managed-agent** architecture that
//! combines the best ideas from OpenAI Agents SDK and HuggingFace smolagents:
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
//! ```rust,ignore
//! use machi::agent::{Agent, RunConfig};
//!
//! let agent = Agent::new("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .model("gpt-4o")
//!     .provider(openai_provider.clone());
//!
//! let result = agent.run("Hello!", RunConfig::default()).await?;
//! println!("{}", result.text().unwrap_or("no output"));
//! ```
//!
//! # Heterogeneous Multi-Agent
//!
//! ```rust,ignore
//! let researcher = Agent::new("researcher")
//!     .instructions("You research topics thoroughly.")
//!     .model("gpt-4o")
//!     .provider(openai_provider.clone());
//!
//! let writer = Agent::new("writer")
//!     .instructions("You write clear summaries.")
//!     .model("claude-sonnet")
//!     .provider(claude_provider.clone());
//!
//! let orchestrator = Agent::new("orchestrator")
//!     .instructions("Delegate research and writing tasks to your team.")
//!     .model("gpt-4o")
//!     .provider(openai_provider.clone())
//!     .managed_agent(researcher)
//!     .managed_agent(writer);
//!
//! // The orchestrator's LLM can call "researcher" and "writer" as tools.
//! // Each sub-agent uses its own provider (GPT-4o / Claude respectively).
//! let result = orchestrator.run("Write about Rust", RunConfig::default()).await?;
//! ```

mod config;
pub mod error;
mod hook;
pub mod result;
mod runner;

pub use config::{Agent, Instructions, OutputSchema};
pub use error::AgentError;
pub use result::{
    NextStep, RunConfig, RunEvent, RunResult, StepInfo, ToolCallRecord, ToolCallRequest, UserInput,
};
pub use runner::Runner;
