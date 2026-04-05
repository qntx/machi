//! A lightweight, ergonomic framework for building AI agents in Rust.
//!
//! Machi provides the building blocks for constructing AI agents that reason,
//! use tools, and collaborate — powered by any LLM backend.
//!
//! # Core Concepts
//!
//! - **[`Agent`](agent::Agent)** — A self-contained unit with its own LLM provider,
//!   instructions, tools, and optional sub-agents.
//! - **[`Runner`](agent::Runner)** — A stateless execution engine driving the
//!   `ReAct` loop (think → act → observe → repeat).
//! - **[`Tool`](tool::Tool) / [`DynTool`](tool::DynTool)** — Capabilities that
//!   agents can invoke (filesystem, shell, web search, or custom).
//! - **[`ChatProvider`](chat::ChatProvider)** — Trait abstracting over LLM backends
//!   (`OpenAI`, Ollama, or custom).
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `openai` | `OpenAI` API backend |
//! | `ollama` | Ollama local LLM backend |
//! | `derive` | `#[tool]` proc-macro for deriving tools |
//! | `toolkit` | Built-in filesystem, shell, and web search tools |
//! | `mcp` | Model Context Protocol server integration |
//! | `a2a` | Agent-to-Agent protocol support |
//! | `memory-sqlite` | SQLite-backed session persistence |
//! | `schema` | Structured output via JSON Schema generation |
//! | `full` | All of the above (default) |
//!
//! # Quick Start
//!
//! ```rust
//! use machi::agent::{Agent, RunConfig};
//! use machi::chat::ChatRequest;
//! use machi::message::Message;
//!
//! // Build a chat request
//! let request = ChatRequest::new("gpt-4o")
//!     .system("You are a helpful assistant.")
//!     .user("Hello!")
//!     .temperature(0.7);
//!
//! // Configure an agent
//! let agent = Agent::new("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .model("gpt-4o");
//!
//! // Construct messages manually
//! let msgs = vec![
//!     Message::system("You are helpful."),
//!     Message::user("What is Rust?"),
//! ];
//! ```

#[cfg(feature = "a2a")]
pub mod a2a;
pub mod agent;
#[allow(
    clippy::module_name_repetitions,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
pub mod context;
pub mod audio;
pub mod chat;
pub mod embedding;
pub mod error;
pub mod guardrail;
pub mod hooks;
pub mod llms;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod memory;
#[allow(
    clippy::module_name_repetitions,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
pub mod middleware;
pub mod message;
pub mod prelude;
pub mod stream;
pub mod tool;
#[cfg(feature = "toolkit")]
pub mod tools;
pub mod usage;

pub use error::{Error, Result};
#[cfg(feature = "derive")]
pub use machi_derive::tool;
pub use tool::ToolError;
