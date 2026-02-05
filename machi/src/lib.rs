//! Machi - A Rust framework for building AI agents
//!
//! This crate provides a lightweight, ergonomic framework for building AI agents
//! that can use tools and interact with language models.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Use OpenAI (requires OPENAI_API_KEY env var)
//!     let provider = OpenAI::from_env()?;
//!
//!     // Or use Ollama (local, no API key needed)
//!     // let provider = Ollama::with_defaults()?;
//!
//!     let request = ChatRequest::new(provider.default_model())
//!         .system("You are a helpful assistant.")
//!         .user("Hello!");
//!
//!     let response = provider.chat(&request).await?;
//!     println!("{}", response.text().unwrap_or_default());
//!
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! The framework is organized into several layers:
//!
//! - **Message Types** (`message`): Core types for chat messages
//! - **Provider Trait** (`provider`): Abstraction for LLM providers
//! - **LLM Backends** (`llms`): Implementations for OpenAI, Ollama, etc.
//! - **Tool System** (`tool`): Framework for defining and executing tools
//! - **Agent** (`agent`): High-level agent abstraction (coming soon)

pub mod agent;
pub mod audio;
pub mod chat;
pub mod embedding;
pub mod error;
pub mod llms;
pub mod message;
pub mod prelude;
pub mod stream;
pub mod tool;
pub mod usage;

pub use error::{Error, LlmError, Result, ToolError};

#[cfg(feature = "derive")]
pub use machi_derive::tool;
