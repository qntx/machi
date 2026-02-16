//! LLM backend implementations.
//!
//! This module contains implementations for various LLM providers.
//! Each backend is organized into its own submodule.
//!
//! # Available Backends
//!
//! - [`openai`] - `OpenAI` API (GPT-4o, GPT-4, etc.)
//! - [`ollama`] - Ollama local LLM server

pub mod error;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "ollama")]
pub mod ollama;

pub use error::LlmError;
#[cfg(feature = "ollama")]
pub use ollama::{Ollama, OllamaConfig};
#[cfg(feature = "openai")]
pub use openai::{OpenAI, OpenAIConfig};
