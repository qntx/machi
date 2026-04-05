//! Ollama API client implementation.
//!
//! This module provides a client for the Ollama local LLM server, supporting:
//! - Chat completions (synchronous and streaming)
//! - Text embeddings

mod chat;
mod client;
mod config;
mod embedding;
mod stream;

pub use client::Ollama;
#[allow(clippy::module_name_repetitions)]
pub use config::OllamaConfig;
