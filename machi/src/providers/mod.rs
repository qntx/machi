//! LLM Provider implementations for various model APIs.
//!
//! This module provides a unified interface for interacting with different LLM providers.
//! Each provider implements the [`Model`] trait, enabling seamless switching between providers.
//!
//! # Supported Providers
//!
//! - **`OpenAI`**: GPT-5, GPT-4.1, O3/O4 series, and compatible APIs
//! - **Anthropic**: Claude 4.5, Claude 4, Claude 3.5, and other Claude models
//! - **Ollama**: Local LLM inference (Llama, Qwen, Mistral, `DeepSeek`, etc.)
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::providers::openai::OpenAIClient;
//! use machi::providers::anthropic::AnthropicClient;
//! use machi::providers::ollama::OllamaClient;
//!
//! // Create an OpenAI client
//! let openai = OpenAIClient::from_env();
//! let gpt5 = openai.completion_model("gpt-5");
//!
//! // Create an Anthropic client
//! let anthropic = AnthropicClient::from_env();
//! let claude = anthropic.completion_model("claude-sonnet-4-5-latest");
//!
//! // Create an Ollama client (local, no API key needed)
//! let ollama = OllamaClient::new();
//! let llama = ollama.completion_model("llama3.3");
//! ```

pub mod anthropic;
pub mod common;
pub mod ollama;
pub mod openai;

pub use common::*;

// Re-export main client types for convenience
pub use anthropic::AnthropicClient;
pub use ollama::OllamaClient;
pub use openai::OpenAIClient;
