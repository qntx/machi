//! Ollama API client and model implementations.
//!
//! This module provides integration with Ollama's local LLM server,
//! supporting models like Llama 3, Mistral, Qwen, `DeepSeek`, and more.
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::providers::ollama::OllamaClient;
//!
//! // Connect to local Ollama server (default: http://localhost:11434)
//! let client = OllamaClient::new();
//! let model = client.completion_model("llama3.3");
//!
//! let response = model.generate(messages, options).await?;
//! ```
//!
//! # Features
//!
//! - Local inference with no API key required
//! - OpenAI-compatible chat completions API
//! - Streaming support
//! - Tool/function calling (model dependent)

mod client;
mod completion;
mod streaming;

pub use client::*;
pub use completion::*;

/// Llama 3.3 70B model - latest Llama model.
pub const LLAMA3_3: &str = "llama3.3";
/// Llama 3.2 model.
pub const LLAMA3_2: &str = "llama3.2";
/// Llama 3.1 model.
pub const LLAMA3_1: &str = "llama3.1";
/// Llama 3.1 405B model - largest open model.
pub const LLAMA3_1_405B: &str = "llama3.1:405b";
/// Llama 3 8B model.
pub const LLAMA3: &str = "llama3";

/// Qwen 2.5 model - latest Qwen.
pub const QWEN2_5: &str = "qwen2.5";
/// Qwen 2.5 Coder model - optimized for coding.
pub const QWEN2_5_CODER: &str = "qwen2.5-coder";
/// Qwen 3 model - latest generation with tool calling.
pub const QWEN3: &str = "qwen3";
/// Qwen 3 8B model - balanced performance.
pub const QWEN3_8B: &str = "qwen3:8b";
/// `QwQ` model - reasoning model.
pub const QWQ: &str = "qwq";

/// `DeepSeek` V3 model.
pub const DEEPSEEK_V3: &str = "deepseek-v3";
/// `DeepSeek` R1 model - reasoning model.
pub const DEEPSEEK_R1: &str = "deepseek-r1";
/// `DeepSeek` Coder V2 model.
pub const DEEPSEEK_CODER_V2: &str = "deepseek-coder-v2";

/// Mistral model.
pub const MISTRAL: &str = "mistral";
/// Mistral Small model.
pub const MISTRAL_SMALL: &str = "mistral-small";
/// Mistral Nemo model.
pub const MISTRAL_NEMO: &str = "mistral-nemo";
/// Mixtral 8x7B model.
pub const MIXTRAL: &str = "mixtral";

/// `CodeLlama` model.
pub const CODELLAMA: &str = "codellama";
/// `StarCoder` 2 model.
pub const STARCODER2: &str = "starcoder2";

/// Gemma 2 model from Google.
pub const GEMMA2: &str = "gemma2";
/// Phi-3 model from Microsoft.
pub const PHI3: &str = "phi3";
/// Command R model from Cohere.
pub const COMMAND_R: &str = "command-r";
/// Nous Hermes 2 model.
pub const NOUS_HERMES2: &str = "nous-hermes2";
