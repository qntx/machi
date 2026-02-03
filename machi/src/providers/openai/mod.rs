//! `OpenAI` API client and model implementations.
//!
//! This module provides integration with `OpenAI`'s Chat Completions API,
//! supporting models like GPT-4o, GPT-4, GPT-3.5-turbo, and compatible APIs.
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::providers::openai::OpenAIClient;
//!
//! let client = OpenAIClient::from_env();
//! let model = client.completion_model("gpt-4o");
//!
//! let response = model.generate(messages, options).await?;
//! ```

mod client;
mod completion;
mod streaming;

pub use client::*;
pub use completion::*;

/// GPT-5 model identifier - latest flagship model with reasoning capabilities.
pub const GPT_5: &str = "gpt-5";
/// GPT-5 mini model identifier - smaller, faster version of GPT-5.
pub const GPT_5_MINI: &str = "gpt-5-mini";
/// GPT-5.1 model identifier - improved version of GPT-5.
pub const GPT_5_1: &str = "gpt-5.1";
/// GPT-5.2 model identifier - smarter and more precise responses.
pub const GPT_5_2: &str = "gpt-5.2";
/// GPT-5.1 Codex model identifier - optimized for agentic coding.
pub const GPT_5_1_CODEX: &str = "gpt-5.1-codex";

/// GPT-4.1 model identifier - smartest non-reasoning model.
pub const GPT_4_1: &str = "gpt-4.1";
/// GPT-4.1 with specific version.
pub const GPT_4_1_2025_04_14: &str = "gpt-4.1-2025-04-14";
/// GPT-4.1 mini model identifier - smaller, faster version.
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// GPT-4.1 mini with specific version.
pub const GPT_4_1_MINI_2025_04_14: &str = "gpt-4.1-mini-2025-04-14";
/// GPT-4.1 nano model identifier - fastest, most cost-efficient.
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// GPT-4.1 nano with specific version.
pub const GPT_4_1_NANO_2025_04_14: &str = "gpt-4.1-nano-2025-04-14";

/// O3 model identifier - reasoning model for complex tasks.
pub const O3: &str = "o3";
/// O3 Pro model identifier - uses more compute for better reasoning.
pub const O3_PRO: &str = "o3-pro";
/// O3 mini model identifier - fast reasoning model.
pub const O3_MINI: &str = "o3-mini";
/// O4 mini model identifier - fast, cost-efficient reasoning model.
pub const O4_MINI: &str = "o4-mini";
/// O4 mini with specific version.
pub const O4_MINI_2025_04_16: &str = "o4-mini-2025-04-16";
/// O1 model identifier - original reasoning model.
pub const O1: &str = "o1";
/// O1 mini model identifier.
pub const O1_MINI: &str = "o1-mini";
/// O1 Pro model identifier - enhanced reasoning.
pub const O1_PRO: &str = "o1-pro";

/// GPT-4o model identifier (legacy - being retired Feb 13, 2026).
pub const GPT_4O: &str = "gpt-4o";
/// GPT-4o mini model identifier (legacy).
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// GPT-4 Turbo model identifier (legacy).
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// GPT-4 model identifier (legacy).
pub const GPT_4: &str = "gpt-4";
/// GPT-3.5 Turbo model identifier (legacy).
pub const GPT_3_5_TURBO: &str = "gpt-3.5-turbo";
