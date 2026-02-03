//! Anthropic API client and model implementations.
//!
//! This module provides integration with Anthropic's Messages API,
//! supporting Claude models like Claude 3.5 Sonnet, Claude 3 Opus, etc.
//!
//! # Example
//!
//! ```rust,ignore
//! use machi::providers::anthropic::AnthropicClient;
//!
//! let client = AnthropicClient::from_env();
//! let model = client.completion_model("claude-3-5-sonnet-latest");
//!
//! let response = model.generate(messages, options).await?;
//! ```

mod client;
mod completion;
mod streaming;

pub use client::*;
pub use completion::*;

/// Claude Sonnet 4.5 - best balance of intelligence, speed, and cost.
/// Recommended starting model for most use cases.
pub const CLAUDE_SONNET_4_5: &str = "claude-sonnet-4-5-20250929";
/// Claude Sonnet 4.5 alias - points to latest snapshot.
pub const CLAUDE_SONNET_4_5_LATEST: &str = "claude-sonnet-4-5-latest";
/// Claude Opus 4.5 - most capable model for complex tasks.
pub const CLAUDE_OPUS_4_5: &str = "claude-opus-4-5-20260120";
/// Claude Opus 4.5 alias - points to latest snapshot.
pub const CLAUDE_OPUS_4_5_LATEST: &str = "claude-opus-4-5-latest";

/// Claude 4 Opus model identifier.
pub const CLAUDE_4_OPUS: &str = "claude-opus-4-0";
/// Claude 4 Sonnet model identifier.
pub const CLAUDE_4_SONNET: &str = "claude-sonnet-4-0";
/// Claude 4 Haiku model identifier - fast and cost-efficient.
pub const CLAUDE_4_HAIKU: &str = "claude-haiku-4-0";

/// Claude 3.7 Sonnet model identifier.
pub const CLAUDE_3_7_SONNET: &str = "claude-3-7-sonnet-latest";
/// Claude 3.7 Sonnet with specific version.
pub const CLAUDE_3_7_SONNET_20250219: &str = "claude-3-7-sonnet-20250219";

/// Claude 3.5 Sonnet model identifier (latest alias).
pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-latest";
/// Claude 3.5 Sonnet with specific version.
pub const CLAUDE_3_5_SONNET_20241022: &str = "claude-3-5-sonnet-20241022";
/// Claude 3.5 Haiku model identifier (latest alias).
pub const CLAUDE_3_5_HAIKU: &str = "claude-3-5-haiku-latest";
/// Claude 3.5 Haiku with specific version.
pub const CLAUDE_3_5_HAIKU_20241022: &str = "claude-3-5-haiku-20241022";

/// Claude 3 Opus model identifier (retired Oct 2025).
pub const CLAUDE_3_OPUS: &str = "claude-3-opus-20240229";
/// Claude 3 Sonnet model identifier (retired Jul 2025).
pub const CLAUDE_3_SONNET: &str = "claude-3-sonnet-20240229";
/// Claude 3 Haiku model identifier.
pub const CLAUDE_3_HAIKU: &str = "claude-3-haiku-20240307";

/// Anthropic API version 2023-06-01.
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
/// Latest Anthropic API version.
pub const ANTHROPIC_VERSION_LATEST: &str = ANTHROPIC_VERSION_2023_06_01;

/// Beta header for 1M token context window.
pub const BETA_CONTEXT_1M: &str = "context-1m-2025-08-07";
/// Beta header for prompt caching.
pub const BETA_PROMPT_CACHING: &str = "prompt-caching-2024-07-31";
/// Beta header for computer use.
pub const BETA_COMPUTER_USE: &str = "computer-use-2024-10-22";
