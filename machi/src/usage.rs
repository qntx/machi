//! Token usage tracking for LLM operations.
//!
//! This module provides types for tracking token consumption across
//! different LLM providers with a unified interface.
//!
//! # `OpenAI` API Alignment
//!
//! The `Usage` struct aligns with `OpenAI`'s usage object:
//! - `prompt_tokens` / `completion_tokens` / `total_tokens`
//! - `prompt_tokens_details` (`cached_tokens`, `audio_tokens`)
//! - `completion_tokens_details` (`reasoning_tokens`, `audio_tokens`, prediction tokens)

use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

/// Detailed breakdown of prompt/input tokens.
///
/// # `OpenAI` API Alignment
/// Maps to `prompt_tokens_details` in the API response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromptTokensDetails {
    /// Cached tokens that were reused (prompt caching).
    #[serde(default)]
    pub cached_tokens: u32,

    /// Audio tokens in the input.
    #[serde(default)]
    pub audio_tokens: u32,
}

/// Detailed breakdown of completion/output tokens.
///
/// # `OpenAI` API Alignment
/// Maps to `completion_tokens_details` in the API response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionTokensDetails {
    /// Reasoning tokens (for o1/o3 models).
    #[serde(default)]
    pub reasoning_tokens: u32,

    /// Audio tokens in the output.
    #[serde(default)]
    pub audio_tokens: u32,

    /// Accepted prediction tokens (Predicted Outputs feature).
    #[serde(default)]
    pub accepted_prediction_tokens: u32,

    /// Rejected prediction tokens (Predicted Outputs feature).
    #[serde(default)]
    pub rejected_prediction_tokens: u32,
}

/// Token usage statistics from an LLM operation.
///
/// # `OpenAI` API Alignment
///
/// This struct maps to `OpenAI`'s usage object in API responses:
/// ```json
/// {
///     "prompt_tokens": 100,
///     "completion_tokens": 50,
///     "total_tokens": 150,
///     "prompt_tokens_details": { "cached_tokens": 0, "audio_tokens": 0 },
///     "completion_tokens_details": { "reasoning_tokens": 0, ... }
/// }
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    /// Number of tokens in the input/prompt.
    #[serde(default, alias = "prompt_tokens")]
    pub input_tokens: u32,

    /// Number of tokens in the output/completion.
    #[serde(default, alias = "completion_tokens")]
    pub output_tokens: u32,

    /// Total tokens used (input + output).
    #[serde(default)]
    pub total_tokens: u32,

    /// Detailed breakdown of prompt tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,

    /// Detailed breakdown of completion tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

impl Usage {
    /// Create a new usage record.
    #[must_use]
    pub const fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    /// Create an empty usage record.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    /// Set prompt tokens details.
    #[must_use]
    pub const fn with_prompt_details(mut self, details: PromptTokensDetails) -> Self {
        self.prompt_tokens_details = Some(details);
        self
    }

    /// Set completion tokens details.
    #[must_use]
    pub const fn with_completion_details(mut self, details: CompletionTokensDetails) -> Self {
        self.completion_tokens_details = Some(details);
        self
    }

    /// Set cached tokens (convenience method).
    #[must_use]
    pub fn with_cached(mut self, cached: u32) -> Self {
        let details = self.prompt_tokens_details.unwrap_or_default();
        self.prompt_tokens_details = Some(PromptTokensDetails {
            cached_tokens: cached,
            ..details
        });
        self
    }

    /// Set reasoning tokens (convenience method).
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: u32) -> Self {
        let details = self.completion_tokens_details.unwrap_or_default();
        self.completion_tokens_details = Some(CompletionTokensDetails {
            reasoning_tokens: reasoning,
            ..details
        });
        self
    }

    /// Check if usage is empty (no tokens used).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total_tokens == 0
    }

    /// Get cached tokens count.
    #[must_use]
    pub const fn cached_tokens(&self) -> u32 {
        match &self.prompt_tokens_details {
            Some(d) => d.cached_tokens,
            None => 0,
        }
    }

    /// Get reasoning tokens count.
    #[must_use]
    pub const fn reasoning_tokens(&self) -> u32 {
        match &self.completion_tokens_details {
            Some(d) => d.reasoning_tokens,
            None => 0,
        }
    }

    /// Get audio tokens count (input + output).
    #[must_use]
    pub const fn audio_tokens(&self) -> u32 {
        let input = match &self.prompt_tokens_details {
            Some(d) => d.audio_tokens,
            None => 0,
        };
        let output = match &self.completion_tokens_details {
            Some(d) => d.audio_tokens,
            None => 0,
        };
        input + output
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Merge prompt details
        let prompt_details = match (self.prompt_tokens_details, rhs.prompt_tokens_details) {
            (Some(a), Some(b)) => Some(PromptTokensDetails {
                cached_tokens: a.cached_tokens + b.cached_tokens,
                audio_tokens: a.audio_tokens + b.audio_tokens,
            }),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };

        // Merge completion details
        let completion_details = match (
            self.completion_tokens_details,
            rhs.completion_tokens_details,
        ) {
            (Some(a), Some(b)) => Some(CompletionTokensDetails {
                reasoning_tokens: a.reasoning_tokens + b.reasoning_tokens,
                audio_tokens: a.audio_tokens + b.audio_tokens,
                accepted_prediction_tokens: a.accepted_prediction_tokens
                    + b.accepted_prediction_tokens,
                rejected_prediction_tokens: a.rejected_prediction_tokens
                    + b.rejected_prediction_tokens,
            }),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };

        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            output_tokens: self.output_tokens + rhs.output_tokens,
            total_tokens: self.total_tokens + rhs.total_tokens,
            prompt_tokens_details: prompt_details,
            completion_tokens_details: completion_details,
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Usage(in: {}, out: {}, total: {})",
            self.input_tokens, self.output_tokens, self.total_tokens
        )?;
        let cached = self.cached_tokens();
        if cached > 0 {
            write!(f, " [cached: {cached}]")?;
        }
        let reasoning = self.reasoning_tokens();
        if reasoning > 0 {
            write!(f, " [reasoning: {reasoning}]")?;
        }
        let audio = self.audio_tokens();
        if audio > 0 {
            write!(f, " [audio: {audio}]")?;
        }
        Ok(())
    }
}
