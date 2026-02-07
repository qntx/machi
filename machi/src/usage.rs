//! Token usage tracking for LLM operations.
//!
//! This module provides types for tracking token consumption across
//! different LLM providers with a unified interface.
//!
//! # OpenAI API Alignment
//!
//! The `Usage` struct aligns with OpenAI's usage object:
//! - `prompt_tokens` / `completion_tokens` / `total_tokens`
//! - `prompt_tokens_details` (cached_tokens, audio_tokens)
//! - `completion_tokens_details` (reasoning_tokens, audio_tokens, prediction tokens)

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

/// Detailed breakdown of prompt/input tokens.
///
/// # OpenAI API Alignment
/// Maps to `prompt_tokens_details` in the API response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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
/// # OpenAI API Alignment
/// Maps to `completion_tokens_details` in the API response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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
/// # OpenAI API Alignment
///
/// This struct maps to OpenAI's usage object in API responses:
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    mod prompt_tokens_details {
        use super::*;

        #[test]
        fn default_is_zero() {
            let details = PromptTokensDetails::default();
            assert_eq!(details.cached_tokens, 0);
            assert_eq!(details.audio_tokens, 0);
        }

        #[test]
        fn serde_roundtrip() {
            let details = PromptTokensDetails {
                cached_tokens: 100,
                audio_tokens: 50,
            };
            let json = serde_json::to_string(&details).unwrap();
            let parsed: PromptTokensDetails = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, details);
        }

        #[test]
        fn serde_default_on_missing_fields() {
            let json = r"{}";
            let details: PromptTokensDetails = serde_json::from_str(json).unwrap();
            assert_eq!(details.cached_tokens, 0);
            assert_eq!(details.audio_tokens, 0);
        }

        #[test]
        fn copy_trait() {
            let details = PromptTokensDetails {
                cached_tokens: 10,
                audio_tokens: 5,
            };
            let copy = details;
            assert_eq!(details, copy);
        }
    }

    mod completion_tokens_details {
        use super::*;

        #[test]
        fn default_is_zero() {
            let details = CompletionTokensDetails::default();
            assert_eq!(details.reasoning_tokens, 0);
            assert_eq!(details.audio_tokens, 0);
            assert_eq!(details.accepted_prediction_tokens, 0);
            assert_eq!(details.rejected_prediction_tokens, 0);
        }

        #[test]
        fn serde_roundtrip() {
            let details = CompletionTokensDetails {
                reasoning_tokens: 100,
                audio_tokens: 50,
                accepted_prediction_tokens: 20,
                rejected_prediction_tokens: 10,
            };
            let json = serde_json::to_string(&details).unwrap();
            let parsed: CompletionTokensDetails = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, details);
        }

        #[test]
        fn serde_default_on_missing_fields() {
            let json = r#"{"reasoning_tokens": 50}"#;
            let details: CompletionTokensDetails = serde_json::from_str(json).unwrap();
            assert_eq!(details.reasoning_tokens, 50);
            assert_eq!(details.audio_tokens, 0);
        }

        #[test]
        fn copy_trait() {
            let details = CompletionTokensDetails {
                reasoning_tokens: 100,
                audio_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            };
            let copy = details;
            assert_eq!(details, copy);
        }
    }

    mod usage {
        use super::*;

        #[test]
        fn new_creates_usage() {
            let usage = Usage::new(100, 50);
            assert_eq!(usage.input_tokens, 100);
            assert_eq!(usage.output_tokens, 50);
            assert_eq!(usage.total_tokens, 150);
            assert!(usage.prompt_tokens_details.is_none());
            assert!(usage.completion_tokens_details.is_none());
        }

        #[test]
        fn zero_creates_empty_usage() {
            let usage = Usage::zero();
            assert_eq!(usage.input_tokens, 0);
            assert_eq!(usage.output_tokens, 0);
            assert_eq!(usage.total_tokens, 0);
        }

        #[test]
        fn with_prompt_details() {
            let details = PromptTokensDetails {
                cached_tokens: 20,
                audio_tokens: 10,
            };
            let usage = Usage::new(100, 50).with_prompt_details(details);
            assert_eq!(usage.prompt_tokens_details, Some(details));
        }

        #[test]
        fn with_completion_details() {
            let details = CompletionTokensDetails {
                reasoning_tokens: 30,
                audio_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            };
            let usage = Usage::new(100, 50).with_completion_details(details);
            assert_eq!(usage.completion_tokens_details, Some(details));
        }

        #[test]
        fn with_cached() {
            let usage = Usage::new(100, 50).with_cached(20);
            assert_eq!(usage.cached_tokens(), 20);
        }

        #[test]
        fn with_cached_preserves_existing_audio() {
            let usage = Usage::new(100, 50)
                .with_prompt_details(PromptTokensDetails {
                    cached_tokens: 0,
                    audio_tokens: 10,
                })
                .with_cached(20);
            assert_eq!(usage.cached_tokens(), 20);
            assert_eq!(usage.prompt_tokens_details.unwrap().audio_tokens, 10);
        }

        #[test]
        fn with_reasoning() {
            let usage = Usage::new(100, 50).with_reasoning(30);
            assert_eq!(usage.reasoning_tokens(), 30);
        }

        #[test]
        fn is_empty_returns_true_for_zero() {
            assert!(Usage::zero().is_empty());
        }

        #[test]
        fn is_empty_returns_false_for_non_zero() {
            assert!(!Usage::new(1, 0).is_empty());
        }

        #[test]
        fn cached_tokens_returns_zero_when_none() {
            let usage = Usage::new(100, 50);
            assert_eq!(usage.cached_tokens(), 0);
        }

        #[test]
        fn reasoning_tokens_returns_zero_when_none() {
            let usage = Usage::new(100, 50);
            assert_eq!(usage.reasoning_tokens(), 0);
        }

        #[test]
        fn audio_tokens_sums_input_and_output() {
            let usage = Usage::new(100, 50)
                .with_prompt_details(PromptTokensDetails {
                    cached_tokens: 0,
                    audio_tokens: 10,
                })
                .with_completion_details(CompletionTokensDetails {
                    reasoning_tokens: 0,
                    audio_tokens: 20,
                    accepted_prediction_tokens: 0,
                    rejected_prediction_tokens: 0,
                });
            assert_eq!(usage.audio_tokens(), 30);
        }

        #[test]
        fn audio_tokens_returns_zero_when_none() {
            let usage = Usage::new(100, 50);
            assert_eq!(usage.audio_tokens(), 0);
        }

        #[test]
        fn add_sums_tokens() {
            let a = Usage::new(100, 50);
            let b = Usage::new(200, 100);
            let c = a + b;

            assert_eq!(c.input_tokens, 300);
            assert_eq!(c.output_tokens, 150);
            assert_eq!(c.total_tokens, 450);
        }

        #[test]
        fn add_merges_prompt_details() {
            let a = Usage::new(100, 50).with_cached(10);
            let b = Usage::new(100, 50).with_cached(20);
            let c = a + b;
            assert_eq!(c.cached_tokens(), 30);
        }

        #[test]
        fn add_merges_prompt_details_one_none() {
            let a = Usage::new(100, 50).with_cached(10);
            let b = Usage::new(100, 50);
            let c = a + b;
            assert_eq!(c.cached_tokens(), 10);
        }

        #[test]
        fn add_merges_completion_details() {
            let a = Usage::new(100, 50).with_reasoning(10);
            let b = Usage::new(100, 50).with_reasoning(20);
            let c = a + b;
            assert_eq!(c.reasoning_tokens(), 30);
        }

        #[test]
        fn add_assign_works() {
            let mut usage = Usage::new(100, 50);
            usage += Usage::new(200, 100);
            assert_eq!(usage.input_tokens, 300);
            assert_eq!(usage.output_tokens, 150);
        }

        #[test]
        fn display_basic() {
            let usage = Usage::new(100, 50);
            let display = usage.to_string();
            assert!(display.contains("100"));
            assert!(display.contains("50"));
            assert!(display.contains("150"));
        }

        #[test]
        fn display_with_cached() {
            let usage = Usage::new(100, 50).with_cached(20);
            let display = usage.to_string();
            assert!(display.contains("cached: 20"));
        }

        #[test]
        fn display_with_reasoning() {
            let usage = Usage::new(100, 50).with_reasoning(30);
            let display = usage.to_string();
            assert!(display.contains("reasoning: 30"));
        }

        #[test]
        fn display_with_audio() {
            let usage = Usage::new(100, 50).with_prompt_details(PromptTokensDetails {
                cached_tokens: 0,
                audio_tokens: 10,
            });
            let display = usage.to_string();
            assert!(display.contains("audio: 10"));
        }

        #[test]
        fn serde_roundtrip() {
            let usage = Usage::new(100, 50).with_cached(20).with_reasoning(30);
            let json = serde_json::to_string(&usage).unwrap();
            let parsed: Usage = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, usage);
        }

        #[test]
        fn serde_alias_prompt_tokens() {
            let json = r#"{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}"#;
            let usage: Usage = serde_json::from_str(json).unwrap();
            assert_eq!(usage.input_tokens, 100);
            assert_eq!(usage.output_tokens, 50);
        }

        #[test]
        fn serde_skips_none_details() {
            let usage = Usage::new(100, 50);
            let json = serde_json::to_string(&usage).unwrap();
            assert!(!json.contains("prompt_tokens_details"));
            assert!(!json.contains("completion_tokens_details"));
        }

        #[test]
        fn default_is_zero() {
            let usage = Usage::default();
            assert!(usage.is_empty());
        }

        #[test]
        fn copy_trait() {
            let usage = Usage::new(100, 50);
            let copy = usage;
            assert_eq!(usage, copy);
        }
    }

    mod integration {
        use super::*;

        #[test]
        fn openai_response_parsing() {
            let json = r#"{
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 20,
                    "audio_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 10,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            }"#;

            let usage: Usage = serde_json::from_str(json).unwrap();
            assert_eq!(usage.input_tokens, 100);
            assert_eq!(usage.cached_tokens(), 20);
            assert_eq!(usage.reasoning_tokens(), 10);
        }

        #[test]
        fn reasoning_model_usage() {
            let usage = Usage::new(100, 500).with_reasoning(400).with_cached(50);

            assert_eq!(usage.reasoning_tokens(), 400);
            assert_eq!(usage.cached_tokens(), 50);

            let display = usage.to_string();
            assert!(display.contains("reasoning: 400"));
            assert!(display.contains("cached: 50"));
        }

        #[test]
        fn usage_accumulation_chain() {
            let usage1 = Usage::new(100, 50);
            let usage2 = Usage::new(200, 100);
            let usage3 = Usage::new(300, 150);

            let total = usage1 + usage2 + usage3;
            assert_eq!(total.input_tokens, 600);
            assert_eq!(total.output_tokens, 300);
            assert_eq!(total.total_tokens, 900);
        }
    }
}
