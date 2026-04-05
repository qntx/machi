//! Context management strategies for controlling message history size.
//!
//! Long-running agents accumulate messages that can exceed the LLM's context
//! window. A [`ContextStrategy`] compacts the message list before each LLM
//! call, keeping the conversation within budget.
//!
//! # Built-in Strategies
//!
//! | Strategy | Description |
//! |----------|-------------|
//! | [`NoCompaction`] | Pass-through — no compaction (default) |
//! | [`SlidingWindow`] | Keep only the last *N* messages |
//! | [`TokenBudget`] | Drop oldest non-system messages to fit a token budget |
//!
//! # Custom Strategies
//!
//! Implement [`ContextStrategy`] for advanced compaction (e.g., LLM-based
//! summarisation of older messages):
//!
//! ```rust,ignore
//! use async_trait::async_trait;
//! use machi::context::ContextStrategy;
//! use machi::message::Message;
//!
//! struct SummarizeOld;
//!
//! #[async_trait]
//! impl ContextStrategy for SummarizeOld {
//!     async fn compact(&self, messages: &[Message]) -> machi::Result<Vec<Message>> {
//!         // Call an LLM to summarise older messages …
//!         Ok(messages.to_vec())
//!     }
//! }
//! ```

use async_trait::async_trait;

use crate::Result;
use crate::message::Message;

/// A shared, thread-safe [`ContextStrategy`] trait object.
pub type SharedContextStrategy = std::sync::Arc<dyn ContextStrategy>;

/// Strategy for compacting the message list before an LLM call.
///
/// Implementations receive the full message list and return a (possibly
/// shortened) list. The runner replaces the working messages with the
/// compacted result before building the [`ChatRequest`](crate::chat::ChatRequest).
///
/// # Contract
///
/// - The returned list **must** preserve the first system message (if any)
///   so the agent's instructions are never lost.
/// - The returned list **must** end with the most recent user or tool message
///   so the LLM can continue the conversation coherently.
/// - The strategy is called on every step, so it should be efficient.
#[async_trait]
#[allow(clippy::module_name_repetitions)]
pub trait ContextStrategy: Send + Sync {
    /// Compact the message list.
    async fn compact(&self, messages: &[Message]) -> Result<Vec<Message>>;
}

// ---------------------------------------------------------------------------
// Built-in strategies
// ---------------------------------------------------------------------------

/// No-op strategy — returns messages unchanged (the default).
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct NoCompaction;

#[async_trait]
impl ContextStrategy for NoCompaction {
    async fn compact(&self, messages: &[Message]) -> Result<Vec<Message>> {
        Ok(messages.to_vec())
    }
}

/// Keep only the system message(s) and the last `max_messages` non-system messages.
///
/// This is a simple sliding window that preserves the agent's instructions
/// while bounding the conversation length.
///
/// # Example
///
/// ```rust
/// use machi::context::SlidingWindow;
///
/// let strategy = SlidingWindow::new(20); // keep last 20 messages + system
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SlidingWindow {
    /// Maximum number of non-system messages to retain.
    pub max_messages: usize,
}

impl SlidingWindow {
    /// Create a new sliding window strategy.
    #[must_use]
    pub const fn new(max_messages: usize) -> Self {
        Self { max_messages }
    }
}

#[async_trait]
impl ContextStrategy for SlidingWindow {
    async fn compact(&self, messages: &[Message]) -> Result<Vec<Message>> {
        // Split into system prefix and rest.
        let system_count = messages
            .iter()
            .take_while(|m| m.role == crate::message::Role::System)
            .count();

        let (system, rest) = messages.split_at(system_count);

        if rest.len() <= self.max_messages {
            return Ok(messages.to_vec());
        }

        let keep_from = rest.len() - self.max_messages;
        let mut result = system.to_vec();
        result.extend_from_slice(&rest[keep_from..]);
        Ok(result)
    }
}

/// Drop oldest non-system messages until the estimated token count fits
/// within `max_tokens`.
///
/// Token estimation uses a simple heuristic (~4 characters per token).
/// For precise counting, implement a custom [`ContextStrategy`] with a
/// proper tokeniser.
///
/// # Example
///
/// ```rust
/// use machi::context::TokenBudget;
///
/// let strategy = TokenBudget::new(120_000); // ~120k tokens
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TokenBudget {
    /// Maximum estimated token count.
    pub max_tokens: usize,
}

impl TokenBudget {
    /// Create a new token budget strategy.
    #[must_use]
    pub const fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Rough token estimate for a message (~4 chars per token).
    fn estimate_tokens(msg: &Message) -> usize {
        let text_len = msg.text().map_or(0, |s| s.len());
        // Add overhead for role, tool ids, etc.
        (text_len / 4).max(1) + 4
    }
}

#[async_trait]
impl ContextStrategy for TokenBudget {
    async fn compact(&self, messages: &[Message]) -> Result<Vec<Message>> {
        let system_count = messages
            .iter()
            .take_while(|m| m.role == crate::message::Role::System)
            .count();

        let (system, rest) = messages.split_at(system_count);

        let system_tokens: usize = system.iter().map(Self::estimate_tokens).sum();
        let budget = self.max_tokens.saturating_sub(system_tokens);

        // Walk from the end, accumulating tokens.
        let mut kept = Vec::new();
        let mut used = 0usize;
        for msg in rest.iter().rev() {
            let tokens = Self::estimate_tokens(msg);
            if used + tokens > budget && !kept.is_empty() {
                break;
            }
            used += tokens;
            kept.push(msg.clone());
        }
        kept.reverse();

        let mut result = system.to_vec();
        result.extend(kept);
        Ok(result)
    }
}
