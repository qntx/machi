//! Conversation compaction strategies.
//!
//! Strategies are pure transforms over message lists. Runtime hosts decide
//! *when* to compact; this crate decides *how*.

#![forbid(unsafe_code)]

pub mod max_messages;
pub mod prune;
pub mod select;
pub mod strategy;
pub mod token_threshold;

pub use max_messages::MaxMessages;
pub use prune::{DropPrefix, PruneToolResults, StripImages, SummarizingCompaction};
pub use select::{
    CompactionRange, apply_range, is_safe_split, select_compaction_range, snap_split_forward,
    tool_pair_invariant_holds,
};
pub use strategy::{CompactionOutcome, CompactionStrategy};
pub use token_threshold::TokenThreshold;
