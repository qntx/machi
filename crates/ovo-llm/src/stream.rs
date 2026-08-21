//! Streaming sample events.

use std::pin::Pin;

use futures::Stream;
use ovo_types::{Message, Usage};

/// One event in a streaming sample.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SampleEvent {
    /// Incremental assistant text.
    TextDelta {
        /// Delta text.
        text: String,
    },
    /// Incremental reasoning / chain-of-thought channel (when providers split channels).
    ReasoningDelta {
        /// Delta text.
        text: String,
    },
    /// Incremental tool-call argument fragment.
    ToolCallDelta {
        /// Tool call index within the assistant message.
        index: u32,
        /// Partial JSON / text args.
        arguments_delta: String,
    },
    /// Full tool-call snapshot (emitted once when known, or after deltas coalesce).
    ToolCalls {
        /// Complete assistant message carrying tool calls.
        message: Message,
    },
    /// First response bytes / headers observed.
    ResponseStarted {
        /// Cache-read tokens if known at start.
        cache_read: Option<u32>,
        /// Cache-creation tokens if known at start.
        cache_creation: Option<u32>,
    },
    /// Decorator is about to retry a failed sample.
    Retrying {
        /// 1-based attempt about to run.
        attempt: u32,
        /// Short reason (`rate_limited`, `http_503`, …).
        reason: String,
    },
    /// Usage totals (may arrive at end).
    Usage(Usage),
    /// Stream completed successfully with a final message.
    Completed {
        /// Final assistant message.
        message: Message,
        /// Provider stop reason.
        stop_reason: Option<String>,
    },
    /// Stream failed after partial progress (terminal).
    Failed {
        /// Error message.
        message: String,
    },
}

/// Opaque pinned sample event stream.
pub type SampleStream = Pin<Box<dyn Stream<Item = SampleEvent> + Send>>;
