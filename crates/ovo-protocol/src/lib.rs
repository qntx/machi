//! Wire-agnostic protocol types shared across tools, LLM adapters, and hosts.
//!
//! This crate sits above [`ovo_types`] in the DAG and must not depend on
//! HTTP clients, provider SDKs, or runtime I/O.

#![forbid(unsafe_code)]

pub mod content;
pub mod event;
pub mod observability;
pub mod tokens;
pub mod tool_id;

pub use content::{ContentBlock, ImageBlock};
pub use event::{TurnEvent, TurnEventKind};
pub use observability::{
    SPAN_COMPACT, SPAN_SAMPLE, SPAN_SESSION, SPAN_SPAWN, SPAN_TOOL, SPAN_TOOL_BATCH, SPAN_TURN,
    SPAN_WORKFLOW, SPAN_WORKFLOW_HOST, field, required_span_names, span_catalogue_snapshot,
};
pub use tokens::{
    IMAGE_TOKEN_COST, MESSAGE_FRAME_TOKENS, PreflightOverflow, check_context_overflow,
    estimate_image_tokens, estimate_text_tokens,
};
pub use tool_id::ToolId;
