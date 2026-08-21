//! Pure shared types for the Ovo agent runtime kernel.
//!
//! This crate is intentionally free of network and provider dependencies so
//! orchestration, tools, and runtimes can share one model without pulling
//! HTTP stacks into the bottom of the DAG.

#![forbid(unsafe_code)]

pub mod deadline;
pub mod error;
pub mod id;
pub mod message;
pub mod usage;

pub use deadline::Deadline;
pub use error::{ErrorCode, OvoError, Result, RetryClass};
pub use id::{AgentId, RunId, SessionId, ToolCallId, WorkflowRunId};
pub use message::{ContentPart, ImageMime, Message, Role, ToolCall};
pub use usage::{CompletionTokensDetails, PromptTokensDetails, Usage};
