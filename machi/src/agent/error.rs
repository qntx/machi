//! Error types for the agent module.

use crate::completion::{CompletionError, PromptError};
use crate::tool::ToolSetError;
use thiserror::Error;

/// Errors that can occur during streaming agent operations.
#[derive(Debug, Error)]
pub enum StreamingError {
    /// Error during completion request.
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),

    /// Error during prompt execution.
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),

    /// Error during tool execution.
    #[error("ToolSetError: {0}")]
    Tool(#[from] ToolSetError),
}
