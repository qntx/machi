//! Error types for the completion module.

use super::message::Message;
use crate::http as http_client;
use crate::tool::ToolSetError;
use crate::tool::server::ToolServerError;
use thiserror::Error;

/// Errors that can occur during completion operations.
#[derive(Debug, Error)]
pub enum CompletionError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Url error (e.g.: invalid URL)
    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[cfg(not(target_family = "wasm"))]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + 'static>),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the completion model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),
}

/// Errors that can occur during prompting operations.
#[derive(Debug, Error)]
pub enum PromptError {
    /// Something went wrong with the completion
    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),

    /// There was an error while using a tool
    #[error("ToolCallError: {0}")]
    ToolError(#[from] ToolSetError),

    /// There was an issue while executing a tool on a tool server
    #[error("ToolServerError: {0}")]
    ToolServerError(#[from] ToolServerError),

    /// The LLM tried to call too many tools during a multi-turn conversation.
    /// To fix this, you may either need to lower the amount of tools your model has access to
    /// (and then create other agents to share the tool load)
    /// or increase the amount of turns given in `.multi_turn()`.
    #[error("MaxDepthError: (reached limit: {max_depth})")]
    MaxDepthError {
        max_depth: usize,
        chat_history: Box<Vec<Message>>,
        prompt: Box<Message>,
    },

    /// A prompting loop was cancelled.
    #[error("PromptCancelled: {reason}")]
    PromptCancelled {
        chat_history: Box<Vec<Message>>,
        reason: String,
    },
}

impl PromptError {
    pub(crate) fn prompt_cancelled(chat_history: Vec<Message>, reason: &str) -> Self {
        Self::PromptCancelled {
            chat_history: Box::new(chat_history),
            reason: reason.to_string(),
        }
    }
}

/// Error type to represent issues with converting messages to and from specific provider messages.
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Message conversion error: {0}")]
    ConversionError(String),
}

impl From<MessageError> for CompletionError {
    fn from(error: MessageError) -> Self {
        CompletionError::RequestError(error.into())
    }
}
