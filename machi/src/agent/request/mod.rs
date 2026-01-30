//! Request handling for agent prompts.
//!
//! This module provides request builders for both synchronous and streaming
//! agent interactions. The builders support configurable multi-turn conversations,
//! chat history management, and hook integration.

pub mod prompt;
pub mod streaming;

pub use prompt::{CancelSignal, PromptRequest, PromptResponse};
pub use streaming::{
    FinalResponse, MultiTurnStreamItem, StreamingPromptRequest, StreamingResult, stream_to_stdout,
};
