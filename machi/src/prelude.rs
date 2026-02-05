//! Prelude module for convenient imports.
//!
//! This module re-exports commonly used types and traits for easy access.
//!
//! # Usage
//!
//! ```rust,ignore
//! use machi::prelude::*;
//! ```

#[cfg(feature = "ollama")]
pub use crate::llms::{Ollama, OllamaConfig};
#[cfg(feature = "openai")]
pub use crate::llms::{OpenAI, OpenAIConfig};

#[cfg(feature = "derive")]
pub use machi_derive::tool;

pub use crate::error::{Error, LlmError, Result, ToolError};

pub use crate::agent::{Agent, AgentBuilder, AgentConfig, AgentMemory, RunResult, RunState};
pub use crate::audio::{
    AudioFormat, AudioProvider, SpeechRequest, SpeechResponse, SpeechToTextProvider,
    TextToSpeechProvider, TimestampGranularity, TranscriptionRequest, TranscriptionResponse,
    TranscriptionResponseFormat, TranscriptionSegment, TranscriptionWord, Voice,
};
pub use crate::chat::{
    BoxedChatProvider, ChatProvider, ChatProviderExt, ChatRequest, ChatResponse, ResponseFormat,
    SharedChatProvider, ToolChoice,
};
pub use crate::embedding::{
    Embedding, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    EncodingFormat,
};
pub use crate::message::{
    Annotation, Content, ContentPart, FunctionCall, ImageDetail, ImageMime, InputAudio, Message,
    MessageAggregator, MessageBuilder, MessageDelta, Role, ThinkingBlock, ToolCall, ToolCallDelta,
};
pub use crate::stream::{StopReason, StreamAggregator, StreamChunk};
pub use crate::tool::{
    AlwaysDenyHandler, AutoApproveHandler, BoxedConfirmationHandler, BoxedTool,
    ConfirmationHandler, DynTool, Tool, ToolBox, ToolCallResult, ToolConfirmationRequest,
    ToolConfirmationResponse, ToolDefinition, ToolExecutionPolicy, ToolResult, ToolType,
};
pub use crate::usage::{CompletionTokensDetails, PromptTokensDetails, Usage, UsageTracker};
