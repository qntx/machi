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

pub use crate::agent::{
    Agent, Instructions, ManagedAgentTool, NextStep, RunConfig, RunEvent, RunResult, Runner,
    StepInfo, ToolCallRecord, ToolCallRequest, UserInput,
};
pub use crate::callback::{
    AgentHooks, BoxedAgentHooks, BoxedRunHooks, LoggingAgentHooks, LoggingRunHooks, NoopAgentHooks,
    NoopRunHooks, RunContext, RunHooks, SharedAgentHooks, SharedRunHooks,
};
pub use crate::error::{Error, LlmError, Result, ToolError};

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
#[cfg(feature = "memory-sqlite")]
pub use crate::memory::SqliteSession;
pub use crate::memory::{BoxedSession, InMemorySession, MemoryError, Session, SharedSession};
pub use crate::message::{
    Annotation, Content, ContentPart, FunctionCall, ImageDetail, ImageMime, InputAudio, Message,
    MessageAggregator, MessageBuilder, MessageDelta, Role, ThinkingBlock, ToolCall, ToolCallDelta,
};
pub use crate::stream::{StopReason, StreamAggregator, StreamChunk};
pub use crate::tool::{
    AlwaysDenyHandler, AutoApproveHandler, BoxedConfirmationHandler, BoxedTool,
    ConfirmationHandler, DynTool, SharedConfirmationHandler, Tool, ToolBox, ToolCallResult,
    ToolConfirmationRequest, ToolConfirmationResponse, ToolDefinition, ToolExecutionPolicy,
    ToolResult, ToolType,
};
pub use crate::usage::{CompletionTokensDetails, PromptTokensDetails, Usage, UsageTracker};
