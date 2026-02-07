//! Prelude module for convenient imports.
//!
//! This module re-exports commonly used types and traits for easy access.
//!
//! # Usage
//!
//! ```rust,ignore
//! use machi::prelude::*;
//! ```

#[cfg(feature = "a2a")]
pub use crate::a2a::{A2aAgent, A2aAgentBuilder};
pub use crate::agent::{
    Agent, AgentError, Instructions, NextStep, OutputSchema, RunConfig, RunEvent, RunResult,
    Runner, StepInfo, ToolCallRecord, ToolCallRequest, UserInput,
};
pub use crate::audio::{
    AudioFormat, AudioProvider, SpeechRequest, SpeechResponse, SpeechToTextProvider,
    TextToSpeechProvider, TimestampGranularity, TranscriptionRequest, TranscriptionResponse,
    TranscriptionResponseFormat, TranscriptionSegment, TranscriptionWord, Voice,
};
pub use crate::callback::{
    AgentHooks, BoxedAgentHooks, BoxedRunHooks, LoggingAgentHooks, LoggingRunHooks, NoopAgentHooks,
    NoopRunHooks, RunContext, RunHooks, SharedAgentHooks, SharedRunHooks,
};
pub use crate::chat::{
    ChatProvider, ChatProviderExt, ChatRequest, ChatResponse, ResponseFormat, SharedChatProvider,
    ToolChoice,
};
pub use crate::embedding::{
    Embedding, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    EncodingFormat,
};
pub use crate::error::{Error, Result};
pub use crate::guardrail::{
    GuardrailOutput, InputGuardrail, InputGuardrailCheck, InputGuardrailResult, OutputGuardrail,
    OutputGuardrailCheck, OutputGuardrailResult,
};
pub use crate::llms::LlmError;
#[cfg(feature = "ollama")]
pub use crate::llms::{Ollama, OllamaConfig};
#[cfg(feature = "openai")]
pub use crate::llms::{OpenAI, OpenAIConfig};
#[cfg(feature = "mcp")]
pub use crate::mcp::{HttpBuilder, McpServer, StdioBuilder};
#[cfg(feature = "memory-sqlite")]
pub use crate::memory::SqliteSession;
pub use crate::memory::{InMemorySession, MemoryError, Session, SharedSession};
pub use crate::message::{
    Annotation, Content, ContentPart, FunctionCall, ImageDetail, ImageMime, InputAudio, Message,
    MessageBuilder, Role, ThinkingBlock, ToolCall,
};
pub use crate::stream::{StopReason, StreamAggregator, StreamChunk};
pub use crate::tool::{
    AlwaysDenyHandler, AutoApproveHandler, BoxedConfirmationHandler, BoxedTool,
    ConfirmationHandler, DynTool, SharedConfirmationHandler, Tool, ToolCallResult,
    ToolConfirmationRequest, ToolConfirmationResponse, ToolDefinition, ToolError,
    ToolExecutionPolicy, ToolResult,
};
#[cfg(feature = "toolkit")]
pub use crate::tools::{
    BingProvider, BraveProvider, DuckDuckGoProvider, EditFileTool, ExecTool, ListDirTool,
    ReadFileTool, SearchProvider, SearxngProvider, TavilyProvider, WebSearchTool, WriteFileTool,
};
pub use crate::usage::{CompletionTokensDetails, PromptTokensDetails, Usage};
#[cfg(feature = "wallet")]
pub use crate::wallet::{EvmWallet, EvmWalletBuilder, WalletError};
#[cfg(feature = "derive")]
pub use machi_derive::tool;
