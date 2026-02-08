//! Prelude module for convenient imports.
//!
//! Re-exports the most commonly used types and traits so you can get
//! started with a single `use` statement.
//!
//! ```rust
//! use machi::prelude::*;
//!
//! let request = ChatRequest::new("gpt-4o")
//!     .system("You are helpful.")
//!     .user("Hello!");
//!
//! let agent = Agent::new("assistant")
//!     .instructions("You are helpful.")
//!     .model("gpt-4o");
//! ```

#[cfg(feature = "a2a")]
pub use crate::a2a::{A2aAgent, A2aAgentBuilder};
pub use crate::agent::{
    Agent, AgentError, Instructions, OutputSchema, RunConfig, RunEvent, RunResult, Runner,
    StepInfo, ToolCallRecord, UserInput,
};
pub use crate::audio::{
    AudioFormat, SpeechRequest, SpeechResponse, SpeechToTextProvider, TextToSpeechProvider,
    TimestampGranularity, TranscriptionRequest, TranscriptionResponse, TranscriptionResponseFormat,
    TranscriptionSegment, TranscriptionWord, Voice,
};
pub use crate::callback::{
    AgentHooks, BoxedAgentHooks, BoxedRunHooks, LogLevel, LoggingAgentHooks, LoggingRunHooks,
    NoopAgentHooks, NoopRunHooks, RunContext, RunHooks, SharedAgentHooks, SharedRunHooks,
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
pub use crate::memory::{BoxedSession, InMemorySession, MemoryError, Session, SharedSession};
pub use crate::message::{
    Annotation, Content, ContentPart, FunctionCall, ImageDetail, ImageMime, InputAudio, Message,
    Role, ThinkingBlock, ToolCall,
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
pub use crate::usage::Usage;
#[cfg(feature = "wallet")]
pub use crate::wallet::{
    DerivationStyle, DerivedAddress, EvmChain, EvmWallet, HdWallet, Wallet, WalletError,
};
#[cfg(feature = "derive")]
pub use machi_derive::tool;
