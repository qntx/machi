//! Prelude module — core traits and types for quick setup.
//!
//! ```rust
//! use machi::prelude::*;
//! ```

#[cfg(feature = "derive")]
pub use machi_derive::tool;

pub use crate::agent::{
    Agent, AgentError, RunConfig, RunEvent, RunResult, Runner, StepInfo, ToolCallRecord, UserInput,
};
pub use crate::chat::{
    ChatProvider, ChatRequest, ChatResponse, ResponseFormat, SharedChatProvider,
};
pub use crate::error::{Error, Result};
pub use crate::guardrail::{
    GuardrailOutput, InputGuardrail, InputGuardrailCheck, OutputGuardrail, OutputGuardrailCheck,
};
pub use crate::hooks::{Hooks, LoggingHooks, NoopHooks, RunContext, SharedHooks};
#[cfg(feature = "ollama")]
pub use crate::llms::{Ollama, OllamaConfig};
#[cfg(feature = "openai")]
pub use crate::llms::{OpenAI, OpenAIConfig};
pub use crate::message::{ContentPart, Message, Role};
pub use crate::stream::{StopReason, StreamChunk};
pub use crate::tool::{
    BoxedTool, ConfirmationHandler, DynTool, Tool, ToolConfirmationRequest,
    ToolConfirmationResponse, ToolDefinition, ToolError, ToolExecutionPolicy, ToolResult,
};
pub use crate::usage::Usage;
