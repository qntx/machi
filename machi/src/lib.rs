//! Machi - A Rust implementation of smolagents
//!
//! This crate provides a lightweight, ergonomic framework for building AI agents
//! that can use tools and interact with language models.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use machi::prelude::*;
//!
//! #[tool(description = "Add two numbers")]
//! async fn add(a: i32, b: i32) -> Result<i32, ToolError> {
//!     Ok(a + b)
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     // Use OpenAI (requires OPENAI_API_KEY env var)
//!     let model = OpenAIClient::from_env().completion_model("gpt-4o");
//!
//!     // Or use Ollama (local, no API key needed)
//!     // let model = OllamaClient::new().completion_model("qwen3");
//!
//!     let mut agent = Agent::builder()
//!         .model(model)
//!         .tool(Box::new(Add))
//!         .build();
//!
//!     let result = agent.run("What is 2 + 3?").await;
//! }
//! ```

pub mod agent;
pub mod callback;
pub mod error;
pub mod managed;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod memory;
pub mod message;
pub mod multimodal;
pub mod prompts;
pub mod providers;
pub mod tool;
#[cfg(feature = "toolkit")]
pub mod tools;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::agent::{
        Agent, AgentBuilder, AgentConfig, AgentStream, FinalAnswerChecks, RunOptions, RunResult,
        RunState, StreamEvent, StreamItem,
    };
    // Callback system (includes metrics and logging)
    pub use crate::callback::{
        // Async callbacks
        AsyncCallbackRegistry,
        AsyncCallbackRegistryBuilder,
        // Sync callbacks
        CallbackContext,
        CallbackRegistry,
        CallbackRegistryBuilder,
        // Metrics
        LoggingConfig,
        MetricsCollector,
        MetricsSnapshot,
        // Handlers
        Priority,
        RunMetrics,
        logging_handler,
        metrics_handler,
        tracing_handler,
    };
    pub use crate::error::{AgentError, Result};
    pub use crate::managed::{
        BoxedManagedAgent, ManagedAgent, ManagedAgentArgs, ManagedAgentInfo, ManagedAgentRegistry,
        ManagedAgentTool,
    };
    #[cfg(feature = "mcp")]
    pub use crate::mcp::McpClient;
    pub use crate::memory::{
        ActionStep, AgentMemory, MemoryStep, PlanningStep, TaskStep, ToolCall,
    };
    pub use crate::message::{ChatMessage, MessageContent, MessageRole};
    pub use crate::multimodal::{AgentAudio, AgentImage, AgentOutput, AudioFormat, ImageFormat};
    pub use crate::prompts::{PromptRender, PromptTemplates};
    pub use crate::providers::{
        anthropic::{
            AnthropicClient, CLAUDE_3_5_SONNET, CLAUDE_4_OPUS, CLAUDE_4_SONNET, CLAUDE_OPUS_4_5,
            CLAUDE_OPUS_4_5_LATEST, CLAUDE_SONNET_4_5, CLAUDE_SONNET_4_5_LATEST,
            CompletionModel as AnthropicModel,
        },
        common::{
            FromEnv, GenerateOptions, Model, ModelResponse, ModelStream, TokenUsage,
            model_requires_max_completion_tokens, model_supports_stop_parameter,
        },
        mock::MockModel,
        ollama::{
            CompletionModel as OllamaModel, DEEPSEEK_R1, LLAMA3_2, LLAMA3_3, MISTRAL, OllamaClient,
            QWEN2_5,
        },
        openai::{
            CompletionModel as OpenAIModel, GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO, GPT_4O,
            GPT_4O_MINI, GPT_5, GPT_5_MINI, O3, O3_MINI, O3_PRO, O4_MINI, OpenAIClient,
        },
    };
    pub use crate::tool::{
        BoxedTool, DynTool, Tool, ToolBox, ToolCallResult, ToolDefinition, ToolError, ToolResult,
    };
    #[cfg(feature = "toolkit")]
    pub use crate::tools::{FinalAnswerTool, UserInputTool, VisitWebpageTool, WebSearchTool};

    // Re-export derive macro
    #[cfg(feature = "derive")]
    pub use machi_derive::tool;
}

// Re-export commonly used items at crate root
pub use error::{AgentError, Result};
#[cfg(feature = "derive")]
pub use machi_derive::tool;
