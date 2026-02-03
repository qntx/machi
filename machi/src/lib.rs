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
//!     let model = OpenAIModel::new("gpt-4");
//!     let agent = ToolCallingAgent::builder()
//!         .model(model)
//!         .tools(vec![Box::new(Add)])
//!         .build();
//!
//!     let result = agent.run("What is 2 + 3?").await;
//! }
//! ```

pub mod agent;
pub mod callback;
pub mod error;
pub mod memory;
pub mod message;
pub mod model;
pub mod prompts;
pub mod tool;
pub mod tools;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::agent::{Agent, AgentConfig, CodeAgent, RunResult, ToolCallingAgent};
    pub use crate::callback::{CallbackManager, StepEvent};
    pub use crate::error::{AgentError, Result};
    pub use crate::memory::{
        ActionStep, AgentMemory, MemoryStep, PlanningStep, TaskStep, TokenUsage, ToolCall,
    };
    pub use crate::message::{ChatMessage, MessageContent, MessageRole};
    pub use crate::model::{AnthropicModel, GenerateOptions, Model, ModelResponse, OpenAIModel};
    pub use crate::prompts::PromptTemplates;
    pub use crate::tool::{BoxedTool, DynTool, Tool, ToolBox, ToolDefinition, ToolError};
    pub use crate::tools::{FinalAnswerTool, UserInputTool, VisitWebpageTool, WebSearchTool};

    // Re-export derive macro
    pub use machi_derive::tool;
}

// Re-export commonly used items at crate root
pub use error::{AgentError, Result};
pub use machi_derive::tool;
