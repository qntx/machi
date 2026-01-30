//! Agent module for building and executing LLM agents.
//!
//! This module provides the [`Agent`] struct and its builder for creating
//! configurable LLM agents that combine a completion model with a system prompt,
//! context documents, and tools.
//!
//! # Overview
//!
//! An [`Agent`] represents an LLM model combined with:
//! - A **preamble** (system prompt)
//! - **Static context** documents always provided to the agent
//! - **Dynamic context** documents fetched via RAG at prompt-time
//! - **Tools** that the agent can call
//! - **Model parameters** (temperature, max_tokens, etc.)
//!
//! # Example: Basic Agent
//!
//! ```rust,ignore
//! use machi::{
//!     completion::{Chat, Completion, Prompt},
//!     providers::openai,
//! };
//!
//! let openai = openai::Client::from_env();
//!
//! // Configure the agent
//! let agent = openai.agent("gpt-4o")
//!     .preamble("System prompt")
//!     .context("Context document 1")
//!     .context("Context document 2")
//!     .tool(tool1)
//!     .tool(tool2)
//!     .temperature(0.8)
//!     .additional_params(json!({"foo": "bar"}))
//!     .build();
//!
//! // Use the agent for completions and prompts
//! let chat_response = agent.chat("Prompt", chat_history)
//!     .await
//!     .expect("Failed to chat with Agent");
//!
//! let prompt_response = agent.prompt("Prompt")
//!     .await
//!     .expect("Failed to prompt the Agent");
//!
//! // Generate a completion request builder for more control
//! let completion_req_builder = agent.completion("Prompt", chat_history)
//!     .await
//!     .expect("Failed to create completion request builder");
//!
//! let response = completion_req_builder
//!     .temperature(0.9)  // Override the agent's temperature
//!     .send()
//!     .await
//!     .expect("Failed to send completion request");
//! ```
//!
//! # Example: RAG Agent
//!
//! ```rust,ignore
//! use machi::{
//!     completion::Prompt,
//!     embeddings::EmbeddingsBuilder,
//!     providers::openai,
//!     vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
//! };
//!
//! let openai = openai::Client::from_env();
//! let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
//!
//! // Create and populate vector store
//! let mut vector_store = InMemoryVectorStore::default();
//! let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
//!     .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien")
//!     .simple_document("doc1", "Definition of a *glarb-glarb*: An ancient farming tool")
//!     .build()
//!     .await
//!     .expect("Failed to build embeddings");
//!
//! vector_store.add_documents(embeddings).await.expect("Failed to add documents");
//!
//! // Create index and agent
//! let index = vector_store.index(embedding_model);
//! let agent = openai.agent(openai::GPT_4O)
//!     .preamble("You are a dictionary assistant.")
//!     .dynamic_context(1, index)
//!     .build();
//!
//! let response = agent.prompt("What does \"glarb-glarb\" mean?")
//!     .await
//!     .expect("Failed to prompt the agent");
//! ```

mod as_tool;
mod builder;
mod core;
pub mod error;
mod hook;
pub mod request;

// Re-export core types
pub use builder::{AgentBuilder, AgentBuilderSimple};
pub use core::Agent;
pub use error::StreamingError;
pub use hook::{PromptHook, StreamingPromptHook, ToolCallHookAction};

// Re-export request types
pub use request::{
    CancelSignal, FinalResponse, MultiTurnStreamItem, PromptRequest, PromptResponse,
    StreamingPromptRequest, StreamingResult, stream_to_stdout,
};

// Re-export text type for convenience
pub use crate::completion::message::Text;
