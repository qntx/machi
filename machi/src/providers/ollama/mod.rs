//! Ollama API client and Machi integration
//!
//! # Example
//! ```rust
//! use crate::providers::ollama;
//!
//! // Create a new Ollama client (defaults to http://localhost:11434)
//! let client = ollama::Client::new();
//!
//! // Create a completion model interface using, for example, the "llama3.2" model
//! let comp_model = client.completion_model("llama3.2");
//!
//! // Create an embedding interface using the "all-minilm" model
//! let emb_model = ollama::Client::new().embedding_model("all-minilm");
//!
//! // Also create an agent if needed
//! let agent = client.agent("llama3.2");
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod message;

pub use client::*;
pub use completion::{
    CompletionModel, CompletionResponse, StreamingCompletionResponse, ToolDefinition,
    LLAMA3_2, LLAVA, MISTRAL,
};
pub use embedding::{EmbeddingModel, EmbeddingResponse, ALL_MINILM, NOMIC_EMBED_TEXT};
pub use message::*;
